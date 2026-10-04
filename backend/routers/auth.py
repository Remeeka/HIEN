import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db import models, schemas
from backend.services import auth_service

logger = logging.getLogger("hien.auth_router")

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@router.post("/register", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def register_endpoint(
    req: schemas.UserRegisterReq,
    db: Session = Depends(get_db)
):
    """
    Register a new user in PostgreSQL with bcrypt password hashing and canonical role.
    """
    if not req.email or not req.password or not req.full_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Full name, email, and password are required."
        )

    if req.confirm_password and req.password != req.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password confirmation does not match."
        )

    if len(req.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long."
        )

    clean_email = req.email.strip().lower()
    existing_user = db.query(models.User).filter(
        models.User.email.ilike(clean_email)
    ).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email address already exists."
        )

    username_seed = clean_email.split("@")[0].replace(".", "_")
    existing_username = db.query(models.User).filter(models.User.username == username_seed).first()
    username = username_seed if not existing_username else f"{username_seed}_{uuid.uuid4().hex[:4]}"

    assigned_role = auth_service.normalize_role(req.role or "Return Inspection Officer")

    user_id = f"usr-{uuid.uuid4().hex[:8]}"
    new_user = models.User(
        id=user_id,
        username=username,
        email=clean_email,
        full_name=req.full_name.strip(),
        password_hash=auth_service.hash_password(req.password),
        role=assigned_role,
        assigned_hub_id=req.assigned_hub_id or "LOC001",
        is_active=True,
        avatar_url=""
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info(f"Registered new user: {new_user.email} ({new_user.id}) with role: {new_user.role}")
    return schemas.UserOut.model_validate(new_user)

@router.post("/login", response_model=schemas.TokenResponse)
def login_endpoint(
    req: schemas.UserLoginReq,
    db: Session = Depends(get_db)
):
    """
    Authenticate user via Email or Username + Password + Role validation.
    Returns signed Bearer JWT and user profile.
    """
    identifier = req.username_or_email or req.email or req.username
    if not identifier or not req.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please select a role and enter your credentials."
        )

    user = auth_service.get_user_by_identifier(db, identifier)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials. Please check your username and password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not auth_service.verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials. Please check your username and password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive. Please contact system administrator.",
        )

    # Role validation if specified at login
    if req.role:
        expected_role = auth_service.normalize_role(req.role)
        actual_role = auth_service.normalize_role(user.role)
        if expected_role.lower() != actual_role.lower():
            logger.warning(
                f"Role mismatch on login for {user.email}: selected '{expected_role}', actual '{actual_role}'"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Role mismatch. This account is not assigned to the selected enterprise role.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    canonical_role = auth_service.normalize_role(user.role)

    # Generate JWT Token
    token_data = {
        "sub": user.id,
        "username": user.username,
        "email": user.email,
        "role": canonical_role
    }
    access_token = auth_service.create_access_token(token_data)

    return schemas.TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=schemas.UserOut.model_validate(user)
    )

@router.get("/me", response_model=schemas.UserOut)
def get_current_user_profile(
    current_user: models.User = Depends(auth_service.get_current_active_user)
):
    """
    Get profile of the currently authenticated active user from Bearer JWT.
    """
    return schemas.UserOut.model_validate(current_user)

@router.post("/logout")
def logout_endpoint(
    current_user: Optional[models.User] = Depends(auth_service.get_current_user)
):
    """
    Logout active session.
    """
    return {"status": "success", "message": "Successfully logged out."}

@router.get("/users", response_model=List[schemas.UserOut])
def list_system_users(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth_service.get_current_active_user)
):
    """
    Retrieve directory of system users (Requires active authenticated user).
    """
    users = db.query(models.User).order_by(models.User.created_at).all()
    return [schemas.UserOut.model_validate(u) for u in users]
