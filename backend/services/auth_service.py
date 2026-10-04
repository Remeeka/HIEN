import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Union
import bcrypt
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy import or_

from backend.config import settings
from backend.db.database import get_db, SessionLocal
from backend.db import models, schemas

logger = logging.getLogger("hien.auth")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

def hash_password(password: str) -> str:
    """Hash plaintext password securely using bcrypt with salt."""
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plaintext password against bcrypt hash."""
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        return bcrypt.checkpw(pwd_bytes, hashed_password.encode("utf-8"))
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generate cryptographically signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, settings.AUTH_SECRET_KEY, algorithm=settings.AUTH_ALGORITHM)

def decode_access_token(token: str) -> dict:
    """Decode and validate JWT access token signature and expiration."""
    try:
        payload = jwt.decode(token, settings.AUTH_SECRET_KEY, algorithms=[settings.AUTH_ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_user_by_identifier(db: Session, identifier: str) -> Optional[models.User]:
    """Retrieve user by username or email case-insensitively."""
    ident = identifier.strip().lower()
    return db.query(models.User).filter(
        or_(
            models.User.email.ilike(ident),
            models.User.username.ilike(ident)
        )
    ).first()

def authenticate_user(db: Session, identifier: str, password: str) -> Optional[models.User]:
    """Authenticate user with username/email and password."""
    user = get_user_by_identifier(db, identifier)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user

CANONICAL_ROLES = [
    "Platform Administrator",
    "Return Inspection Officer",
    "Store / Hub Manager",
    "Logistics Operator",
    "Analytics Viewer",
]

ROLE_SLUG_MAP = {
    "platform administrator": "Platform Administrator",
    "admin": "Platform Administrator",
    "administrator": "Platform Administrator",
    "platform_administrator": "Platform Administrator",

    "return inspection officer": "Return Inspection Officer",
    "inspector": "Return Inspection Officer",
    "return inspector": "Return Inspection Officer",
    "inspection officer": "Return Inspection Officer",
    "return_inspection_officer": "Return Inspection Officer",

    "store / hub manager": "Store / Hub Manager",
    "store/hub manager": "Store / Hub Manager",
    "hub manager": "Store / Hub Manager",
    "hub_manager": "Store / Hub Manager",
    "store_manager": "Store / Hub Manager",
    "manager": "Store / Hub Manager",

    "logistics operator": "Logistics Operator",
    "logistics": "Logistics Operator",
    "operator": "Logistics Operator",
    "logistics_operator": "Logistics Operator",

    "analytics viewer": "Analytics Viewer",
    "analytics_viewer": "Analytics Viewer",
    "analyst": "Analytics Viewer",
    "viewer": "Analytics Viewer",
}

def normalize_role(role: Optional[str]) -> str:
    """Normalize role string to canonical display name."""
    if not role:
        return "Return Inspection Officer"
    cleaned = role.strip().lower()
    return ROLE_SLUG_MAP.get(cleaned, role.strip())

def seed_initial_users(db: Optional[Session] = None) -> None:
    """
    Seed initial system users and enterprise demo profiles with secure password hashes
    in PostgreSQL.
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        # Default Seed User Profiles with Exact Demo Credentials
        seed_users = [
            {
                "id": "usr-admin-01",
                "username": "admin",
                "email": "admin@hien.io",
                "password": "Admin@123",
                "full_name": "Admin User",
                "role": "Platform Administrator",
                "assigned_hub_id": "LOC001",
                "avatar_url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=120&auto=format&fit=crop&q=80"
            },
            {
                "id": "usr-inspector-01",
                "username": "inspector",
                "email": "inspector@hien.io",
                "password": "Inspector@123",
                "full_name": "Return Inspection Officer",
                "role": "Return Inspection Officer",
                "assigned_hub_id": "LOC001",
                "avatar_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=120&auto=format&fit=crop&q=80"
            },
            {
                "id": "usr-manager-01",
                "username": "manager",
                "email": "manager@hien.io",
                "password": "Manager@123",
                "full_name": "Store / Hub Manager",
                "role": "Store / Hub Manager",
                "assigned_hub_id": "LOC002",
                "avatar_url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80"
            },
            {
                "id": "usr-logistics-01",
                "username": "logistics",
                "email": "logistics@hien.io",
                "password": "Logistics@123",
                "full_name": "Logistics Operator",
                "role": "Logistics Operator",
                "assigned_hub_id": "LOC003",
                "avatar_url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=120&auto=format&fit=crop&q=80"
            },
            {
                "id": "usr-analytics-01",
                "username": "analytics",
                "email": "analytics@hien.io",
                "password": "Analytics@123",
                "full_name": "Analytics Viewer",
                "role": "Analytics Viewer",
                "assigned_hub_id": "LOC001",
                "avatar_url": "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=120&auto=format&fit=crop&q=80"
            },
            {
                "id": "usr-marcus-01",
                "username": "marcus.vance",
                "email": "marcus.vance@hien.io",
                "password": "Inspector@123",
                "full_name": "Marcus Vance",
                "role": "Return Inspection Officer",
                "assigned_hub_id": "LOC001",
                "avatar_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=120&auto=format&fit=crop&q=80"
            },
            {
                "id": "usr-elena-01",
                "username": "elena.rostova",
                "email": "elena.rostova@hien.io",
                "password": "Manager@123",
                "full_name": "Elena Rostova",
                "role": "Store / Hub Manager",
                "assigned_hub_id": "LOC002",
                "avatar_url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80"
            },
            {
                "id": "usr-david-01",
                "username": "david.kim",
                "email": "david.kim@hien.io",
                "password": "Logistics@123",
                "full_name": "David Kim",
                "role": "Logistics Operator",
                "assigned_hub_id": "LOC003",
                "avatar_url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=120&auto=format&fit=crop&q=80"
            },
            {
                "id": "usr-aris-01",
                "username": "aris.pendelton",
                "email": "pendelton@hien.io",
                "password": "Analytics@123",
                "full_name": "Dr. Aris Pendelton",
                "role": "Analytics Viewer",
                "assigned_hub_id": "LOC001",
                "avatar_url": "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=120&auto=format&fit=crop&q=80"
            }
        ]

        for u_data in seed_users:
            existing = db.query(models.User).filter(
                or_(
                    models.User.email == u_data["email"],
                    models.User.username == u_data["username"]
                )
            ).first()

            canonical_role = normalize_role(u_data["role"])
            if not existing:
                new_user = models.User(
                    id=u_data["id"],
                    username=u_data["username"],
                    email=u_data["email"],
                    password_hash=hash_password(u_data["password"]),
                    full_name=u_data["full_name"],
                    role=canonical_role,
                    assigned_hub_id=u_data.get("assigned_hub_id"),
                    avatar_url=u_data.get("avatar_url", ""),
                    is_active=True
                )
                db.add(new_user)
                logger.info(f"Seeded user: {u_data['username']} ({canonical_role})")
            else:
                existing.role = canonical_role
                existing.password_hash = hash_password(u_data["password"])
                if u_data.get("full_name"):
                    existing.full_name = u_data["full_name"]

        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error seeding users: {e}")
    finally:
        if close_db:
            db.close()

def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> models.User:
    """Dependency: Extract, decode JWT and retrieve active user."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        # Fallback query by username if sub was username
        user = db.query(models.User).filter(models.User.username == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return user

def get_current_active_user(
    current_user: models.User = Depends(get_current_user)
) -> models.User:
    """Ensure user is active."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def require_roles(allowed_roles: List[str]):
    """Role-based authorization dependency factory with normalized role matching."""
    norm_allowed = [normalize_role(r).lower() for r in allowed_roles]
    def role_checker(current_user: models.User = Depends(get_current_active_user)):
        user_norm = normalize_role(current_user.role).lower()
        if user_norm not in norm_allowed and "platform administrator" not in user_norm and "admin" not in user_norm:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {allowed_roles}"
            )
        return current_user
    return role_checker
