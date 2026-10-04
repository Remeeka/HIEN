import os
import sys
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool

load_dotenv()

# Read Database URL from environment with PostgreSQL as default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/hien")

Base = declarative_base()

def get_engine():
    """
    Create SQLAlchemy Engine with resilience:
    Attempts PostgreSQL connection first; if unreachable during local development,
    falls back cleanly to SQLite without crashing the application.
    """
    db_url = DATABASE_URL
    is_sqlite = db_url.startswith("sqlite")
    
    try:
        if is_sqlite:
            engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False},
                poolclass=NullPool
            )
        else:
            engine = create_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800,
                poolclass=QueuePool
            )
        # Test connection immediately
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine, db_url
    except Exception as e:
        print(f"⚠️ Primary Database Connection ({db_url.split('@')[-1] if '@' in db_url else db_url}) failed: {e}")
        print("⚡ Engaging Local Resilient SQLite Database: sqlite:///./hien.db")
        fallback_url = "sqlite:///./hien.db"
        engine = create_engine(
            fallback_url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool
        )
        return engine, fallback_url

engine, ACTIVE_DB_URL = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database session with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def check_db_health() -> dict:
    """Return live database health and connection info."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_type = "PostgreSQL" if "postgresql" in str(engine.url) else "SQLite (Local Resilient)"
            return {
                "status": "connected",
                "database_engine": db_type,
                "url_dialect": str(engine.dialect.name),
                "healthy": True
            }
    except Exception as e:
        return {
            "status": "disconnected",
            "error": str(e),
            "healthy": False
        }

def init_db():
    """Initialize all tables defined in models."""
    from backend.db import models
    Base.metadata.create_all(bind=engine)
