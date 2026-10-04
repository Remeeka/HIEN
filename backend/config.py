import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/hien")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Authentication & Security
    AUTH_SECRET_KEY: str = os.getenv("AUTH_SECRET_KEY", "hien-production-secure-jwt-secret-key-2026-hyperlocal-exchange")
    AUTH_ALGORITHM: str = os.getenv("AUTH_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    
    # Admin Seed Credentials
    AUTH_ADMIN_EMAIL: str = os.getenv("AUTH_ADMIN_EMAIL", "admin@hien.io")
    AUTH_ADMIN_PASSWORD: str = os.getenv("AUTH_ADMIN_PASSWORD", "Admin@12345")
    AUTH_ADMIN_USERNAME: str = os.getenv("AUTH_ADMIN_USERNAME", "admin")

settings = Settings()
