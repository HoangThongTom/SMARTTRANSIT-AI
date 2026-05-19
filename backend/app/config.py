"""
SmartTransit AI — Database Configuration & Connection
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from loguru import logger

# ─────────────────────────────────────────────────────────────
# ENVIRONMENT CONFIG
# ─────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://smartuser:smartpass123@localhost:5432/smarttransit"
)

MODEL_PATH  = os.getenv("MODEL_PATH",  "ai_core/models/traffic_model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "ai_core/models/scaler.joblib")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG    = os.getenv("DEBUG", "false").lower() == "true"


# ─────────────────────────────────────────────────────────────
# SQLALCHEMY SETUP
# ─────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,   # Tự reconnect nếu connection chết
    echo=DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────
# DEPENDENCY INJECTION (FastAPI)
# ─────────────────────────────────────────────────────────────
def get_db():
    """FastAPI dependency: tự đóng session sau khi dùng."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """Kiểm tra kết nối DB khi khởi động."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.success("✅ Database connection OK")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False
