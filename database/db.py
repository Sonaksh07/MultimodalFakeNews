import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database.models import Base
from configs.logging_config import logger

DB_PATH = Path("database/fakenews_system.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH.absolute()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

_initialized = False

def init_db():
    """Initializes the database tables. Idempotent and only logs once per
    process -- Streamlit reruns the whole script on every interaction, so
    without this guard this would log on every single click."""
    global _initialized
    try:
        Base.metadata.create_all(bind=engine)
        if not _initialized:
            logger.info(f"Database initialized successfully at {DB_PATH}")
            _initialized = True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise e

def get_db():
    """Dependency for acquiring a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
