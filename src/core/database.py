from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from src.core.config import config
from src.core.logging import logger

Base = declarative_base()

def get_engine():
    db_url = config.database_url
    # Ensure directory exists for sqlite file
    if db_url.startswith("sqlite:///"):
        file_path = db_url.replace("sqlite:///", "")
        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            path_obj = Path.cwd() / path_obj
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(db_url, connect_args={"check_same_thread": False})
    return create_engine(db_url)

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Creates all database tables."""
    import src.models.db  # Ensure models are imported
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized and schema verified.")

def get_db() -> Generator[Session, None, None]:
    """Dependency for providing database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
