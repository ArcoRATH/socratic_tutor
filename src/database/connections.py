from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.core.config import settings

# 1. Create the SQLAlchemy Engine
# The 'check_same_thread' is only needed for SQLite, but we keep it generic.
# For Postgres, we rely on the DATABASE_URL from our .env file.
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Checks if the connection is alive before using it
    pool_size=10,        # Number of permanent connections to keep open
    max_overflow=20      # Extra connections allowed during peak traffic
)

# 2. Create a SessionFactory
# This allows us to create 'SessionLocal' instances for each request or node turn.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 3. Base class for our Models
# Every model in /database/models.py will inherit from this Base.
Base = declarative_base()

def get_db():
    """
    Utility generator to provide a database session.
    Ensures the session is closed automatically after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Creates the pgvector extension and all ORM tables.
    Imports models lazily so models.py can reuse this module's Base
    without a circular import.
    """
    from src.database import models

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)