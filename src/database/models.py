from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from src.database.connections import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    progress = relationship("UserProgress", back_populates="user")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    source_type = Column(String)  # 'pdf', 'web', 'image'
    raw_content = Column(Text)    # Full text extraction from Kreuzberg
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    # pgvector column: Gemini 1.5 Pro text-embedding-004 uses 768 dimensions
    embedding = Column(Vector(768)) 

    document = relationship("Document", back_populates="chunks")

class UserProgress(Base):
    """
    Stores the current LangGraph state for a user on a specific document.
    """
    __tablename__ = "user_progress"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    document_id = Column(Integer, ForeignKey("documents.id"))
    
    current_chunk_index = Column(Integer, default=0)
    # Stores the LangGraph 'AgentState' as JSON for easy checkpointing
    graph_state = Column(JSON, default=dict)
    
    last_accessed = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="progress")

class ConceptMastery(Base):
    """
    Persists the MasteryRecord Pydantic model we defined earlier.
    """
    __tablename__ = "concept_mastery"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    concept_name = Column(String, nullable=False)
    
    mastery_score = Column(Float, default=0.0)
    attempts = Column(Integer, default=0)
    struggle_points = Column(JSON)  # List of strings/keywords
    
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())