from typing import Annotated, Dict, List, Optional, Sequence, TypedDict
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

class MasteryRecord(BaseModel):
    """Data structure for tracking understanding of a specific topic."""
    concept_id: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    attempts: int = 0
    struggle_points: List[str] = []
    last_feedback: Optional[str] = None

class AgentState(TypedDict):
    # Core LangGraph message history
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Session metadata
    document_id: int
    current_chunk_index: int
    total_chunks: int
    
    # Active learning context
    active_chunk_text: str
    active_concept: str
    previous_chunk_text: str
    mastery_map: Dict[str, MasteryRecord]
    
    # Flow control
    last_question: Optional[str]
    retry_count: int
    is_session_complete: bool
    pedagogical_style: str