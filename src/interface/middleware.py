from typing import Any, Dict

from langchain_core.messages import messages_from_dict, messages_to_dict
from sqlalchemy.orm import Session

from src.agents.state import MasteryRecord
from src.core.constants import DEFAULT_STYLE
from src.database.models import ConceptMastery, User, UserProgress

STATE_EXCLUDED_KEYS = ("messages", "mastery_map")


def build_initial_state() -> Dict[str, Any]:
    """A complete AgentState skeleton safe to use before any ingestion."""
    return {
        "messages": [],
        "document_id": None,
        "current_chunk_index": 0,
        "total_chunks": 0,
        "active_chunk_text": "",
        "active_concept": "",
        "previous_chunk_text": "",
        "mastery_map": {},
        "last_question": None,
        "retry_count": 0,
        "is_session_complete": False,
        "pedagogical_style": DEFAULT_STYLE,
    }


def serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """AgentState -> JSON-safe dict for the user_progress.graph_state column."""
    serialized = {k: v for k, v in state.items() if k not in STATE_EXCLUDED_KEYS}
    serialized["messages"] = messages_to_dict(list(state.get("messages", [])))
    serialized["mastery_map"] = {
        name: record.model_dump()
        for name, record in state.get("mastery_map", {}).items()
    }
    return serialized


def deserialize_state(raw: Dict[str, Any]) -> Dict[str, Any]:
    """JSON dict from the DB -> AgentState with real message/MasteryRecord objects."""
    state = build_initial_state()
    state.update({k: v for k, v in raw.items() if k not in STATE_EXCLUDED_KEYS})
    state["messages"] = messages_from_dict(raw.get("messages", []))
    state["mastery_map"] = {
        name: MasteryRecord(**record)
        for name, record in raw.get("mastery_map", {}).items()
    }
    return state


class LearningMiddleware:
    def __init__(self, db_session: Session):
        self.db = db_session

    def get_or_create_user(self, telegram_id: str) -> User:
        user = self.db.query(User).filter(User.telegram_id == str(telegram_id)).first()
        if not user:
            user = User(telegram_id=str(telegram_id))
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        return user

    def load_context(self, telegram_id: str) -> Dict[str, Any]:
        """Hydrates the LangGraph state from the database."""
        user = self.get_or_create_user(telegram_id)
        progress = self.db.query(UserProgress).filter(UserProgress.user_id == user.id).first()

        if not progress or not progress.graph_state:
            return build_initial_state()

        return deserialize_state(progress.graph_state)

    def save_context(self, telegram_id: str, state: Dict[str, Any]):
        """Persists the updated LangGraph state back to PostgreSQL."""
        user = self.get_or_create_user(telegram_id)
        progress = self.db.query(UserProgress).filter(UserProgress.user_id == user.id).first()
        serialized = serialize_state(state)

        if progress:
            progress.graph_state = serialized
            progress.current_chunk_index = state.get("current_chunk_index", 0)
            if state.get("document_id") is not None:
                progress.document_id = state["document_id"]
        else:
            progress = UserProgress(
                user_id=user.id,
                document_id=state.get("document_id"),
                current_chunk_index=state.get("current_chunk_index", 0),
                graph_state=serialized,
            )
            self.db.add(progress)

        self._sync_mastery(user.id, state.get("mastery_map", {}))
        self.db.commit()

    def _sync_mastery(self, user_id: int, mastery_map: Dict[str, MasteryRecord]):
        """Upserts one ConceptMastery row per tracked concept."""
        for concept_name, record in mastery_map.items():
            row = self.db.query(ConceptMastery).filter(
                ConceptMastery.user_id == user_id,
                ConceptMastery.concept_name == concept_name,
            ).first()
            if not row:
                row = ConceptMastery(user_id=user_id, concept_name=concept_name)
                self.db.add(row)
            row.mastery_score = record.score
            row.attempts = record.attempts
            row.struggle_points = record.struggle_points
