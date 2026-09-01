from langgraph.graph import StateGraph, END
from sqlalchemy import func
from langchain_core.messages import AIMessage

from src.agents.state import AgentState, MasteryRecord
from src.core.constants import MASTERY_THRESHOLD, MAX_RETRIES, PEDAGOGICAL_PROMPTS
from src.database.connections import SessionLocal
from src.database.models import DocumentChunk
from src.engine.llm import SocraticLLM

llm = SocraticLLM()


def retrieve_node(state: AgentState):
    """Step A: Load the chunk at the current index and refresh the chunk count."""
    document_id = state.get("document_id")
    index = state.get("current_chunk_index", 0)
    if document_id is None:
        return {"active_chunk_text": "", "active_concept": "", "total_chunks": 0}

    with SessionLocal() as db:
        total = db.query(func.count(DocumentChunk.id)).filter(
            DocumentChunk.document_id == document_id
        ).scalar() or 0
        chunk = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.chunk_index == index,
        ).first()

    if chunk is None:
        return {"active_chunk_text": "", "active_concept": "", "total_chunks": total}

    concept = chunk.content.strip().splitlines()[0][:80] or f"chunk-{index}"
    return {
        "active_chunk_text": chunk.content,
        "active_concept": concept,
        "total_chunks": total,
    }


def teach_node(state: AgentState):
    """Step B: Explain the concept in the configured pedagogical style."""
    style = PEDAGOGICAL_PROMPTS.get(
        state.get("pedagogical_style", "SOCRATIC"), PEDAGOGICAL_PROMPTS["SOCRATIC"]
    )
    explanation = llm.teach(
        chunk=state["active_chunk_text"],
        style=style,
        full_context=state.get("previous_chunk_text", ""),
    )
    return {"messages": [AIMessage(content=explanation)]}


def challenge_node(state: AgentState):
    """Step C: Pose one Socratic question and mark it as pending."""
    question = llm.challenge(state["active_chunk_text"], list(state.get("messages", [])))
    return {
        "messages": [AIMessage(content=question)],
        "last_question": question,
    }


def evaluate_node(state: AgentState):
    """Step D: Grade the pending answer and update mastery."""
    user_answer = state["messages"][-1].content
    try:
        evaluation = llm.evaluate(
            chunk=state["active_chunk_text"],
            question=state.get("last_question") or "",
            user_answer=user_answer,
        )
    except Exception:
        evaluation = {}

    accuracy = min(max(float(evaluation.get("accuracy_score", 0.0)), 0.0), 1.0)
    feedback = evaluation.get("feedback") or "I couldn't assess that answer — try rephrasing it."
    struggle = evaluation.get("struggle_points") or []
    if isinstance(struggle, str):
        struggle = [struggle]

    concept = state["active_concept"]
    mastery_map = dict(state.get("mastery_map", {}))
    record = mastery_map.get(concept, MasteryRecord(concept_id=concept))

    # Mastery formula: M_new = M_old + ((1 - M_old) * accuracy) / (1 + attempts)
    attempts = state.get("retry_count", 0)
    new_score = record.score + ((1 - record.score) * accuracy) / (1 + attempts)
    record.score = min(new_score, 1.0)
    record.attempts += 1
    record.struggle_points = list(set(record.struggle_points + struggle))
    record.last_feedback = feedback
    mastery_map[concept] = record

    return {
        "mastery_map": mastery_map,
        "messages": [AIMessage(content=feedback)],
        "last_question": None,
        "retry_count": attempts + 1,
    }


def advance_node(state: AgentState):
    """Move to the next chunk and reset per-chunk flow flags."""
    return {
        "current_chunk_index": state.get("current_chunk_index", 0) + 1,
        "previous_chunk_text": state.get("active_chunk_text", ""),
        "last_question": None,
        "retry_count": 0,
    }


def complete_node(state: AgentState):
    return {
        "messages": [AIMessage(
            content="That was the final section — you've completed this document. "
                    "Upload another one whenever you're ready."
        )],
        "is_session_complete": True,
        "last_question": None,
        "retry_count": 0,
    }


def no_document_node(state: AgentState):
    return {
        "messages": [AIMessage(
            content="I don't have any material to teach yet. "
                    "Upload a PDF and I'll turn it into a lesson plan."
        )],
    }


def route_entry(state):
    """Pick the first node for this turn based on where the learner is."""
    if state.get("is_session_complete"):
        return "complete"
    if not state.get("document_id"):
        return "no_document"
    if state.get("last_question"):
        return "evaluate"
    return "retrieve"


def route_after_eval(state):
    """Mastery-gated routing: advance on threshold, move on after MAX_RETRIES."""
    record = state.get("mastery_map", {}).get(state.get("active_concept", ""))
    score = record.score if record else 0.0
    if score >= MASTERY_THRESHOLD:
        if state.get("current_chunk_index", 0) >= state.get("total_chunks", 0) - 1:
            return "complete"
        return "advance"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "advance"
    return "retry"


workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("teach", teach_node)
workflow.add_node("challenge", challenge_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("advance", advance_node)
workflow.add_node("complete", complete_node)
workflow.add_node("no_document", no_document_node)

workflow.set_conditional_entry_point(
    route_entry,
    {
        "retrieve": "retrieve",
        "evaluate": "evaluate",
        "complete": "complete",
        "no_document": "no_document",
    },
)
workflow.add_edge("retrieve", "teach")
workflow.add_edge("teach", "challenge")
workflow.add_edge("challenge", END)
workflow.add_edge("advance", "retrieve")
workflow.add_conditional_edges(
    "evaluate",
    route_after_eval,
    {
        "retry": "teach",
        "advance": "advance",
        "complete": "complete",
    },
)
workflow.add_edge("complete", END)

app = workflow.compile()
