import re
from typing import Any, Dict, List, Tuple

from src.core.constants import LABEL_BACK_MATTER, LABEL_CONTENT, LABEL_FRONT_MATTER

TOC_LINE_RATIO = 0.5       # share of dot-leader/page-number lines that marks a TOC chunk
BOILERPLATE_RATIO = 0.25   # share of copyright/publisher lines that marks boilerplate
SMALL_DOC_THRESHOLD = 6    # chunks below this count skip the LLM classification pass
PREVIEW_CHARS = 120

_TOC_LINE = re.compile(r"\.{2,}\s*\d{1,4}\s*$|^\s*\d{1,4}\s*$")
_COPYRIGHT_LINE = re.compile(r"©|copyright|all rights reserved|published by|isbn", re.IGNORECASE)


def _is_toc_like(chunk: str) -> bool:
    """Char-weighted: TOC pages are dominated by dot-leader/page-number lines."""
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    if not lines:
        return False
    total = sum(len(line) for line in lines)
    hits = sum(len(line) for line in lines if _TOC_LINE.search(line))
    return hits / total >= TOC_LINE_RATIO


def _is_boilerplate(chunk: str) -> bool:
    """Char-weighted: copyright/publisher pages match on most of their text."""
    lines = [line for line in chunk.splitlines() if line.strip()]
    if not lines:
        return False
    total = sum(len(line) for line in lines)
    hits = sum(len(line) for line in lines if _COPYRIGHT_LINE.search(line))
    return hits / total >= BOILERPLATE_RATIO


def _trim_non_content(chunks: List[str]) -> Tuple[List[str], int]:
    """Deterministically drops obvious TOC/copyright chunks. Returns (kept, dropped)."""
    kept, dropped = [], 0
    for chunk in chunks:
        if _is_toc_like(chunk) or _is_boilerplate(chunk):
            dropped += 1
        else:
            kept.append(chunk)
    return kept, dropped


def _preview(chunk: str, chars: int = PREVIEW_CHARS) -> str:
    return " ".join(chunk.split())[:chars]


def select_content_chunks(chunks: List[str], llm: Any) -> Dict[str, Any]:
    """
    Keeps only the teachable body of a document: heuristic pre-trim, then an
    LLM classification pass over chunk previews. Falls back to keeping
    everything (minus heuristic trims) if the classifier fails.
    """
    trimmed, heuristic_dropped = _trim_non_content(chunks)

    if len(trimmed) <= SMALL_DOC_THRESHOLD:
        return {
            "chunks": trimmed,
            "front": 0,
            "back": 0,
            "heuristic_dropped": heuristic_dropped,
        }

    previews = [_preview(chunk) for chunk in trimmed]
    try:
        labels: List[str] = []
        for start in range(0, len(previews), 250):
            labels.extend(llm.classify_chunk_previews(previews[start:start + 250]))
        if len(labels) != len(trimmed):
            raise ValueError(f"classifier returned {len(labels)} labels for {len(trimmed)} chunks")
    except Exception as error:
        print(f"Content classification failed, keeping all chunks: {error}")
        return {
            "chunks": trimmed,
            "front": 0,
            "back": 0,
            "heuristic_dropped": heuristic_dropped,
        }

    content = [chunk for chunk, label in zip(trimmed, labels) if label == LABEL_CONTENT]
    return {
        "chunks": content,
        "front": sum(1 for label in labels if label == LABEL_FRONT_MATTER),
        "back": sum(1 for label in labels if label == LABEL_BACK_MATTER),
        "heuristic_dropped": heuristic_dropped,
    }
