import json
import google.generativeai as genai
from typing import Any, Dict, List

from src.core.config import settings
from src.core.constants import (
    LABEL_BACK_MATTER,
    LABEL_CONTENT,
    LABEL_FRONT_MATTER,
    SYSTEM_PROMPT_EVALUATE,
    SYSTEM_PROMPT_TEACH,
)

TEACH_STRUCTURE = """Structure the explanation as:
1. A one-line hook that connects this to something familiar or to what we just covered.
2. The core explanation in your assigned style.
3. 2-3 key takeaways as short bullet points (markdown).
"""

VALID_LABELS = {LABEL_FRONT_MATTER, LABEL_CONTENT, LABEL_BACK_MATTER}


class SocraticLLM:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._teach_model = genai.GenerativeModel(
            model_name=settings.LLM_MODEL,
            system_instruction=SYSTEM_PROMPT_TEACH,
            generation_config={"temperature": 0.7, "top_p": 0.95},
        )
        self._eval_model = genai.GenerativeModel(
            model_name=settings.LLM_MODEL,
            system_instruction=SYSTEM_PROMPT_EVALUATE,
            generation_config={"temperature": 0.2},
        )

    @staticmethod
    def _format_history(session_history: List[Any], limit: int = 6) -> str:
        """Renders the last few messages so the model knows where the learner is."""
        lines = []
        for message in list(session_history)[-limit:]:
            role = "Tutor" if getattr(message, "type", "") == "ai" else "Learner"
            content = str(getattr(message, "content", message)).strip().replace("\n", " ")
            lines.append(f"{role}: {content[:300]}")
        return "\n".join(lines) if lines else "(conversation just started)"

    def teach(self, chunk: str, style: str, full_context: str = "") -> str:
        """
        Step B: Explains a concept using a specific pedagogical style.
        """
        context_block = (
            f"WHAT WE COVERED JUST BEFORE THIS (use it for continuity, do not re-teach it):\n"
            f"{full_context[:1500]}\n\n" if full_context else ""
        )
        prompt = f"""
{context_block}SECTION TO TEACH (the only topic allowed):
---
{chunk}
---

TEACHING STYLE (required): {style}

{TEACH_STRUCTURE}
Respond in markdown. Do not ask any question.
"""
        response = self._teach_model.generate_content(prompt)
        return response.text

    def challenge(self, chunk: str, session_history: List[Any]) -> str:
        """
        Step C: Generates exactly one critical thinking question.
        """
        prompt = f"""
SECTION THE LEARNER JUST READ:
---
{chunk}
---

RECENT CONVERSATION:
{self._format_history(session_history)}

TASK: Ask exactly ONE question about this section.
- It must require application, inference, or prediction — never pure recall.
- Never a yes/no question.
- Anchor it in a concrete scenario where possible.
- Calibrate difficulty to the learner's demonstrated level in the conversation.
- Output only the question itself, nothing else.
"""
        response = self._teach_model.generate_content(prompt)
        return response.text.strip()

    def evaluate(self, chunk: str, question: str, user_answer: str) -> Dict[str, Any]:
        """
        Step D: Assess the user's response and provide structured feedback.
        """
        prompt = f"""
SECTION (ground truth):
---
{chunk}
---

QUESTION ASKED:
{question}

LEARNER'S ANSWER:
{user_answer}

TASK: Grade the answer.
- is_correct: true only if the core concept is genuinely understood.
- accuracy_score: 0.0 to 1.0, weighting concept accuracy above detail accuracy.
- feedback: 1-3 sentences, encouraging but Socratic — name what is missing or
  mistaken without revealing the full answer.
- struggle_points: 1-3 short keywords for the sub-ideas missed or misunderstood.

RETURN JSON ONLY:
{{
    "is_correct": bool,
    "accuracy_score": float,
    "feedback": str,
    "struggle_points": [str]
}}
"""
        response = self._eval_model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        return json.loads(response.text)

    def classify_chunk_previews(self, previews: List[str]) -> List[str]:
        """
        Labels each chunk preview as FRONT_MATTER, CONTENT, or BACK_MATTER.
        Raises on failure so callers can fall back to keeping everything.
        """
        listing = "\n".join(f"{i}: {preview}" for i, preview in enumerate(previews))
        prompt = f"""
The lines below are numbered previews of consecutive sections extracted from a book/document.

{listing}

TASK: Classify each section by its number:
- FRONT_MATTER: cover, copyright, dedication, table of contents, preface, foreword,
  about the author, acknowledgements, list of figures/tables.
- BACK_MATTER: index, glossary, references, bibliography, appendix, about the publisher.
- CONTENT: actual teaching material (chapters and everything inside them).

Index sections look like lists of terms followed by page numbers. Prefaces talk about
the book itself rather than the subject matter.

RETURN JSON ONLY:
{{
    "labels": ["CONTENT", "FRONT_MATTER", ...]
}}
The list must have exactly {len(previews)} entries, in order.
"""
        response = self._eval_model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        raw = json.loads(response.text).get("labels", [])
        if not isinstance(raw, list):
            raise ValueError("classifier did not return a list")
        labels = [label if label in VALID_LABELS else LABEL_CONTENT for label in raw]
        return (labels + [LABEL_CONTENT] * len(previews))[:len(previews)]
