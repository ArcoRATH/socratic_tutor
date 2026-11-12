import asyncio
import os
import re
import time

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from langchain_core.messages import AIMessage, HumanMessage

from src.core.config import settings
from src.core.constants import DEFAULT_STYLE, PEDAGOGICAL_PROMPTS
from src.database.connections import SessionLocal, init_db
from src.database.models import Document, DocumentChunk
from src.database.vector_store import VectorStore
from src.engine.extractor import DocumentExtractor
from src.engine.filter import select_content_chunks
from src.engine.llm import SocraticLLM
from src.interface.middleware import LearningMiddleware, build_initial_state
from src.agents.graph import app as langgraph_app

MAX_PDF_BYTES = 20 * 1024 * 1024  # Telegram bot API cannot download larger files


def _md_to_telegram_html(text: str) -> str:
    """Converts the LLM's markdown subset to Telegram-safe HTML."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"```(?:[a-zA-Z0-9_-]*\n)?(.*?)```", r"<pre>\1</pre>", text, flags=re.S)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = re.sub(r"(?m)^#{1,6}\s+(.+)$", r"<b>\1</b>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"<i>\1</i>", text)
    text = re.sub(r"(?m)^[-*]\s+", "• ", text)
    return text


def _truncate_for_telegram(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind("\n\n", 0, limit)
    if cut == -1:
        cut = text.rfind("\n", 0, limit)
    if cut == -1:
        cut = limit
    return text[:cut]


async def _reply(update: Update, text: str):
    """Replies with HTML formatting; falls back to plain text if Telegram rejects the markup."""
    try:
        await update.message.reply_text(
            _md_to_telegram_html(_truncate_for_telegram(text)), parse_mode=ParseMode.HTML
        )
    except BadRequest:
        await update.message.reply_text(_truncate_for_telegram(text))


def _extract_reply(messages) -> str:
    """Joins the AI messages produced this turn (after the user's last message)."""
    parts = []
    for message in reversed(list(messages)):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.content:
            parts.append(message.content)
    return "\n\n".join(reversed(parts))


def _is_pdf_document(document_meta) -> bool:
    mime = (document_meta.mime_type or "").lower()
    name = (document_meta.file_name or "").lower()
    return mime == "application/pdf" or (
        mime in ("", "application/octet-stream") and name.endswith(".pdf")
    )


def _looks_text_based(chunks, min_chars: int = 200) -> bool:
    return sum(len(chunk) for chunk in chunks) >= min_chars


def _store_document(safe_name: str, raw_chunks, content_chunks, user_id) -> int:
    """Sync DB/embedding pipeline — runs in a worker thread via asyncio.to_thread."""
    capped = content_chunks[: settings.MAX_EMBED_CHUNKS]
    with SessionLocal() as db:
        document = Document(
            title=safe_name,
            source_type="pdf",
            raw_content="\n\n".join(raw_chunks),
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        VectorStore(db).add_chunks(document.id, capped)

        state = build_initial_state()
        state["document_id"] = document.id
        state["total_chunks"] = len(capped)
        LearningMiddleware(db).save_context(user_id, state)
        return document.id


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /start command."""
    await _reply(
        update,
        "Welcome to your Socratic Tutor 🎓\n\n"
        "Upload a text-based PDF and I'll turn it into a lesson plan, then teach it section "
        "by section — explaining, quizzing, and tracking how well each idea sticks.\n\n"
        "Commands:\n"
        "/style — choose how I explain things\n"
        "/status — show the latest document and your live session\n"
        "/reset — abandon the current lesson and start fresh",
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles PDF uploads: download -> extract -> chunk -> embed -> session setup."""
    document_meta = update.message.document
    user_id = update.effective_user.id

    if not _is_pdf_document(document_meta):
        await _reply(update, "I only accept text-based PDF files for now — please upload a PDF.")
        return

    if (document_meta.file_size or 0) > MAX_PDF_BYTES:
        await _reply(update, "That file is over 20 MB, which is the Telegram bot limit — I can't download it. Try splitting or compressing it.")
        return

    await _reply(update, "Extracting and organizing your material, one moment...")

    os.makedirs("data", exist_ok=True)
    safe_name = os.path.basename(document_meta.file_name or f"upload_{document_meta.file_unique_id}")
    file_path = os.path.join("data", f"{user_id}_{int(time.time())}_{safe_name}")

    telegram_file = await document_meta.get_file()
    await telegram_file.download_to_drive(file_path)

    try:
        chunks = await DocumentExtractor().process_file(file_path)
    except Exception as error:
        await _reply(update, f"I couldn't read that file: {error}")
        return

    if not _looks_text_based(chunks):
        await _reply(update, "This PDF has no extractable text — it may be scanned or empty. I need a text-based PDF.")
        return

    await _reply(update, f"Book parsed — {len(chunks)} raw sections. Filtering out TOC, preface, and index…")

    try:
        selection = await asyncio.to_thread(select_content_chunks, chunks, SocraticLLM())
    except Exception as error:
        await _reply(update, f"I couldn't process that PDF: {error}")
        return

    content_chunks = selection["chunks"] or chunks
    if not _looks_text_based(content_chunks):
        await _reply(update, "I extracted text but couldn't identify teachable content in this PDF. Try another file.")
        return

    await _reply(update, f"Found {len(content_chunks)} lesson sections. Embedding them now — big books take a minute or two…")

    try:
        await asyncio.to_thread(_store_document, safe_name, chunks, content_chunks, user_id)
    except Exception as error:
        await _reply(update, f"I couldn't store that PDF: {error}")
        return

    minutes = max(1, sum(len(chunk.split()) for chunk in content_chunks) // 150)
    skipped = selection["front"] + selection["back"] + selection["heuristic_dropped"]
    skipped_note = f" (skipped {skipped} non-content sections: TOC, preface, index, …)" if skipped else ""
    capped_note = ""
    if len(content_chunks) > settings.MAX_EMBED_CHUNKS:
        capped_note = (
            f"\nThat's more than I can fit in one session — this pass covers the first "
            f"{settings.MAX_EMBED_CHUNKS} sections."
        )
    await _reply(
        update,
        f'I broke "{safe_name}" into {len(content_chunks)} lesson section(s){skipped_note} — about {minutes} min of study.'
        f"{capped_note}\n"
        "Send any message to start lesson 1.",
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The main loop: User Message -> Middleware -> LangGraph -> Telegram."""
    user_id = update.effective_user.id
    user_text = update.message.text

    with SessionLocal() as db:
        middleware = LearningMiddleware(db)

        current_state = middleware.load_context(user_id)
        current_state["messages"] = list(current_state.get("messages", [])) + [HumanMessage(content=user_text)]

        final_state = await langgraph_app.ainvoke(current_state)

        middleware.save_context(user_id, final_state)

        reply = _extract_reply(final_state["messages"])
        await _reply(update, reply if reply else "Something went wrong on my end — please try again.")


async def style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shows or changes the teaching style (/style, /style ELI5, ...)."""
    args = context.args or []
    if not args:
        user_id = update.effective_user.id
        with SessionLocal() as db:
            current = LearningMiddleware(db).load_context(user_id).get("pedagogical_style", DEFAULT_STYLE)
        options = "\n".join(f"/style {name}" for name in PEDAGOGICAL_PROMPTS)
        await _reply(update, f"Current teaching style: **{current}**\n\n{options}\n\nApplies from your next lesson.")
        return

    choice = args[0].strip().upper()
    if choice not in PEDAGOGICAL_PROMPTS:
        await _reply(update, f'Unknown style "{args[0]}". Options: {", ".join(PEDAGOGICAL_PROMPTS)}')
        return

    user_id = update.effective_user.id
    with SessionLocal() as db:
        middleware = LearningMiddleware(db)
        state = middleware.load_context(user_id)
        state["pedagogical_style"] = choice
        middleware.save_context(user_id, state)

    await _reply(update, f"Teaching style set to **{choice}** — applies from your next lesson.")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clears the current learning session."""
    user_id = update.effective_user.id
    with SessionLocal() as db:
        LearningMiddleware(db).save_context(user_id, build_initial_state())
    await _reply(update, "Session cleared. Upload a PDF to start a new lesson.")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shows the latest ingestion and the caller's live session state."""
    lines = []
    with SessionLocal() as db:
        document = db.query(Document).order_by(Document.id.desc()).first()
        if document is None:
            lines.append("No documents ingested yet — upload a PDF to begin.")
        else:
            embedded = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == document.id)
                .count()
            )
            capped = embedded >= settings.MAX_EMBED_CHUNKS
            lines.append(
                f"**Latest document**: {document.title} (id {document.id})\n"
                f"Sections embedded: **{embedded}**{' / capped' if capped else ''}"
            )

        state = LearningMiddleware(db).load_context(update.effective_user.id)
        total = state.get("total_chunks") or 0
        index = state.get("current_chunk_index") or 0
        if state.get("document_id") and total:
            lines.append(
                f"**Your session**: document {state['document_id']}, "
                f"section **{index + 1}/{total}** (next up)"
            )
        else:
            lines.append("No active session — upload a PDF or send a message to start.")

    await _reply(update, "\n\n".join(lines))


def main():
    init_db()
    application = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('reset', reset))
    application.add_handler(CommandHandler('status', status))
    application.add_handler(CommandHandler('style', style))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    application.run_polling()


if __name__ == '__main__':
    main()
