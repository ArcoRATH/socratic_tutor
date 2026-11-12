import time

from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.orm import Session
from typing import List, Optional

from src.database.models import DocumentChunk
from src.core.config import settings

_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    """Shared OpenAI-compatible client for SiliconFlow; None when no key is set."""
    global _client
    if not settings.EMBEDDING_API_KEY:  # None or empty string both mean "no embeddings"
        return None
    if _client is None:
        _client = OpenAI(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
        )
    return _client


class VectorStore:
    BATCH_SIZE = 32  # texts per embedding request (bge-m3 caps ~8k tokens/line)

    def __init__(self, session: Session):
        self.session = session
        self.model = settings.EMBEDDING_MODEL  # BAAI/bge-m3, fixed 1024 dims

    def get_relevant_chunks(
        self,
        query: str,
        document_id: int,
        top_k: int = 3
    ) -> List[DocumentChunk]:
        """
        Performs a cosine similarity search within a specific document.
        Returns [] when embeddings are unavailable (no API key or embed failure).
        """
        query_vector = self._query_embedding(query)
        if query_vector is None:
            return []

        # Cosine Distance = 1 - Cosine Similarity. Smaller is better.
        stmt = (
            select(DocumentChunk)
            .filter(
                DocumentChunk.document_id == document_id,
                DocumentChunk.embedding.is_not(None),
            )
            .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
            .limit(top_k)
        )

        results = self.session.execute(stmt).scalars().all()
        return list(results)

    def add_chunks(
        self,
        document_id: int,
        text_chunks: List[str],
        embed_limit: Optional[int] = None,
    ) -> int:
        """
        Stores every chunk for a new document, embedding only the first
        `embed_limit` chunks. Embedding is a quota-bounded extra: on failure
        the remaining chunks are stored with a NULL embedding and stay teachable.
        """
        total = len(text_chunks)
        embed_limit = total if embed_limit is None else min(embed_limit, total)
        embeddings: List[Optional[List[float]]] = [None] * total
        embedded = 0

        total_batches = (embed_limit + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        for start in range(0, embed_limit, self.BATCH_SIZE):
            batch = text_chunks[start:min(start + self.BATCH_SIZE, embed_limit)]
            print(f"Embedding batch {start // self.BATCH_SIZE + 1}/{total_batches} "
                  f"({len(batch)} chunks)", flush=True)
            try:
                batch_embeddings = self._embed_batch(batch)
            except Exception as error:
                print(f"Embedding failed at batch {start // self.BATCH_SIZE + 1}/{total_batches} "
                      f"({error}); storing remaining chunks unembedded", flush=True)
                break
            for i, embedding in enumerate(batch_embeddings):
                embeddings[start + i] = embedding
            embedded += len(batch_embeddings)

        new_chunks = [
            DocumentChunk(
                document_id=document_id,
                content=text,
                chunk_index=index,
                embedding=embeddings[index],
            )
            for index, text in enumerate(text_chunks)
        ]
        self.session.add_all(new_chunks)
        self.session.commit()
        print(f"Embedded {embedded}/{total} chunks", flush=True)
        return embedded

    def _query_embedding(self, text: str) -> Optional[List[float]]:
        """Embeds a search query; None if embeddings are unavailable."""
        if _get_client() is None:
            print("No embedding API key configured; vector search disabled", flush=True)
            return None
        try:
            return self._embed_batch([text])[0]
        except Exception as error:
            print(f"Query embedding failed ({error}); vector search disabled", flush=True)
            return None

    def _embed_batch(self, texts: List[str], attempts: int = 4) -> List[List[float]]:
        client = _get_client()
        if client is None:
            raise RuntimeError("EMBEDDING_API_KEY (SILICONFLOW_API_KEY) is not set")
        for attempt in range(attempts):
            try:
                # NOTE: do not send `dimensions` — fixed-width models like
                # bge-m3 (native 1024) reject that parameter with HTTP 400.
                response = client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as error:
                if attempt == attempts - 1:
                    raise
                wait = 20 * (attempt + 1)
                print(f"Embedding batch failed ({error}); retrying in {wait}s")
                time.sleep(wait)
