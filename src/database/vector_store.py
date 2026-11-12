import time

import google.generativeai as genai
from sqlalchemy import select, and_
from sqlalchemy.orm import Session
from typing import List, Optional

from src.database.models import DocumentChunk
from src.core.config import settings

# Configure Gemini for Embeddings
genai.configure(api_key=settings.GEMINI_API_KEY)

class VectorStore:
    BATCH_SIZE = 90  # texts per embedding request (API caps batch sizes)

    def __init__(self, session: Session):
        self.session = session
        self.model = settings.EMBEDDING_MODEL  # gemini-embedding-001, pinned to 768 dims below

    def _get_embedding(self, text: str) -> List[float]:
        """Generates a 768-dimensional embedding using Gemini."""
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=768
        )
        return result['embedding']

    def get_relevant_chunks(
        self, 
        query: str, 
        document_id: int, 
        top_k: int = 3
    ) -> List[DocumentChunk]:
        """
        Performs a cosine similarity search within a specific document.
        """
        # 1. Embed the user query
        query_vector = self._get_embedding(query)

        # 2. Query pgvector using Cosine Distance (<=> operator)
        # Cosine Distance = 1 - Cosine Similarity. Smaller is better.
        stmt = (
            select(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
            .limit(top_k)
        )

        results = self.session.execute(stmt).scalars().all()
        return list(results)

    def add_chunks(self, document_id: int, text_chunks: List[str]):
        """
        Embeds and stores multiple chunks for a new document.
        Batches requests and retries on rate limits so full-length books work.
        """
        new_chunks = []
        total_batches = (len(text_chunks) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        for start in range(0, len(text_chunks), self.BATCH_SIZE):
            print(f"Embedding batch {start // self.BATCH_SIZE + 1}/{total_batches} "
                  f"({len(text_chunks[start:start + self.BATCH_SIZE])} chunks)", flush=True)
            batch = text_chunks[start:start + self.BATCH_SIZE]
            embeddings = self._embed_batch(batch)
            new_chunks.extend(
                DocumentChunk(
                    document_id=document_id,
                    content=text,
                    chunk_index=start + i,
                    embedding=embedding,
                )
                for i, (text, embedding) in enumerate(zip(batch, embeddings))
            )
        self.session.add_all(new_chunks)
        self.session.commit()
        note = " (capped at MAX_EMBED_CHUNKS)" if len(text_chunks) == settings.MAX_EMBED_CHUNKS else ""
        print(f"Embedded {len(text_chunks)}/{len(text_chunks)} chunks{note}", flush=True)

    def _embed_batch(self, texts: List[str], attempts: int = 4) -> List[List[float]]:
        for attempt in range(attempts):
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=texts,
                    task_type="retrieval_document",
                    output_dimensionality=768
                )
                return result['embedding']
            except Exception as error:
                if attempt == attempts - 1:
                    raise
                wait = 20 * (attempt + 1)
                print(f"Embedding batch failed ({error}); retrying in {wait}s")
                time.sleep(wait)