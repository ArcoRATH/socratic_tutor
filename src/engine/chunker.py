from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.core.config import settings

class SocraticChunker:
    def __init__(self):
        # Initialize embeddings using the versatile config we built
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            output_dimensionality=768
        )
        
        # The 'percentile' threshold is generally the most robust for textbooks.
        # It triggers a break when the semantic difference between sentences 
        # is in the top 5% (95th percentile) of all differences in the doc.
        self.chunker = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95.0 
        )

    def create_chunks(self, text: str) -> List[str]:
        """
        Splits raw text into semantically cohesive blocks.
        """
        if not text.strip():
            return []

        # The chunker returns a list of 'Document' objects; we just want the strings.
        documents = self.chunker.create_documents([text])
        return [doc.page_content for doc in documents]

    def estimate_mastery_time(self, chunks: List[str]) -> int:
        """
        Optional: Estimates total study time based on chunk complexity.
        Useful for the Telegram UI to say "This book will take ~4 hours."
        """
        # Average reading speed + Socratic thinking time (~200 words/min)
        total_words = sum(len(chunk.split()) for chunk in chunks)
        return max(1, total_words // 150)