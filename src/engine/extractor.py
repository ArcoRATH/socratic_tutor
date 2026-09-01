from typing import List

from kreuzberg import extract_file
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentExtractor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=2200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def process_file(self, file_path: str) -> List[str]:
        """
        Extracts text using Kreuzberg and splits it into lesson-sized chunks.
        Size-based splitting on purpose: no per-sentence embedding calls, so
        full-length books (1000+ chunks) cannot blow the embedding quota.
        """
        try:
            result = await extract_file(file_path)
            raw_text = result.content

            if not raw_text.strip():
                raise ValueError("Extracted text is empty. Check file integrity or OCR settings.")

            # ". " as a separator leaves a leading ". " on continuation chunks — strip it.
            chunks = [chunk.strip() for chunk in self.splitter.split_text(raw_text)]
            return [chunk[2:] if chunk.startswith(". ") else chunk for chunk in chunks if chunk]
        except Exception as e:
            print(f"Extraction Error: {str(e)}")
            return []

    async def process_url(self, url: str) -> List[str]:
        """
        Uses Kreuzberg's web-scraping capabilities.
        """
        return await self.process_file(url)
