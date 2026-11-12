from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    TELEGRAM_BOT_TOKEN: str = Field(..., env="TELEGRAM_BOT_TOKEN")

    # Embedding provider (SiliconFlow, OpenAI-compatible). Optional: without a
    # key, chunks are stored with a NULL embedding and teaching still works.
    EMBEDDING_API_KEY: Optional[str] = Field(None, env="SILICONFLOW_API_KEY")
    EMBEDDING_BASE_URL: str = Field("https://api.siliconflow.cn/v1")

    # Database
    DATABASE_URL: str = Field("postgresql://user:pass@localhost:5432/socratic_db")

    # Model Selection (Versatility)
    LLM_MODEL: str = Field("gemini-3.5-flash")
    EMBEDDING_MODEL: str = Field("BAAI/bge-m3")
    # Must match models.py's Vector column; bge-m3 outputs 1024 dims.
    EMBEDDING_DIMENSIONS: int = Field(1024)

    # App Settings
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    MAX_EMBED_CHUNKS: int = Field(30, description="Max sections to embed per document; the rest are stored unembedded and remain teachable")

    # Pydantic configuration to read from .env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
