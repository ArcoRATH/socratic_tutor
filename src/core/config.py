from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    TELEGRAM_BOT_TOKEN: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    
    # Database
    DATABASE_URL: str = Field("postgresql://user:pass@localhost:5432/socratic_db")
    
    # Model Selection (Versatility)
    LLM_MODEL: str = Field("gemini-3.5-flash")
    EMBEDDING_MODEL: str = Field("models/gemini-embedding-001")
    
    # App Settings
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Pydantic configuration to read from .env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()