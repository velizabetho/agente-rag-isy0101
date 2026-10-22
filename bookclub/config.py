"""Application settings loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    planner_model: str = "openai/gpt-oss-20b"
    judge_model: str = "openai/gpt-oss-20b"

    embedding_provider: str = "local"
    embedding_model: str = "intfloat/multilingual-e5-small"
    openai_api_key: str = ""

    mongodb_uri: str = ""
    mongodb_db: str = "clubes_lectura"

    sqlite_path: Path = DATA_DIR / "bookclub.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
