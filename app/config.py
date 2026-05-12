from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT_DIR / ".env"

load_dotenv(ENV_FILE, override=False)


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    embedding_provider: str = Field(..., alias="EMBEDDING_PROVIDER")
    qdrant_url: str = Field(..., alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(..., alias="QDRANT_COLLECTION_NAME")
    embedding_model: str = Field(..., alias="EMBEDDING_MODEL")
    reranker_model: str = Field(..., alias="RERANKER_MODEL")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls.model_validate(
            {
                "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER"),
                "QDRANT_URL": os.getenv("QDRANT_URL"),
                "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", ""),
                "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME"),
                "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
                "RERANKER_MODEL": os.getenv("RERANKER_MODEL"),
            }
        )


class QdrantSettings(BaseModel):
    """Qdrant connection settings loaded from environment variables."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    qdrant_url: str = Field(..., alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")

    @classmethod
    def from_env(cls) -> "QdrantSettings":
        return cls.model_validate(
            {
                "QDRANT_URL": os.getenv("QDRANT_URL"),
                "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", ""),
            }
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings.from_env()


@lru_cache(maxsize=1)
def get_qdrant_settings() -> QdrantSettings:
    """Return cached Qdrant-only settings."""

    return QdrantSettings.from_env()
