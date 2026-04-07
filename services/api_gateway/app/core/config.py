from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DocIntel AI Platform"
    api_v1_prefix: str = "/api/v1"
    environment: str = "development"
    debug: bool = True
    secret_key: str = "change-me"
    database_url: str = "sqlite:///./data/docintel.db"
    upload_dir: str = "./data/uploads"
    model_dir: str = "./data/models"
    index_dir: str = "./data/index"
    allowed_origins: List[str] | str = ["http://localhost:5173", "http://localhost:3000"]
    auto_approve_threshold: float = 0.92
    human_review_threshold: float = 0.70
    mlflow_tracking_uri: str = "http://localhost:5001"
    qdrant_url: str = "http://localhost:6333"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def supported_extensions(self) -> set[str]:
        return {".txt", ".md", ".json", ".csv", ".pdf"}

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[4]


settings = Settings()
