# @file config.py
# @description Конфигурация приложения (переменные окружения)
# @dependencies —
# @created 2026-02-26

import os
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Настройки из переменных окружения. Префикс: MAGIC_MASTER_."""
    max_upload_mb: int = 100
    allowed_extensions: set[str] = {"wav", "mp3", "flac"}
    temp_dir: str = "/tmp/masterflow"
    default_target_lufs: float = -14.0
    jobs_max_entries: int = 100
    jobs_done_ttl_seconds: int = 3600
    # Режим отладки: все функции без авторизации (MAGIC_MASTER_DEBUG=1)
    debug_mode: bool = Field(False, validation_alias="DEBUG")

    @field_validator("debug_mode", mode="before")
    @classmethod
    def parse_debug_mode(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    class Config:
        env_prefix = "MAGIC_MASTER_"


settings = Settings()
