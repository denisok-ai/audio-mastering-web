# @file config.py
# @description Конфигурация приложения (переменные окружения)
# @dependencies —
# @created 2026-02-26

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки из переменных окружения."""
    max_upload_mb: int = 100
    allowed_extensions: set[str] = {"wav", "mp3", "flac"}
    temp_dir: str = "/tmp/masterflow"
    default_target_lufs: float = -14.0

    class Config:
        env_prefix = "MASTERFLOW_"


settings = Settings()
