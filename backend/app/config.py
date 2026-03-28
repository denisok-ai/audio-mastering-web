# @file config.py
# @description Конфигурация приложения (переменные окружения)
# @dependencies —
# @created 2026-02-26

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

# Пути к .env: корень проекта и каталог backend (при запуске из backend/ или из корня)
_backend_dir = Path(__file__).resolve().parent.parent
_root_dir = _backend_dir.parent
_ENV_FILES = [str(_root_dir / ".env"), str(_backend_dir / ".env"), ".env"]


class Settings(BaseSettings):
    """Настройки из переменных окружения. Префикс: MAGIC_MASTER_."""
    model_config = SettingsConfigDict(
        env_prefix="MAGIC_MASTER_",
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",  # игнорировать переменные из .env без поля в Settings
    )
    max_upload_mb: int = 100
    # Лимиты по формату (DJ-сеты, длинные WAV): WAV до 800 МБ, MP3 до 300 МБ
    max_upload_mb_wav: int = 800
    max_upload_mb_mp3: int = 300
    max_upload_mb_flac: int = 500
    # Лимиты по тарифу (эффективный лимит = min(тариф, формат))
    max_upload_mb_free: int = 100
    max_upload_mb_pro: int = 300
    max_upload_mb_studio: int = 800
    allowed_extensions: set[str] = {"wav", "mp3", "flac"}
    temp_dir: str = "/tmp/masterflow"
    default_target_lufs: float = -14.0
    jobs_max_entries: int = 100
    jobs_done_ttl_seconds: int = 3600
    # Семафоры одновременных задач мастеринга: Pro/Studio и Free (на мощном сервере увеличьте)
    semaphore_priority: int = 2   # MAGIC_MASTER_SEMAPHORE_PRIORITY
    semaphore_normal: int = 1     # MAGIC_MASTER_SEMAPHORE_NORMAL
    # Режим отладки: все функции без авторизации (MAGIC_MASTER_DEBUG=1 или MAGIC_MASTER_DEBUG_MODE=1)
    debug_mode: bool = Field(False, validation_alias="DEBUG")

    # AI agents: backend (openai | deepseek | anthropic | local), ключи и лимиты по тарифам
    ai_backend: str = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    # DeepSeek (API совместим с OpenAI)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    # Дневные лимиты AI-запросов: Free, Pro, Studio (-1 = без лимита)
    ai_limit_free: int = 5
    ai_limit_pro: int = 50
    ai_limit_studio: int = -1
    # Опциональные промпты LLM (если пусто — используются встроенные). MAGIC_MASTER_AI_PROMPT_*.
    ai_prompt_recommend: str = ""   # рекомендация пресета по анализу
    ai_prompt_report: str = ""      # отчёт по анализу + рекомендации
    ai_prompt_nl_config: str = ""   # NL → настройки цепочки
    ai_prompt_chat: str = ""        # системный промпт чат-помощника

    # Email verification (P41): если True — новые пользователи должны подтвердить email
    require_email_verify: bool = Field(False, validation_alias="REQUIRE_EMAIL_VERIFY")

    # P46: глобальный rate limit — запросов/минуту с одного IP для API
    global_rate_limit: int = 300

    # P56: CORS — список разрешённых origins через запятую; пусто = ["*"] (обратная совместимость)
    cors_origins: str = ""

    # P56: YooKassa webhook — опциональный whitelist IP через запятую; пусто = проверка отключена
    yookassa_webhook_ip_whitelist: str = ""

    # Admin panel (P18): первый admin создаётся через env
    admin_email: str = ""
    admin_password: str = ""  # MAGIC_MASTER_ADMIN_PASSWORD; если пусто при создании — используется changeMe123!

    # SMTP для рассылок (P22)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_use_tls: bool = True

    # YooKassa (P23)
    yookassa_shop_id: str = ""
    yookassa_secret_key: str = ""
    # URL на который YooKassa редиректит после оплаты
    yookassa_return_url: str = "http://localhost:8000/pricing"

    # Telegram уведомления для админа (P51)
    # Создайте бота через @BotFather, узнайте chat_id через @userinfobot
    telegram_bot_token: str = ""
    telegram_admin_chat_id: str = ""
    # Секрет webhook бота уведомлений (если задан — проверяется на POST /bot/notify/webhook)
    telegram_bot_webhook_secret: str = ""
    # Ссылка на клиентский бот (кнопки меню в уведомительном боте ведут пользователя туда)
    user_bot_telegram_url: str = "https://t.me/magicmasterpro_user_bot"

    # Telegram user bot (отдельный от админ-уведомлений): интерактивный бот + webhook
    user_bot_token: str = ""
    user_bot_webhook_secret: str = ""
    # Канал для постов (например @PROmagicmaster или -100...)
    user_bot_channel_id: str = ""
    # Публичный URL сайта (https://magicmaster.pro) — ссылки и setWebhook
    public_base_url: str = ""

    # Расширения: дополнительный путь к пресетам сообщества (файл .json или каталог с .json)
    # MAGIC_MASTER_COMMUNITY_PRESETS_EXTRA — подгружаются после presets_community.json
    community_presets_extra: str = ""

    # Алерты мониторинга (3.3): уведомления в Telegram при деградации health или превышении порога очереди
    # MAGIC_MASTER_ALERT_MONITORING_ENABLED=1, MAGIC_MASTER_ALERT_QUEUE_THRESHOLD=20 (0 = выкл)
    alert_monitoring_enabled: bool = False
    alert_queue_threshold: int = 0
    alert_throttle_minutes: int = 60

    # Фоновый мониторинг аномалий (работает при alert_monitoring_enabled=1)
    anomaly_check_interval: int = 60
    anomaly_cpu_threshold: float = 85.0
    anomaly_ram_threshold: float = 85.0
    anomaly_disk_threshold: float = 85.0
    anomaly_error_rate_threshold: float = 35.0
    anomaly_rss_max_mb: int = 8192
    anomaly_min_jobs_for_error_rate: int = 8

    # Задача 9.2 — изоляция вокала (Demucs). При False эндпоинт /api/v2/isolate-vocal возвращает 503.
    # MAGIC_MASTER_ENABLE_VOCAL_ISOLATION=1, MAGIC_MASTER_DEMUCS_MODEL=htdemucs
    enable_vocal_isolation: bool = Field(False, validation_alias="ENABLE_VOCAL_ISOLATION")
    demucs_model: str = "htdemucs"

    @field_validator("debug_mode", "require_email_verify", "enable_vocal_isolation", mode="before")
    @classmethod
    def parse_bool_flag(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)


settings = Settings()
