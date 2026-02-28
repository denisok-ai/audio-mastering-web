"""SQLite database setup, User model, and MasteringRecord model via SQLAlchemy."""
import time
from pathlib import Path
from typing import List, Optional

try:
    from sqlalchemy import Boolean, Text, create_engine, Column, ForeignKey, Integer, String, Float
    from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    DATABASE_URL: str = ""  # type: ignore[assignment]
    # Заглушки для type hints — чтобы main.py не падал при импорте
    Session = object  # type: ignore[misc,assignment]

if DB_AVAILABLE:
    # DB file рядом с бэкендом (выше app/)
    _DB_PATH = Path(__file__).resolve().parent.parent / "magic_master.db"
    _DB_URL = f"sqlite:///{_DB_PATH}"
    DATABASE_URL: str = _DB_URL  # публичный алиас для бэкапа (P50)

    engine = create_engine(
        _DB_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    class Base(DeclarativeBase):  # type: ignore[misc]
        pass
else:
    engine = None  # type: ignore[assignment]
    SessionLocal = None  # type: ignore[assignment]
    Base = None  # type: ignore[assignment]


if DB_AVAILABLE:
    class User(Base):  # type: ignore[misc]
        __tablename__ = "users"

        id: int = Column(Integer, primary_key=True, index=True)
        email: str = Column(String(255), unique=True, index=True, nullable=False)
        hashed_password: str = Column(String(255), nullable=False)
        tier: str = Column(String(32), nullable=False, default="pro")
        created_at: float = Column(Float, nullable=False, default=time.time)
        last_login_at: Optional[float] = Column(Float, nullable=True)
        # Admin panel extensions (P18)
        is_admin: bool = Column(Boolean, nullable=False, default=False)
        is_blocked: bool = Column(Boolean, nullable=False, default=False)
        subscription_expires_at: Optional[float] = Column(Float, nullable=True)
        subscription_status: str = Column(String(32), nullable=False, default="none")
        # none | active | expired | cancelled
        subscription_warning_sent: bool = Column(Boolean, nullable=False, default=False)
        # Email verification (P41): True = подтверждён; по умолчанию True для legacy-аккаунтов
        is_verified: bool = Column(Boolean, nullable=False, default=True)

    class MasteringRecord(Base):  # type: ignore[misc]
        """История мастерингов для залогиненных пользователей."""
        __tablename__ = "mastering_records"

        id: int = Column(Integer, primary_key=True, index=True)
        user_id: int = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
        filename: str = Column(String(512), nullable=False, default="")
        style: str = Column(String(64), nullable=False, default="standard")
        out_format: str = Column(String(16), nullable=False, default="wav")
        before_lufs: Optional[float] = Column(Float, nullable=True)
        after_lufs: Optional[float] = Column(Float, nullable=True)
        target_lufs: Optional[float] = Column(Float, nullable=True)
        duration_sec: Optional[float] = Column(Float, nullable=True)
        created_at: float = Column(Float, nullable=False, default=time.time)

    class SavedPreset(Base):  # type: ignore[misc]
        """Сохранённые пресеты цепочки мастеринга для залогиненных пользователей (P10)."""
        __tablename__ = "saved_presets"

        id: int = Column(Integer, primary_key=True, index=True)
        user_id: int = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
        name: str = Column(String(255), nullable=False, default="")
        config: str = Column(String(8192), nullable=False, default="{}")  # JSON цепочки модулей
        style: str = Column(String(64), nullable=False, default="standard")
        target_lufs: float = Column(Float, nullable=False, default=-14.0)
        created_at: float = Column(Float, nullable=False, default=time.time)

    class Transaction(Base):  # type: ignore[misc]
        """Финансовые транзакции: ручные и YooKassa (P18/P23)."""
        __tablename__ = "transactions"

        id: int = Column(Integer, primary_key=True, index=True)
        user_id: int = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
        amount: float = Column(Float, nullable=False, default=0.0)
        currency: str = Column(String(8), nullable=False, default="RUB")
        tier: str = Column(String(32), nullable=False, default="pro")
        payment_system: str = Column(String(32), nullable=False, default="manual")
        # manual | yookassa
        external_id: Optional[str] = Column(String(256), nullable=True)
        status: str = Column(String(32), nullable=False, default="succeeded")
        # pending | succeeded | failed | refunded
        description: str = Column(String(512), nullable=False, default="")
        created_at: float = Column(Float, nullable=False, default=time.time)

    class NewsPost(Base):  # type: ignore[misc]
        """Новости и объявления (P18/P21)."""
        __tablename__ = "news_posts"

        id: int = Column(Integer, primary_key=True, index=True)
        title: str = Column(String(512), nullable=False, default="")
        body: str = Column(Text, nullable=False, default="")
        author_id: Optional[int] = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
        is_published: bool = Column(Boolean, nullable=False, default=False)
        published_at: Optional[float] = Column(Float, nullable=True)
        created_at: float = Column(Float, nullable=False, default=time.time)
        updated_at: float = Column(Float, nullable=False, default=time.time)

    class EmailCampaign(Base):  # type: ignore[misc]
        """Email-рассылки для пользователей (P18/P22)."""
        __tablename__ = "email_campaigns"

        id: int = Column(Integer, primary_key=True, index=True)
        subject: str = Column(String(512), nullable=False, default="")
        body_html: str = Column(Text, nullable=False, default="")
        body_text: str = Column(Text, nullable=False, default="")
        target_tier: Optional[str] = Column(String(32), nullable=True)
        # null = все, иначе free|pro|studio
        status: str = Column(String(32), nullable=False, default="draft")
        # draft | sending | sent | failed
        total_recipients: int = Column(Integer, nullable=False, default=0)
        sent_count: int = Column(Integer, nullable=False, default=0)
        created_at: float = Column(Float, nullable=False, default=time.time)
        sent_at: Optional[float] = Column(Float, nullable=True)

    class AuditLog(Base):  # type: ignore[misc]
        """Журнал действий администраторов. P55."""
        __tablename__ = "audit_logs"

        id: int = Column(Integer, primary_key=True, index=True)
        admin_id: Optional[int] = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
        admin_email: str = Column(String(255), nullable=False, default="")
        action: str = Column(String(64), nullable=False, default="")   # block_user, delete_user, set_tier, ...
        target_type: str = Column(String(32), nullable=False, default="")   # user | transaction | news | ...
        target_id: Optional[int] = Column(Integer, nullable=True)
        details: str = Column(String(1024), nullable=False, default="")   # JSON или текст
        ip: str = Column(String(64), nullable=False, default="")
        created_at: float = Column(Float, nullable=False, default=time.time)

    class ApiKey(Base):  # type: ignore[misc]
        """API-ключи для программного доступа (Pro/Studio). P52."""
        __tablename__ = "api_keys"

        id: int = Column(Integer, primary_key=True, index=True)
        user_id: int = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
        name: str = Column(String(255), nullable=False, default="")
        key_prefix: str = Column(String(8), nullable=False, default="")   # первые 8 символов (для отображения)
        key_hash: str = Column(String(255), nullable=False, unique=True)   # SHA-256 hex
        is_active: bool = Column(Boolean, nullable=False, default=True)
        created_at: float = Column(Float, nullable=False, default=time.time)
        last_used_at: Optional[float] = Column(Float, nullable=True)

    class SystemSetting(Base):  # type: ignore[misc]
        """Переопределения настроек из админки (key -> value). Дефолты из .env."""
        __tablename__ = "system_settings"

        id: int = Column(Integer, primary_key=True, index=True)
        key: str = Column(String(128), unique=True, nullable=False, index=True)
        value: str = Column(Text, nullable=False, default="")
        updated_at: float = Column(Float, nullable=False, default=time.time)
        updated_by: Optional[int] = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    class PromptTemplate(Base):  # type: ignore[misc]
        """Версии промптов LLM. Один активный на slug (is_active=True)."""
        __tablename__ = "prompt_templates"

        id: int = Column(Integer, primary_key=True, index=True)
        slug: str = Column(String(64), nullable=False, index=True)  # recommend | report | nl_config | chat
        name: str = Column(String(255), nullable=False, default="")
        body: str = Column(Text, nullable=False, default="")
        is_builtin: bool = Column(Boolean, nullable=False, default=False)
        is_active: bool = Column(Boolean, nullable=False, default=False)
        version: int = Column(Integer, nullable=False, default=1)
        created_at: float = Column(Float, nullable=False, default=time.time)
        created_by: Optional[int] = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    class AiUsageLog(Base):  # type: ignore[misc]
        """Лог вызовов AI по типам для отчётности."""
        __tablename__ = "ai_usage_log"

        id: int = Column(Integer, primary_key=True, index=True)
        user_id: Optional[int] = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
        type: str = Column(String(32), nullable=False, index=True)  # recommend | report | nl_config | chat
        tier: str = Column(String(32), nullable=False, default="free")
        created_at: float = Column(Float, nullable=False, default=time.time)

    class MasteringJobEvent(Base):  # type: ignore[misc]
        """События мастеринга для отчётов (все задачи, не только авторизованные)."""
        __tablename__ = "mastering_job_events"

        id: int = Column(Integer, primary_key=True, index=True)
        user_id: Optional[int] = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
        job_id: str = Column(String(64), nullable=False, index=True)
        style: str = Column(String(64), nullable=False, default="standard")
        status: str = Column(String(32), nullable=False, default="running")  # running | done | error
        created_at: float = Column(Float, nullable=False, default=time.time)
        completed_at: Optional[float] = Column(Float, nullable=True)

else:
    User = None  # type: ignore[assignment,misc]
    MasteringRecord = None  # type: ignore[assignment,misc]
    SavedPreset = None  # type: ignore[assignment,misc]
    Transaction = None  # type: ignore[assignment,misc]
    NewsPost = None  # type: ignore[assignment,misc]
    EmailCampaign = None  # type: ignore[assignment,misc]
    ApiKey = None  # type: ignore[assignment,misc]
    AuditLog = None  # type: ignore[assignment,misc]
    SystemSetting = None  # type: ignore[assignment,misc]
    PromptTemplate = None  # type: ignore[assignment,misc]
    AiUsageLog = None  # type: ignore[assignment,misc]
    MasteringJobEvent = None  # type: ignore[assignment,misc]


def create_tables() -> None:
    """Создать таблицы если не существуют. Вызывается при старте FastAPI."""
    if not DB_AVAILABLE:
        return
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "already exists" not in msg and "duplicate" not in msg:
            raise
    _run_migrations()


def _run_migrations() -> None:
    """P36: Безопасные миграции — добавляет новые колонки в существующие таблицы.

    SQLAlchemy create_all() не изменяет уже существующие таблицы.
    Эта функция использует SQLite PRAGMA + ALTER TABLE ADD COLUMN.
    """
    if not DB_AVAILABLE or engine is None:
        return

    # Описание миграций: {table: [(column_name, sql_type, default_expr)]}
    migrations = {
        "users": [
            ("is_admin",                  "BOOLEAN NOT NULL DEFAULT 0"),
            ("is_blocked",                "BOOLEAN NOT NULL DEFAULT 0"),
            ("subscription_expires_at",   "REAL"),
            ("subscription_status",       "VARCHAR(32) NOT NULL DEFAULT 'none'"),
            ("subscription_warning_sent", "BOOLEAN NOT NULL DEFAULT 0"),
            ("last_login_at",             "REAL"),
            ("is_verified",               "BOOLEAN NOT NULL DEFAULT 1"),
        ],
        "mastering_records": [
            ("duration_sec", "REAL"),
        ],
    }

    with engine.connect() as conn:
        for table, columns in migrations.items():
            try:
                result = conn.execute(
                    __import__("sqlalchemy").text(f"PRAGMA table_info({table})")
                )
                existing = {row[1] for row in result}  # row[1] = column name
            except Exception:  # noqa: BLE001
                continue
            for col_name, col_type in columns:
                if col_name not in existing:
                    try:
                        conn.execute(
                            __import__("sqlalchemy").text(
                                f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                            )
                        )
                        conn.commit()
                    except Exception:  # noqa: BLE001
                        pass


def get_db():
    """FastAPI dependency: генератор сессии БД. Если БД недоступна — возвращает None."""
    if not DB_AVAILABLE or SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_mastering_record(
    db,
    user_id: int,
    filename: str,
    style: str,
    out_format: str,
    before_lufs: Optional[float],
    after_lufs: Optional[float],
    target_lufs: Optional[float],
    duration_sec: Optional[float] = None,
):
    if not DB_AVAILABLE or db is None:
        return None
    rec = MasteringRecord(
        user_id=user_id,
        filename=filename,
        style=style,
        out_format=out_format,
        before_lufs=before_lufs,
        after_lufs=after_lufs,
        target_lufs=target_lufs,
        duration_sec=duration_sec,
        created_at=time.time(),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def get_user_history(db, user_id: int, limit: int = 30) -> List:
    if not DB_AVAILABLE or db is None:
        return []
    return (
        db.query(MasteringRecord)
        .filter(MasteringRecord.user_id == user_id)
        .order_by(MasteringRecord.created_at.desc())
        .limit(limit)
        .all()
    )


def delete_mastering_record(db, record_id: int, user_id: int) -> bool:
    if not DB_AVAILABLE or db is None:
        return False
    rec = db.query(MasteringRecord).filter(
        MasteringRecord.id == record_id,
        MasteringRecord.user_id == user_id,
    ).first()
    if not rec:
        return False
    db.delete(rec)
    db.commit()
    return True


def get_user_stats(db, user_id: int) -> dict:
    if not DB_AVAILABLE or db is None:
        return {"total": 0, "avg_lufs_change": None, "top_style": None}
    records = db.query(MasteringRecord).filter(MasteringRecord.user_id == user_id).all()
    total = len(records)
    if total == 0:
        return {"total": 0, "avg_lufs_change": None, "top_style": None}

    changes = [
        r.after_lufs - r.before_lufs
        for r in records
        if r.before_lufs is not None and r.after_lufs is not None
    ]
    avg_change = round(sum(changes) / len(changes), 2) if changes else None

    style_counts: dict = {}
    for r in records:
        style_counts[r.style] = style_counts.get(r.style, 0) + 1
    top_style = max(style_counts, key=style_counts.get) if style_counts else None

    return {"total": total, "avg_lufs_change": avg_change, "top_style": top_style}


def get_user_by_email(db, email: str):
    if not DB_AVAILABLE or db is None:
        return None
    return db.query(User).filter(User.email == email.lower().strip()).first()


def create_user(db, email: str, hashed_password: str, tier: str = "pro"):
    if not DB_AVAILABLE or db is None:
        return None
    user = User(
        email=email.lower().strip(),
        hashed_password=hashed_password,
        tier=tier,
        created_at=time.time(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# --- SavedPreset (P10) ---

def create_saved_preset(
    db,
    user_id: int,
    name: str,
    config: str,
    style: str = "standard",
    target_lufs: float = -14.0,
):
    if not DB_AVAILABLE or db is None:
        return None
    preset = SavedPreset(
        user_id=user_id,
        name=name.strip() or "Без имени",
        config=config,
        style=style,
        target_lufs=target_lufs,
        created_at=time.time(),
    )
    db.add(preset)
    db.commit()
    db.refresh(preset)
    return preset


def get_user_presets(db, user_id: int, limit: int = 50) -> List:
    if not DB_AVAILABLE or db is None:
        return []
    return (
        db.query(SavedPreset)
        .filter(SavedPreset.user_id == user_id)
        .order_by(SavedPreset.created_at.desc())
        .limit(limit)
        .all()
    )


def get_saved_preset_by_id(db, preset_id: int, user_id: int):
    if not DB_AVAILABLE or db is None:
        return None
    return (
        db.query(SavedPreset)
        .filter(SavedPreset.id == preset_id, SavedPreset.user_id == user_id)
        .first()
    )


def delete_saved_preset(db, preset_id: int, user_id: int) -> bool:
    if not DB_AVAILABLE or db is None:
        return False
    preset = get_saved_preset_by_id(db, preset_id, user_id)
    if not preset:
        return False
    db.delete(preset)
    db.commit()
    return True


# --- Transaction (P18) ---

def create_transaction(
    db,
    user_id: int,
    amount: float,
    tier: str,
    payment_system: str = "manual",
    currency: str = "RUB",
    external_id: Optional[str] = None,
    status: str = "succeeded",
    description: str = "",
):
    if not DB_AVAILABLE or db is None:
        return None
    tx = Transaction(
        user_id=user_id,
        amount=amount,
        currency=currency,
        tier=tier,
        payment_system=payment_system,
        external_id=external_id,
        status=status,
        description=description,
        created_at=time.time(),
    )
    db.add(tx)
    db.commit()
    db.refresh(tx)
    return tx


def get_transactions(db, user_id: Optional[int] = None, status: Optional[str] = None,
                     limit: int = 100, offset: int = 0) -> List:
    if not DB_AVAILABLE or db is None:
        return []
    q = db.query(Transaction)
    if user_id is not None:
        q = q.filter(Transaction.user_id == user_id)
    if status:
        q = q.filter(Transaction.status == status)
    return q.order_by(Transaction.created_at.desc()).offset(offset).limit(limit).all()


def count_transactions(db, user_id: Optional[int] = None, status: Optional[str] = None) -> int:
    """Количество транзакций с учётом фильтров (для пагинации)."""
    if not DB_AVAILABLE or db is None:
        return 0
    q = db.query(Transaction)
    if user_id is not None:
        q = q.filter(Transaction.user_id == user_id)
    if status:
        q = q.filter(Transaction.status == status)
    return q.count()


# --- NewsPost (P18) ---

def create_news_post(db, title: str, body: str, author_id: Optional[int] = None,
                     is_published: bool = False) -> Optional[object]:
    if not DB_AVAILABLE or db is None:
        return None
    now = time.time()
    post = NewsPost(
        title=title,
        body=body,
        author_id=author_id,
        is_published=is_published,
        published_at=now if is_published else None,
        created_at=now,
        updated_at=now,
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    return post


def get_news_posts(db, published_only: bool = True, limit: int = 20, offset: int = 0) -> List:
    if not DB_AVAILABLE or db is None:
        return []
    q = db.query(NewsPost)
    if published_only:
        q = q.filter(NewsPost.is_published == True)  # noqa: E712
    return q.order_by(NewsPost.published_at.desc(), NewsPost.created_at.desc()).offset(offset).limit(limit).all()


def count_news_posts(db, published_only: bool = False) -> int:
    """Количество постов с учётом фильтра (для пагинации)."""
    if not DB_AVAILABLE or db is None:
        return 0
    q = db.query(NewsPost)
    if published_only:
        q = q.filter(NewsPost.is_published == True)  # noqa: E712
    return q.count()


def get_news_post_by_id(db, post_id: int) -> Optional[object]:
    if not DB_AVAILABLE or db is None:
        return None
    return db.query(NewsPost).filter(NewsPost.id == post_id).first()


def update_news_post(db, post_id: int, **kwargs) -> Optional[object]:
    if not DB_AVAILABLE or db is None:
        return None
    post = get_news_post_by_id(db, post_id)
    if not post:
        return None
    now = time.time()
    for key, val in kwargs.items():
        setattr(post, key, val)
    post.updated_at = now
    if kwargs.get("is_published") and not post.published_at:
        post.published_at = now
    db.commit()
    db.refresh(post)
    return post


def delete_news_post(db, post_id: int) -> bool:
    if not DB_AVAILABLE or db is None:
        return False
    post = get_news_post_by_id(db, post_id)
    if not post:
        return False
    db.delete(post)
    db.commit()
    return True


# --- EmailCampaign (P18) ---

def create_email_campaign(db, subject: str, body_html: str, body_text: str,
                           target_tier: Optional[str] = None) -> Optional[object]:
    if not DB_AVAILABLE or db is None:
        return None
    camp = EmailCampaign(
        subject=subject,
        body_html=body_html,
        body_text=body_text,
        target_tier=target_tier,
        status="draft",
        created_at=time.time(),
    )
    db.add(camp)
    db.commit()
    db.refresh(camp)
    return camp


def get_email_campaigns(db, limit: int = 50, offset: int = 0) -> List:
    if not DB_AVAILABLE or db is None:
        return []
    return db.query(EmailCampaign).order_by(EmailCampaign.created_at.desc()).offset(offset).limit(limit).all()


def count_email_campaigns(db) -> int:
    """Общее количество рассылок (для пагинации)."""
    if not DB_AVAILABLE or db is None:
        return 0
    return db.query(EmailCampaign).count()


def get_email_campaign_by_id(db, campaign_id: int) -> Optional[object]:
    if not DB_AVAILABLE or db is None:
        return None
    return db.query(EmailCampaign).filter(EmailCampaign.id == campaign_id).first()


def update_email_campaign(db, campaign_id: int, **kwargs) -> Optional[object]:
    if not DB_AVAILABLE or db is None:
        return None
    camp = get_email_campaign_by_id(db, campaign_id)
    if not camp:
        return None
    for key, val in kwargs.items():
        setattr(camp, key, val)
    db.commit()
    db.refresh(camp)
    return camp


# --- Subscription expiry (P28) ---

def check_and_expire_subscription(db, user_id: int) -> Optional[str]:
    """Проверяет истечение подписки и понижает тариф до 'free' если срок вышел.

    P28/P32: отправляет письмо об истечении (разово), а также письмо-предупреждение за ≤3 дня.
    Возвращает актуальный tier пользователя (после возможного даунгрейда), или None если не найден.
    """
    if not DB_AVAILABLE or db is None:
        return None
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return None

    now = time.time()
    if user.tier != "free" and user.subscription_expires_at is not None:
        expired = user.subscription_expires_at < now
        if expired:
            old_tier = user.tier
            user.tier = "free"
            user.subscription_status = "expired"
            db.commit()
            # Письмо об истечении — только если ещё не отправляли
            if not getattr(user, "subscription_warning_sent", False):
                try:
                    from .mailer import send_subscription_expired_email
                    send_subscription_expired_email(user.email, old_tier)
                except Exception:  # noqa: BLE001
                    pass
        else:
            # Проверяем — меньше 3 дней до истечения, письмо ещё не отправляли
            days_left_sec = user.subscription_expires_at - now
            warning_days = 3
            if (
                days_left_sec <= warning_days * 86400
                and not getattr(user, "subscription_warning_sent", False)
            ):
                days_left = max(1, int(days_left_sec / 86400) + 1)
                try:
                    from .mailer import send_subscription_expiry_warning_email
                    send_subscription_expiry_warning_email(
                        user.email, user.tier, user.subscription_expires_at, days_left
                    )
                    user.subscription_warning_sent = True
                    db.commit()
                except Exception:  # noqa: BLE001
                    pass

    return user.tier


# ─── Audit Log (P55) ─────────────────────────────────────────────────────────

def write_audit_log(
    db,
    admin_id: Optional[int],
    admin_email: str,
    action: str,
    target_type: str = "",
    target_id: Optional[int] = None,
    details: str = "",
    ip: str = "",
) -> None:
    """Записать действие администратора в журнал."""
    if not DB_AVAILABLE or AuditLog is None:
        return
    try:
        entry = AuditLog(
            admin_id=admin_id,
            admin_email=admin_email[:255],
            action=action[:64],
            target_type=target_type[:32],
            target_id=target_id,
            details=details[:1024],
            ip=ip[:64],
            created_at=time.time(),
        )
        db.add(entry)
        db.commit()
    except Exception:  # noqa: BLE001
        pass


def get_audit_logs(
    db,
    limit: int = 50,
    offset: int = 0,
    admin_id: Optional[int] = None,
    action: Optional[str] = None,
    ts_from: Optional[float] = None,
    ts_to: Optional[float] = None,
) -> tuple:
    """Получить записи журнала с пагинацией. Фильтры: admin_id, action, ts_from, ts_to."""
    if not DB_AVAILABLE or AuditLog is None:
        return [], 0
    q = db.query(AuditLog)
    if admin_id is not None:
        q = q.filter(AuditLog.admin_id == admin_id)
    if action and action.strip():
        q = q.filter(AuditLog.action == action.strip())
    if ts_from is not None:
        q = q.filter(AuditLog.created_at >= ts_from)
    if ts_to is not None:
        q = q.filter(AuditLog.created_at <= ts_to)
    total = q.count()
    logs = q.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit).all()
    return logs, total


# ─── Prompt templates (admin LLM prompts) ───────────────────────────────────────

def get_active_prompt_body(db, slug: str) -> Optional[str]:
    """Текст активного промпта по slug; None если нет кастомного (использовать env/builtin)."""
    if not DB_AVAILABLE or PromptTemplate is None:
        return None
    row = (
        db.query(PromptTemplate)
        .filter(PromptTemplate.slug == slug, PromptTemplate.is_active == True)  # noqa: E712
        .first()
    )
    return row.body if row else None


def get_prompt_history(db, slug: str, limit: int = 20) -> List:
    """История версий промпта по slug (последние сначала)."""
    if not DB_AVAILABLE or PromptTemplate is None:
        return []
    return (
        db.query(PromptTemplate)
        .filter(PromptTemplate.slug == slug)
        .order_by(PromptTemplate.created_at.desc())
        .limit(limit)
        .all()
    )


def create_prompt_version(
    db, slug: str, body: str, admin_id: Optional[int] = None, name: str = ""
) -> Optional[object]:
    """Создать новую версию промпта и сделать её активной. version = max+1."""
    if not DB_AVAILABLE or PromptTemplate is None:
        return None
    now = time.time()
    prev = (
        db.query(PromptTemplate)
        .filter(PromptTemplate.slug == slug)
        .order_by(PromptTemplate.version.desc())
        .first()
    )
    next_version = (prev.version + 1) if prev else 1
    db.query(PromptTemplate).filter(PromptTemplate.slug == slug).update({PromptTemplate.is_active: False})
    row = PromptTemplate(
        slug=slug,
        name=name or f"v{next_version}",
        body=body,
        is_builtin=False,
        is_active=True,
        version=next_version,
        created_at=now,
        created_by=admin_id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def set_active_prompt(db, slug: str, template_id: int) -> bool:
    """Сделать указанную версию промпта активной."""
    if not DB_AVAILABLE or PromptTemplate is None:
        return False
    db.query(PromptTemplate).filter(PromptTemplate.slug == slug).update({PromptTemplate.is_active: False})
    row = db.query(PromptTemplate).filter(PromptTemplate.slug == slug, PromptTemplate.id == template_id).first()
    if not row:
        db.rollback()
        return False
    row.is_active = True
    db.commit()
    return True


def reset_prompt_to_builtin(db, slug: str) -> bool:
    """Деактивировать все кастомные промпты по slug (использовать встроенный/env)."""
    if not DB_AVAILABLE or PromptTemplate is None:
        return False
    db.query(PromptTemplate).filter(PromptTemplate.slug == slug).update({PromptTemplate.is_active: False})
    db.commit()
    return True


def log_ai_usage(usage_type: str, user_id: Optional[int] = None, tier: str = "free") -> None:
    """Записать вызов AI в ai_usage_log (собственная сессия)."""
    if not DB_AVAILABLE or AiUsageLog is None or SessionLocal is None:
        return
    try:
        db = SessionLocal()
        try:
            db.add(AiUsageLog(user_id=user_id, type=usage_type, tier=tier, created_at=time.time()))
            db.commit()
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        pass


def log_mastering_job_start(job_id: str, user_id: Optional[int], style: str = "standard") -> None:
    """Записать старт задачи мастеринга."""
    if not DB_AVAILABLE or MasteringJobEvent is None or SessionLocal is None:
        return
    try:
        db = SessionLocal()
        try:
            db.add(MasteringJobEvent(job_id=job_id, user_id=user_id, style=style, status="running", created_at=time.time()))
            db.commit()
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        pass


def log_mastering_job_end(job_id: str, status: str = "done") -> None:
    """Обновить событие мастеринга на завершённое."""
    if not DB_AVAILABLE or MasteringJobEvent is None or SessionLocal is None:
        return
    try:
        db = SessionLocal()
        try:
            row = db.query(MasteringJobEvent).filter(MasteringJobEvent.job_id == job_id).order_by(MasteringJobEvent.created_at.desc()).first()
            if row:
                row.status = status
                row.completed_at = time.time()
                db.commit()
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        pass


# ─── API Keys (P52) ────────────────────────────────────────────────────────────

def create_api_key(db, user_id: int, name: str) -> tuple[object, str]:
    """
    Создать новый API-ключ для пользователя.
    Возвращает (ApiKey ORM-объект, plaintext_key).
    Plaintext отображается ТОЛЬКО один раз — потом только хэш.
    """
    import secrets as _sec
    import hashlib as _hl

    if not DB_AVAILABLE or ApiKey is None:
        raise RuntimeError("БД недоступна")
    raw = "mm_" + _sec.token_urlsafe(32)   # формат: mm_<base64url>
    prefix = raw[:8]
    key_hash = _hl.sha256(raw.encode()).hexdigest()
    key = ApiKey(
        user_id=user_id,
        name=name[:100],
        key_prefix=prefix,
        key_hash=key_hash,
        is_active=True,
        created_at=time.time(),
    )
    db.add(key)
    db.commit()
    db.refresh(key)
    return key, raw


def get_api_keys_for_user(db, user_id: int) -> list:
    """Список API-ключей пользователя."""
    if not DB_AVAILABLE or ApiKey is None:
        return []
    return db.query(ApiKey).filter(ApiKey.user_id == user_id).order_by(ApiKey.created_at.desc()).all()


def revoke_api_key(db, key_id: int, user_id: int) -> bool:
    """Деактивировать ключ (мягкое удаление — ставим is_active=False)."""
    if not DB_AVAILABLE or ApiKey is None:
        return False
    k = db.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.user_id == user_id).first()
    if not k:
        return False
    k.is_active = False
    db.commit()
    return True


def get_user_by_api_key(db, raw_key: str) -> Optional[object]:
    """
    Найти пользователя по plaintext API-ключу.
    Обновляет last_used_at при успехе.
    """
    import hashlib as _hl
    if not DB_AVAILABLE or ApiKey is None:
        return None
    key_hash = _hl.sha256(raw_key.encode()).hexdigest()
    k = db.query(ApiKey).filter(ApiKey.key_hash == key_hash, ApiKey.is_active == True).first()  # noqa: E712
    if not k:
        return None
    k.last_used_at = time.time()
    db.commit()
    return db.query(User).filter(User.id == k.user_id).first()
