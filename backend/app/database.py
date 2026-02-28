"""SQLite database setup, User model, and MasteringRecord model via SQLAlchemy."""
import time
from pathlib import Path
from typing import List, Optional

try:
    from sqlalchemy import create_engine, Column, ForeignKey, Integer, String, Float
    from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    # Заглушки для type hints — чтобы main.py не падал при импорте
    Session = object  # type: ignore[misc,assignment]

if DB_AVAILABLE:
    # DB file рядом с бэкендом (выше app/)
    _DB_PATH = Path(__file__).resolve().parent.parent / "magic_master.db"
    _DB_URL = f"sqlite:///{_DB_PATH}"

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

else:
    User = None  # type: ignore[assignment,misc]
    MasteringRecord = None  # type: ignore[assignment,misc]
    SavedPreset = None  # type: ignore[assignment,misc]


def create_tables() -> None:
    """Создать таблицы если не существуют. Вызывается при старте FastAPI."""
    if not DB_AVAILABLE:
        return
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:  # noqa: BLE001
        # SQLite: индекс/таблица уже есть (миграции или повторный старт) — не падать
        msg = str(e).lower()
        if "already exists" not in msg and "duplicate" not in msg:
            raise


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
