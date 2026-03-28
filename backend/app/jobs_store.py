# @file jobs_store.py
# @description Хранилище задач мастеринга: SQLite-персистентность + in-memory cache.
#   Задачи не теряются при рестарте сервиса.
#   Семафоры параллельного выполнения (priority/normal) определены здесь.
# @created 2026-03-01

import asyncio
import json
import logging
import time
from typing import Any, Optional

from .config import settings

logger = logging.getLogger(__name__)

# ─── Семафоры одновременных задач мастеринга ─────────────────────────────────
# Значения из .env: MAGIC_MASTER_SEMAPHORE_PRIORITY / MAGIC_MASTER_SEMAPHORE_NORMAL
sem_priority = asyncio.Semaphore(max(1, getattr(settings, "semaphore_priority", 2)))
sem_normal = asyncio.Semaphore(max(1, getattr(settings, "semaphore_normal", 1)))

# ─── In-memory кэш задач ──────────────────────────────────────────────────────
# job_id → {status, progress, message, created_at, done_at?, result_path?, error?}
_jobs: dict[str, dict] = {}

# ─── SQLite backend (опционально: если БД доступна) ──────────────────────────
_DB_JOBS_ENABLED = False

try:
    from .database import DB_AVAILABLE, engine as _db_engine, SessionLocal as _SessionLocal
    import sqlalchemy as _sa

    if DB_AVAILABLE and _db_engine is not None:
        _metadata = _sa.MetaData()
        _mastering_jobs_table = _sa.Table(
            "mastering_jobs",
            _metadata,
            _sa.Column("job_id", _sa.String(64), primary_key=True),
            _sa.Column("status", _sa.String(16), nullable=False, default="pending"),
            _sa.Column("progress", _sa.Float, nullable=False, default=0.0),
            _sa.Column("message", _sa.Text, nullable=True),
            _sa.Column("created_at", _sa.Float, nullable=False),
            _sa.Column("done_at", _sa.Float, nullable=True),
            _sa.Column("result_path", _sa.Text, nullable=True),
            _sa.Column("preview_path", _sa.Text, nullable=True),
            _sa.Column("error", _sa.Text, nullable=True),
            _sa.Column("user_id", _sa.Integer, nullable=True),
            _sa.Column("meta_json", _sa.Text, nullable=True),  # доп. поля в JSON
        )
        _metadata.create_all(_db_engine, checkfirst=True)
        _DB_JOBS_ENABLED = True
        logger.info("jobs_store: SQLite-персистентность задач включена")
except Exception as _e:  # noqa: BLE001
    logger.warning("jobs_store: SQLite-персистентность недоступна (%s), используем in-memory", _e)


# ─── Публичный API хранилища ──────────────────────────────────────────────────

def create_job(job_id: str, user_id: Optional[int] = None, **meta: Any) -> dict:
    """Создать новую задачу и сохранить в кэш + SQLite."""
    now = time.time()
    job: dict = {
        "status": "pending",
        "progress": 0.0,
        "message": "В очереди...",
        "created_at": now,
        "done_at": None,
        "result_path": None,
        "preview_path": None,
        "error": None,
        "user_id": user_id,
        **meta,
    }
    _jobs[job_id] = job
    if _DB_JOBS_ENABLED:
        _db_create(job_id, job)
    return job


def get_job(job_id: str) -> Optional[dict]:
    """Вернуть задачу из кэша (быстро). При отсутствии — попытаться восстановить из SQLite."""
    if job_id in _jobs:
        return _jobs[job_id]
    if _DB_JOBS_ENABLED:
        job = _db_load(job_id)
        if job:
            _jobs[job_id] = job
            return job
    return None


def update_job(job_id: str, **fields: Any) -> None:
    """Обновить поля задачи. Пишет в кэш и асинхронно в SQLite."""
    job = _jobs.get(job_id)
    if job is None:
        return
    job.update(fields)
    if _DB_JOBS_ENABLED:
        _db_update(job_id, fields)


def finish_job(job_id: str, result_path: Optional[str] = None,
               preview_path: Optional[str] = None, error: Optional[str] = None) -> None:
    """Завершить задачу: установить status=done/error, done_at, пути к файлам."""
    now = time.time()
    status = "error" if error else "done"
    update_job(
        job_id,
        status=status,
        progress=100.0 if not error else _jobs.get(job_id, {}).get("progress", 0.0),
        done_at=now,
        result_path=result_path,
        preview_path=preview_path,
        error=error,
    )


def prune_jobs() -> None:
    """Удалить старые завершённые задачи из кэша. Вызывается периодически."""
    from . import settings_store
    now = time.time()
    ttl = settings_store.get_setting_int("jobs_done_ttl_seconds", 3600)
    max_entries = settings.jobs_max_entries
    to_remove = [
        jid for jid, job in _jobs.items()
        if job.get("status") in ("done", "error")
        and job.get("done_at")
        and now - job["done_at"] > ttl
    ]
    for jid in to_remove:
        _jobs.pop(jid, None)
    if len(_jobs) > max_entries:
        by_created = sorted(_jobs.items(), key=lambda x: x[1].get("created_at", 0))
        for jid, _ in by_created[: len(_jobs) - max_entries]:
            _jobs.pop(jid, None)


def all_jobs() -> dict[str, dict]:
    """Вернуть полный кэш задач (для health/metrics)."""
    return _jobs


def list_recent_error_jobs(limit: int = 15) -> list[dict]:
    """Последние задачи со статусом error (текст ошибки из кэша или SQLite)."""
    limit = max(1, min(50, int(limit)))
    seen: set[str] = set()
    rows: list[dict] = []
    if _DB_JOBS_ENABLED:
        try:
            with _db_engine.connect() as conn:
                db_rows = conn.execute(
                    _sa.text(
                        "SELECT job_id, error, created_at, done_at FROM mastering_jobs "
                        "WHERE status = 'error' AND error IS NOT NULL AND TRIM(error) != '' "
                        "ORDER BY COALESCE(done_at, created_at) DESC LIMIT :lim"
                    ),
                    {"lim": limit},
                ).fetchall()
            for r in db_rows:
                jid = str(r[0])
                seen.add(jid)
                rows.append(
                    {
                        "job_id": jid,
                        "error": (r[1] or "")[:400],
                        "created_at": float(r[2] or 0),
                        "done_at": float(r[3]) if r[3] is not None else None,
                    }
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("list_recent_error_jobs db: %s", e)
    for jid, j in _jobs.items():
        if j.get("status") != "error":
            continue
        err = (j.get("error") or "").strip()
        if not err or jid in seen:
            continue
        rows.append(
            {
                "job_id": jid,
                "error": err[:400],
                "created_at": float(j.get("created_at") or 0),
                "done_at": j.get("done_at"),
            }
        )
    rows.sort(key=lambda x: float(x.get("done_at") or x.get("created_at") or 0), reverse=True)
    return rows[:limit]


def restore_from_db() -> int:
    """При старте: восстановить незавершённые задачи из SQLite в память.
    Помечает все 'running' задачи как 'error' (сервис перезапустился)."""
    if not _DB_JOBS_ENABLED:
        return 0
    count = 0
    try:
        with _db_engine.connect() as conn:
            rows = conn.execute(
                _sa.text(
                    "SELECT job_id, status, progress, message, created_at, done_at, "
                    "result_path, preview_path, error, user_id, meta_json "
                    "FROM mastering_jobs WHERE done_at IS NULL OR done_at > :cutoff"
                ),
                {"cutoff": time.time() - 7200},
            ).fetchall()
        for row in rows:
            jid = row[0]
            status = row[1]
            if status == "running":
                status = "error"
                _db_update(jid, {"status": "error", "error": "Сервис был перезапущен"})
            meta = {}
            if row[10]:
                try:
                    meta = json.loads(row[10])
                except Exception:  # noqa: BLE001
                    pass
            _jobs[jid] = {
                "status": status,
                "progress": row[2] or 0.0,
                "message": row[3] or "",
                "created_at": row[4] or 0.0,
                "done_at": row[5],
                "result_path": row[6],
                "preview_path": row[7],
                "error": row[8],
                "user_id": row[9],
                **meta,
            }
            count += 1
    except Exception as e:  # noqa: BLE001
        logger.warning("jobs_store: не удалось восстановить задачи из БД: %s", e)
    return count


# ─── SQLite helpers ───────────────────────────────────────────────────────────

def _db_create(job_id: str, job: dict) -> None:
    try:
        known = {"status", "progress", "message", "created_at", "done_at",
                 "result_path", "preview_path", "error", "user_id"}
        meta_extra = {k: v for k, v in job.items() if k not in known}
        with _db_engine.begin() as conn:
            conn.execute(_mastering_jobs_table.insert().values(
                job_id=job_id,
                status=job.get("status", "pending"),
                progress=float(job.get("progress", 0.0)),
                message=job.get("message"),
                created_at=float(job.get("created_at", time.time())),
                done_at=job.get("done_at"),
                result_path=job.get("result_path"),
                preview_path=job.get("preview_path"),
                error=job.get("error"),
                user_id=job.get("user_id"),
                meta_json=json.dumps(meta_extra) if meta_extra else None,
            ))
    except Exception as e:  # noqa: BLE001
        logger.debug("jobs_store._db_create: %s", e)


def _db_update(job_id: str, fields: dict) -> None:
    try:
        known_cols = {"status", "progress", "message", "done_at",
                      "result_path", "preview_path", "error"}
        update_data = {k: v for k, v in fields.items() if k in known_cols}
        if not update_data:
            return
        with _db_engine.begin() as conn:
            conn.execute(
                _sa.text(
                    "UPDATE mastering_jobs SET " +
                    ", ".join(f"{k} = :{k}" for k in update_data) +
                    " WHERE job_id = :job_id"
                ),
                {"job_id": job_id, **update_data},
            )
    except Exception as e:  # noqa: BLE001
        logger.debug("jobs_store._db_update: %s", e)


def _db_load(job_id: str) -> Optional[dict]:
    try:
        with _db_engine.connect() as conn:
            row = conn.execute(
                _sa.text(
                    "SELECT status, progress, message, created_at, done_at, "
                    "result_path, preview_path, error, user_id, meta_json "
                    "FROM mastering_jobs WHERE job_id = :jid"
                ),
                {"jid": job_id},
            ).fetchone()
        if not row:
            return None
        meta = {}
        if row[9]:
            try:
                meta = json.loads(row[9])
            except Exception:  # noqa: BLE001
                pass
        return {
            "status": row[0],
            "progress": row[1] or 0.0,
            "message": row[2] or "",
            "created_at": row[3] or 0.0,
            "done_at": row[4],
            "result_path": row[5],
            "preview_path": row[6],
            "error": row[7],
            "user_id": row[8],
            **meta,
        }
    except Exception as e:  # noqa: BLE001
        logger.debug("jobs_store._db_load: %s", e)
        return None
