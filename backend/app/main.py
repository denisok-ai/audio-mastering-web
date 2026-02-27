# @file main.py
# @description FastAPI-приложение: загрузка, мастеринг, экспорт, замер LUFS
# @dependencies app.pipeline, app.chain, app.config
# @created 2026-02-26

import asyncio
import json
import os
import shutil
import time
import uuid

import numpy as np
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .chain import MasteringChain
from .config import settings
from .pipeline import (
    PRESET_LUFS,
    STYLE_CONFIGS,
    compute_lufs_timeline,
    compute_spectrum_bars,
    compute_vectorscope_points,
    export_audio,
    load_audio_from_bytes,
    measure_lufs,
    measure_stereo_correlation,
    run_mastering_pipeline,
)

# Фоновые задачи мастеринга: job_id -> { status, progress, message, created_at, done_at?, ... }
_jobs: dict[str, dict] = {}

app = FastAPI(
    title="Magic Master — автоматический мастеринг",
    description="Загрузите трек → нажмите Magic Master → скачайте результат с целевой громкостью LUFS.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.temp_dir, exist_ok=True)


def _allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in settings.allowed_extensions


def _prune_jobs() -> None:
    """Удаляет старые завершённые задачи и ограничивает число записей в _jobs."""
    now = time.time()
    ttl = settings.jobs_done_ttl_seconds
    max_entries = settings.jobs_max_entries
    to_remove = []
    for jid, job in _jobs.items():
        if job.get("status") in ("done", "error") and job.get("done_at"):
            if now - job["done_at"] > ttl:
                to_remove.append(jid)
    for jid in to_remove:
        del _jobs[jid]
    if len(_jobs) <= max_entries:
        return
    by_created = sorted(_jobs.items(), key=lambda x: x[1].get("created_at", 0))
    for jid, _ in by_created[: len(_jobs) - max_entries]:
        if jid in _jobs:
            del _jobs[jid]


@app.get("/api")
def api_root():
    return {"service": "Magic Master API", "docs": "/docs"}


@app.get("/api/health")
def api_health():
    """Проверка живости сервиса для деплоя и мониторинга."""
    return {"status": "ok"}


# Корень проекта: backend/app/main.py -> app -> backend -> корень
_PROGRESS_PATH = Path(__file__).resolve().parent.parent.parent / "PROGRESS.md"


@app.get("/api/progress", response_class=Response)
def api_progress():
    """Вернуть содержимое PROGRESS.md (статус выполнения плана)."""
    if not _PROGRESS_PATH.is_file():
        raise HTTPException(404, "PROGRESS.md не найден")
    return Response(
        content=_PROGRESS_PATH.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
    )


@app.get("/api/presets")
def get_presets():
    """Список пресетов целевой громкости (LUFS)."""
    return {"presets": PRESET_LUFS}


@app.post("/api/measure")
async def api_measure(file: UploadFile = File(...)):
    """
    Загрузить файл и вернуть текущую громкость в LUFS.
    Форматы: WAV, MP3, FLAC.
    """
    if not _allowed_file(file.filename or ""):
        raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
    # MP3 требует ffmpeg — проверяем заранее
    fname = file.filename or ""
    if fname.lower().endswith(".mp3") and not shutil.which("ffmpeg"):
        raise HTTPException(
            400,
            "Чтение MP3 требует ffmpeg, который не найден на сервере. "
            "Установите: sudo apt-get install -y ffmpeg",
        )
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {settings.max_upload_mb} МБ.")
    try:
        audio, sr = load_audio_from_bytes(data, file.filename or "wav")
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}")
    lufs = measure_lufs(audio, sr)
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    peak_dbfs = float(20 * np.log10(max(peak, 1e-12)))
    duration = float(len(audio) / sr)
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    return {
        "lufs": lufs,
        "sample_rate": sr,
        "peak_dbfs": round(peak_dbfs, 2),
        "duration": round(duration, 3),
        "channels": channels,
    }


@app.get("/api/styles")
def get_styles():
    """Список жанровых пресетов с параметрами."""
    return {"styles": {k: {"lufs": v["lufs"]} for k, v in STYLE_CONFIGS.items()}}


def _run_mastering_job(
    job_id: str,
    data: bytes,
    filename: str,
    target_lufs: float,
    out_format: str,
    style: str = "standard",
):
    """Синхронный мастеринг в потоке; обновляет _jobs[job_id] (progress, result или error)."""
    job = _jobs[job_id]
    try:
        job["progress"] = 2
        job["message"] = "Загрузка аудио…"
        audio, sr = load_audio_from_bytes(data, filename or "wav")

        job["progress"] = 4
        job["message"] = "Измерение исходного уровня…"
        job["before_lufs"] = measure_lufs(audio, sr)

        def on_progress(pct: int, msg: str):
            job["progress"] = pct
            job["message"] = msg

        job["progress"] = 5
        job["message"] = "Мастеринг…"
        mastered = run_mastering_pipeline(
            audio, sr, target_lufs=target_lufs, style=style, progress_callback=on_progress
        )
        job["after_lufs"] = measure_lufs(mastered, sr)
        job["progress"] = 98
        job["message"] = "Экспорт файла…"
        channels = 1 if mastered.ndim == 1 else mastered.shape[1]
        out_bytes = export_audio(mastered, sr, channels, out_format.lower())
        out_name = (filename or "master").rsplit(".", 1)[0] + f"_mastered.{out_format}"
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Готово"
        job["result_bytes"] = out_bytes
        job["filename"] = out_name
        job["done_at"] = time.time()
    except Exception as e:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = ""
        job["error"] = str(e)
        job["done_at"] = time.time()


def _run_mastering_job_v2(
    job_id: str,
    data: bytes,
    filename: str,
    target_lufs: float,
    out_format: str,
    style: str = "standard",
    chain_config: Optional[dict[str, Any]] = None,
    dither_type: Optional[str] = None,
    auto_blank_sec: Optional[float] = None,
):
    """Мастеринг через MasteringChain (v2). Если chain_config is None — используется default_chain."""
    job = _jobs[job_id]
    try:
        job["progress"] = 2
        job["message"] = "Загрузка аудио…"
        audio, sr = load_audio_from_bytes(data, filename or "wav")

        job["progress"] = 4
        job["message"] = "Измерение исходного уровня…"
        job["before_lufs"] = measure_lufs(audio, sr)

        def on_progress(pct: int, msg: str):
            job["progress"] = pct
            job["message"] = msg

        job["progress"] = 5
        job["message"] = "Мастеринг (v2)…"
        if chain_config:
            chain = MasteringChain.from_config(chain_config)
        else:
            chain = MasteringChain.default_chain(target_lufs=target_lufs, style=style)
        mastered = chain.process(
            audio,
            sr,
            target_lufs=target_lufs,
            style=style,
            progress_callback=on_progress,
        )
        job["after_lufs"] = measure_lufs(mastered, sr)
        job["progress"] = 98
        job["message"] = "Экспорт файла…"
        channels = 1 if mastered.ndim == 1 else mastered.shape[1]
        dt = dither_type or (chain_config or {}).get("dither_type") or "tpdf"
        ab = auto_blank_sec if auto_blank_sec is not None else float((chain_config or {}).get("auto_blank_sec", 0) or 0)
        if dt not in ("tpdf", "ns_e", "ns_itu"):
            dt = "tpdf"
        out_bytes = export_audio(
            mastered, sr, channels, out_format.lower(),
            dither_type=dt,
            auto_blank_sec=max(0.0, ab),
        )
        out_name = (filename or "master").rsplit(".", 1)[0] + f"_mastered.{out_format}"
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Готово"
        job["result_bytes"] = out_bytes
        job["filename"] = out_name
        job["done_at"] = time.time()
    except Exception as e:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = ""
        job["error"] = str(e)
        job["done_at"] = time.time()


@app.post("/api/master")
async def api_master(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_lufs: Optional[float] = Form(None),
    preset: Optional[str] = Form(None),
    out_format: str = Form("wav"),
    style: str = Form("standard"),
):
    """
    Запуск мастеринга: возвращает job_id. Статус — GET /api/master/status/{job_id}.
    Результат — GET /api/master/result/{job_id} после status=done.
    """
    if not _allowed_file(file.filename or ""):
        raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
    if out_format.lower() not in ("wav", "mp3", "flac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3 или flac.")
    # MP3 требует ffmpeg — проверяем заранее, чтобы не запускать задачу впустую
    if out_format.lower() == "mp3" and not shutil.which("ffmpeg"):
        raise HTTPException(
            400,
            "Экспорт в MP3 требует ffmpeg, который не найден на сервере. "
            "Установите: sudo apt-get install -y ffmpeg",
        )
    if target_lufs is None:
        target_lufs = settings.default_target_lufs
    if preset and preset.lower() in PRESET_LUFS:
        target_lufs = PRESET_LUFS[preset.lower()]
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {settings.max_upload_mb} МБ.")
    try:
        load_audio_from_bytes(data, file.filename or "wav")
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}")

    _prune_jobs()
    job_id = str(uuid.uuid4())
    style_key = style.lower() if style.lower() in STYLE_CONFIGS else "standard"
    _jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Ожидание…",
        "created_at": time.time(),
        "result_bytes": None,
        "filename": None,
        "error": None,
        "before_lufs": None,
        "after_lufs": None,
        "target_lufs": target_lufs,
        "style": style_key,
    }
    async def run_job():
        await asyncio.to_thread(
            _run_mastering_job,
            job_id,
            data,
            file.filename or "audio.wav",
            target_lufs,
            out_format.lower(),
            style_key,
        )

    background_tasks.add_task(run_job)
    return {"job_id": job_id, "status": "running"}


@app.post("/api/v2/master")
async def api_master_v2(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    target_lufs: Optional[float] = Form(None),
    style: str = Form("standard"),
    out_format: str = Form("wav"),
    dither_type: Optional[str] = Form(None),
    auto_blank_sec: Optional[float] = Form(None),
):
    """
    Мастеринг v2: цепочка из JSON-конфига.
    Форма: file (обязательно), config (JSON-строка, опционально), target_lufs, style, out_format,
    dither_type (tpdf | ns_e | ns_itu), auto_blank_sec (сек, обрезка тишины в конце).
    Ответ: job_id; статус и результат — те же GET /api/master/status/{job_id}, GET /api/master/result/{job_id}.
    """
    if not _allowed_file(file.filename or ""):
        raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
    if out_format.lower() not in ("wav", "mp3", "flac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3 или flac.")
    if out_format.lower() == "mp3" and not shutil.which("ffmpeg"):
        raise HTTPException(
            400,
            "Экспорт в MP3 требует ffmpeg. Установите: sudo apt-get install -y ffmpeg",
        )
    if target_lufs is None:
        target_lufs = settings.default_target_lufs
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {settings.max_upload_mb} МБ.")
    try:
        load_audio_from_bytes(data, file.filename or "wav")
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}")

    chain_config: Optional[dict[str, Any]] = None
    if config and config.strip():
        try:
            chain_config = json.loads(config)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Неверный JSON в config: {e}")

    _prune_jobs()
    job_id = str(uuid.uuid4())
    style_key = style.lower() if style.lower() in STYLE_CONFIGS else "standard"
    _jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Ожидание…",
        "created_at": time.time(),
        "result_bytes": None,
        "filename": None,
        "error": None,
        "before_lufs": None,
        "after_lufs": None,
        "target_lufs": target_lufs,
        "style": style_key,
    }

    async def run_job_v2():
        await asyncio.to_thread(
            _run_mastering_job_v2,
            job_id,
            data,
            file.filename or "audio.wav",
            target_lufs,
            out_format.lower(),
            style_key,
            chain_config,
            dither_type=dither_type,
            auto_blank_sec=auto_blank_sec,
        )

    background_tasks.add_task(run_job_v2)
    return {"job_id": job_id, "status": "running", "version": "v2"}


# Человекочитаемые названия модулей цепочки (для UI)
CHAIN_MODULE_LABELS = {
    "dc_offset": "Удаление DC-смещения",
    "peak_guard": "Защита от пиков",
    "target_curve": "Студийный EQ (Ozone 5 Equalizer)",
    "dynamics": "Многополосная динамика (Ozone 5 Dynamics)",
    "maximizer": "Максимайзер",
    "normalize_lufs": "Нормализация LUFS",
    "final_spectral_balance": "Финальная частотная коррекция",
    "style_eq": "Жанровый EQ",
    "exciter": "Гармонический эксайтер (Ozone 5 Exciter)",
    "imager": "Стерео-расширение (Ozone 5 Imager)",
    "reverb": "Ревербератор (plate/room/hall/theater/cathedral)",
}


@app.get("/api/v2/chain/default")
def api_v2_chain_default(style: str = "standard", target_lufs: float = -14.0):
    """
    Конфиг цепочки по умолчанию: полный список модулей с id, label, enabled и параметрами.
    Ответ можно отправить в POST /api/v2/master (поле config) — в т.ч. с изменённым порядком модулей.
    При отправке поле label можно не передавать (игнорируется бэкендом).
    """
    config = MasteringChain.default_config(target_lufs=target_lufs, style=style)
    modules = []
    for m in config["modules"]:
        m = dict(m)
        mid = m.get("id")
        if mid:
            m["label"] = CHAIN_MODULE_LABELS.get(mid, mid)
            modules.append(m)
    return {"version": "v2", "style": style, "target_lufs": target_lufs, "modules": modules}


@app.post("/api/v2/analyze")
async def api_v2_analyze(file: UploadFile = File(...), extended: bool = Form(False)):
    """
    Загрузить файл и вернуть анализ: LUFS, peak dBFS, длительность, sample rate, stereo_correlation.
    extended=true: дополнительно spectrum_bars, lufs_timeline, timeline_step_sec; для стерео — vectorscope_points (массив [l, r] до 1000 точек).
    """
    if not _allowed_file(file.filename or ""):
        raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
    fname = file.filename or ""
    if fname.lower().endswith(".mp3") and not shutil.which("ffmpeg"):
        raise HTTPException(
            400,
            "Чтение MP3 требует ffmpeg. Установите: sudo apt-get install -y ffmpeg",
        )
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {settings.max_upload_mb} МБ.")
    try:
        audio, sr = load_audio_from_bytes(data, fname)
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}")
    lufs = measure_lufs(audio, sr)
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    peak_dbfs = float(20 * np.log10(max(peak, 1e-12)))
    duration_sec = float(len(audio) / sr)
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    correlation = None
    if channels == 2 and audio.ndim == 2:
        correlation = measure_stereo_correlation(audio)
    out = {
        "version": "v2",
        "lufs": round(lufs, 2) if not np.isnan(lufs) else None,
        "peak_dbfs": round(peak_dbfs, 2),
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sr,
        "channels": channels,
    }
    if correlation is not None:
        out["stereo_correlation"] = round(correlation, 4)
    if extended:
        if audio.size >= 4096:
            try:
                out["spectrum_bars"] = compute_spectrum_bars(audio, sr)
            except Exception:
                pass
        try:
            lufs_timeline, timeline_step_sec = compute_lufs_timeline(audio, sr)
            out["lufs_timeline"] = lufs_timeline
            out["timeline_step_sec"] = timeline_step_sec
        except Exception:
            pass
        if channels == 2 and audio.ndim == 2:
            try:
                out["vectorscope_points"] = compute_vectorscope_points(audio, max_points=1000)
            except Exception:
                pass
    return out


@app.get("/api/master/status/{job_id}")
async def api_master_status(job_id: str):
    """Статус задачи мастеринга: progress 0–100, message."""
    _prune_jobs()
    if job_id not in _jobs:
        raise HTTPException(404, "Задача не найдена")
    job = _jobs[job_id]
    return {
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job.get("error"),
        "before_lufs": job.get("before_lufs"),
        "after_lufs": job.get("after_lufs"),
        "target_lufs": job.get("target_lufs"),
        "style": job.get("style", "standard"),
    }


@app.get("/api/master/result/{job_id}")
async def api_master_result(job_id: str):
    """Скачать результат мастеринга (после status=done)."""
    if job_id not in _jobs:
        raise HTTPException(404, "Задача не найдена")
    job = _jobs[job_id]
    if job["status"] != "done" or not job.get("result_bytes"):
        raise HTTPException(400, "Результат ещё не готов или задача с ошибкой")
    out_bytes = job["result_bytes"]
    filename = job["filename"] or "mastered.wav"
    del _jobs[job_id]
    return Response(
        content=out_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# Раздача фронтенда — после всех API-маршрутов, иначе POST /api/measure отдаёт 405 Method Not Allowed
_frontend = Path(__file__).resolve().parent.parent.parent / "frontend"
if _frontend.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
