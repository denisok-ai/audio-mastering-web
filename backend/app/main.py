# @file main.py
# @description FastAPI-приложение: загрузка, мастеринг, экспорт, замер LUFS
# @dependencies app.pipeline, app.config
# @created 2026-02-26

import asyncio
import io
import os
import uuid

import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .config import settings
from .pipeline import (
    PRESET_LUFS,
    export_audio,
    load_audio_from_bytes,
    measure_lufs,
    run_mastering_pipeline,
)

# Фоновые задачи мастеринга: job_id -> { status, progress, message, result_bytes?, filename?, error? }
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


@app.get("/api")
def api_root():
    return {"service": "Magic Master API", "docs": "/docs"}


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


def _run_mastering_job(
    job_id: str,
    data: bytes,
    filename: str,
    target_lufs: float,
    out_format: str,
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
            audio, sr, target_lufs=target_lufs, progress_callback=on_progress
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
    except Exception as e:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = ""
        job["error"] = str(e)


@app.post("/api/master")
async def api_master(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_lufs: float = Form(-14.0),
    preset: Optional[str] = Form(None),
    out_format: str = Form("wav"),
):
    """
    Запуск мастеринга: возвращает job_id. Статус — GET /api/master/status/{job_id}.
    Результат — GET /api/master/result/{job_id} после status=done.
    """
    if not _allowed_file(file.filename or ""):
        raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
    if out_format.lower() not in ("wav", "mp3", "flac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3 или flac.")
    if preset and preset.lower() in PRESET_LUFS:
        target_lufs = PRESET_LUFS[preset.lower()]
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {settings.max_upload_mb} МБ.")
    try:
        load_audio_from_bytes(data, file.filename or "wav")
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Ожидание…",
        "result_bytes": None,
        "filename": None,
        "error": None,
        "before_lufs": None,
        "after_lufs": None,
        "target_lufs": target_lufs,
    }
    async def run_job():
        await asyncio.to_thread(
            _run_mastering_job,
            job_id,
            data,
            file.filename or "audio.wav",
            target_lufs,
            out_format.lower(),
        )

    background_tasks.add_task(run_job)
    return {"job_id": job_id, "status": "running"}


@app.get("/api/master/status/{job_id}")
async def api_master_status(job_id: str):
    """Статус задачи мастеринга: progress 0–100, message."""
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
