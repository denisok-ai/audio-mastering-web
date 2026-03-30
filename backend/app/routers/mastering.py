"""Mastering API router — /api/master/* и /api/v2/* endpoints.

Синхронные функции обработки (_run_mastering_job, _run_mastering_job_v2)
запускаются в фоне через BackgroundTasks + asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from ..chain import MasteringChain
from ..config import settings
from .. import settings_store
from ..database import (
    DB_AVAILABLE,
    SessionLocal,
    deduct_tokens,
    get_db,
    get_user_tokens_balance,
    count_mastering_jobs_today,
    log_mastering_job_end,
    log_mastering_job_start,
    process_referral_on_invitee_mastering_done,
)
from ..helpers import (
    allowed_file as _allowed_file,
    check_audio_magic_bytes as _check_audio_magic_bytes,
    get_client_ip as _get_client_ip,
    json_safe_float as _json_safe_float,
    safe_content_disposition_filename as _safe_content_disposition_filename,
)
from ..deps import (
    check_rate_limit as _check_rate_limit,
    get_ai_identifier as _get_ai_identifier,
    get_current_user_optional as _get_current_user_optional,
    get_tier_for_ai as _get_tier_for_ai,
    is_priority_user as _is_priority_user,
    record_usage as _record_usage,
    require_feature_batch as _require_feature_batch,
)
from ..mastering_trace import (
    TraceContext,
    trace_chain_modules,
    trace_job_done,
    trace_job_error,
    trace_job_start,
    trace_stage,
)
from ..pipeline import (
    DENOISE_PRESETS,
    PRESET_LUFS,
    STYLE_CONFIGS,
    apply_deesser,
    apply_dynamic_eq,
    apply_output_edge_fade_in,
    apply_parallel_compression,
    apply_reference_match,
    apply_rumble_filter,
    apply_spectral_denoise,
    validate_mastered_not_silent,
    apply_transient_designer,
    compute_lufs_timeline,
    compute_spectrum_bars,
    compute_vectorscope_points,
    export_audio,
    load_audio_from_bytes,
    measure_lufs,
    measure_stereo_correlation,
    resample_audio,
    run_mastering_pipeline,
)
from .. import jobs_store as _jobs_store
from .. import ai as ai_module

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mastering"])

_BATCH_MAX_FILES = 10

# Человекочитаемые названия модулей цепочки (для UI)
CHAIN_MODULE_LABELS = {
    "dc_offset": "Удаление DC-смещения",
    "peak_guard": "Защита от пиков",
    "target_curve": "Студийный EQ",
    "dynamics": "Многополосная динамика",
    "maximizer": "Максимайзер",
    "normalize_lufs": "Нормализация LUFS",
    "final_spectral_balance": "Финальная частотная коррекция",
    "style_eq": "Жанровый EQ",
    "exciter": "Гармонический эксайтер",
    "imager": "Стерео-расширение",
    "reverb": "Ревербератор (plate/room/hall/theater/cathedral)",
}


def _jobs():
    return _jobs_store.all_jobs()


def _prune():
    _jobs_store.prune_jobs()


def _user_id_from_user(user: Optional[dict]) -> Optional[int]:
    """Безопасно извлекает user_id (int) из payload пользователя. JWT sub может быть строкой-числом или UUID."""
    if not user:
        return None
    sub = user.get("sub")
    if sub is None:
        return None
    try:
        return int(sub)
    except (TypeError, ValueError):
        return None


# ─── Вспомогательные функции rate-limit ───────────────────────────────────────

def _is_debug_mode() -> bool:
    """Режим отладки: из settings или MAGIC_MASTER_DEBUG=1 (как в main.py для HTML)."""
    if getattr(settings, "debug_mode", False):
        return True
    return os.environ.get("MAGIC_MASTER_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _check_mastering_rate_limit(user, request: Request) -> None:
    """Поднимает 429 если гость исчерпал недельный лимит (Free: 1 мастеринг в неделю). В режиме отладки лимит не проверяется."""
    if not user and not _is_debug_mode():
        ip = _get_client_ip(request)
        limit_info = _check_rate_limit(ip)
        if not limit_info["ok"]:
            raise HTTPException(
                429,
                f"Лимит Free-тарифа исчерпан: {limit_info['limit']} мастеринг в неделю. "
                f"Сброс: {limit_info['reset_at']}. Перейдите на Pro для большего лимита.",
            )


def _check_and_deduct_paid_tier(user, db) -> None:
    """Pro/Studio: проверяет баланс токенов и дневной лимит, списывает 1 токен. 429 при превышении."""
    if not user or not db or not DB_AVAILABLE:
        return
    tier = (user.get("tier") or "").lower()
    if tier not in ("pro", "studio"):
        return
    uid = _user_id_from_user(user)
    if not uid:
        return
    balance = get_user_tokens_balance(db, uid)
    if balance < 1:
        raise HTTPException(
            429,
            "Недостаточно токенов. 1 мастеринг = 1 токен. Пополните баланс на странице тарифов или докупите токены.",
        )
    daily_cap = 30 if tier == "studio" else 10
    used_today = count_mastering_jobs_today(db, uid)
    if used_today >= daily_cap:
        raise HTTPException(
            429,
            f"Дневной лимит тарифа {tier}: не более {daily_cap} мастерингов в день. Сброс в полночь UTC.",
        )
    if not deduct_tokens(db, uid, 1):
        raise HTTPException(429, "Не удалось списать токен. Попробуйте снова.")


def _check_and_deduct_paid_tier_batch(user, db, n_files: int) -> None:
    """Pro/Studio при пакетной обработке: проверяет токены и дневной лимит, списывает n_files токенов."""
    if not user or not db or not DB_AVAILABLE or n_files <= 0:
        return
    tier = (user.get("tier") or "").lower()
    if tier not in ("pro", "studio"):
        return
    uid = _user_id_from_user(user)
    if not uid:
        return
    balance = get_user_tokens_balance(db, uid)
    if balance < n_files:
        raise HTTPException(
            429,
            f"Недостаточно токенов. Нужно {n_files}, доступно {balance}. 1 мастеринг = 1 токен. Пополните баланс.",
        )
    daily_cap = 30 if tier == "studio" else 10
    used_today = count_mastering_jobs_today(db, uid)
    if used_today + n_files > daily_cap:
        raise HTTPException(
            429,
            f"Дневной лимит тарифа {tier}: не более {daily_cap} мастерингов в день. Сегодня использовано {used_today}, запрошено {n_files}.",
        )
    if not deduct_tokens(db, uid, n_files):
        raise HTTPException(429, "Не удалось списать токены. Попробуйте снова.")


def _get_max_upload_mb(filename: str, user: Optional[dict]) -> int:
    """Эффективный лимит загрузки (МБ): min(лимит тарифа, лимит формата). DJ-сеты: WAV до 800 МБ, MP3 до 300 МБ."""
    tier = (user.get("tier") or "free").lower() if user else "free"
    return settings_store.get_max_upload_mb(filename, tier)


def _validate_upload(data: bytes, filename: str, max_mb: int) -> None:
    """Проверяет размер файла и магические байты."""
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {max_mb} МБ.")
    if not _check_audio_magic_bytes(data, filename):
        raise HTTPException(400, "Содержимое файла не соответствует формату. Ожидается WAV, MP3 или FLAC.")


def _validate_format(filename: str, out_format: str) -> None:
    if not _allowed_file(filename):
        raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
    if out_format.lower() not in ("wav", "mp3", "flac", "opus", "aac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3, flac, opus или aac.")
    if out_format.lower() in ("mp3", "opus", "aac") and not shutil.which("ffmpeg"):
        raise HTTPException(
            400,
            f"Экспорт в {out_format.upper()} требует ffmpeg. Установите: sudo apt-get install -y ffmpeg",
        )


def _normalize_bitrate(out_format: str, bitrate: Optional[int]) -> Optional[int]:
    """Возвращает битрейт для MP3 (128/192/256/320) или OPUS (128/192), иначе None."""
    fmt = out_format.lower()
    if fmt == "mp3" and bitrate is not None and bitrate in (128, 192, 256, 320):
        return bitrate
    if fmt == "opus" and bitrate is not None and bitrate in (128, 192):
        return bitrate
    return None


def _new_job(
    job_id: str,
    target_lufs: float,
    style_key: str,
    data: bytes,
    filename: str,
    out_format: str,
    notify_user_id: Optional[int] = None,
) -> dict:
    """Создаёт запись задачи в jobs_store и возвращает её."""
    job: dict = {
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
        "original_bytes": data,
        "original_filename": filename,
        "out_format": out_format.lower(),
        "notify_user_id": notify_user_id,
    }
    _jobs_store.all_jobs()[job_id] = job
    return job


def _maybe_notify_telegram_mastering_done(job_id: str) -> None:
    """Push в Telegram user bot при готовности мастеринга с сайта."""
    job = _jobs_store.all_jobs().get(job_id)
    if not job or job.get("status") != "done":
        return
    uid = job.get("notify_user_id")
    if not uid or not DB_AVAILABLE:
        return
    try:
        from ..bot.notify_user import send_user_bot_text_sync
        from ..database import SessionLocal, User

        db = SessionLocal()
        try:
            u = db.query(User).filter(User.id == int(uid)).first()
            tid = getattr(u, "telegram_id", None) if u else None
            if tid:
                fname = (job.get("filename") or "track")[:200]
                send_user_bot_text_sync(
                    int(tid),
                    f"✅ Мастеринг готов: <code>{fname}</code>\n"
                    f"Скачайте результат в приложении: {getattr(settings, 'public_base_url', '') or 'https://magicmaster.pro'}/app",
                )
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        pass


def _apply_output_branding(out_bytes: bytes, fmt: str) -> bytes:
    try:
        from ..metadata import embed_magic_master_branding

        return embed_magic_master_branding(out_bytes, fmt)
    except Exception:  # noqa: BLE001
        return out_bytes


def _after_mastering_success_hooks(job_id: str) -> None:
    _maybe_notify_telegram_mastering_done(job_id)
    if not DB_AVAILABLE or SessionLocal is None:
        return
    job = _jobs_store.all_jobs().get(job_id)
    uid = job.get("notify_user_id") if job else None
    if not uid:
        return
    try:
        db = SessionLocal()
        try:
            process_referral_on_invitee_mastering_done(db, int(uid))
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        pass


# ─── Фоновые функции мастеринга ───────────────────────────────────────────────

def _run_mastering_job(
    job_id: str,
    data: bytes,
    filename: str,
    target_lufs: float,
    out_format: str,
    style: str = "standard",
    user_id: Optional[int] = None,
) -> None:
    """Синхронный мастеринг (v1) в потоке; обновляет jobs[job_id] (progress, result или error)."""
    job = _jobs_store.all_jobs()[job_id]
    ctx = TraceContext.build(
        job_id,
        filename or "wav",
        "v1",
        style=style,
        user_id=user_id,
        target_lufs=target_lufs,
    )
    trace_job_start(ctx)
    try:
        job["progress"] = 2
        job["message"] = "Загрузка аудио…"
        audio, sr = load_audio_from_bytes(data, filename or "wav")

        job["progress"] = 4
        job["message"] = "Измерение исходного уровня…"
        job["before_lufs"] = measure_lufs(audio, sr)
        trace_stage(ctx, "input_loaded", audio, sr)

        def on_progress(pct: int, msg: str) -> None:
            job["progress"] = pct
            job["message"] = msg

        job["progress"] = 5
        job["message"] = "Мастеринг…"
        mastered = run_mastering_pipeline(
            audio,
            sr,
            target_lufs=target_lufs,
            style=style,
            progress_callback=on_progress,
            trace_ctx=ctx,
        )
        job["after_lufs"] = measure_lufs(mastered, sr)
        job["progress"] = 98
        job["message"] = "Экспорт файла…"
        channels = 1 if mastered.ndim == 1 else mastered.shape[1]
        out_bytes = export_audio(mastered, sr, channels, out_format.lower())
        out_bytes = _apply_output_branding(out_bytes, out_format.lower())
        out_ext = "m4a" if out_format.lower() == "aac" else out_format.lower()
        out_name = (filename or "master").rsplit(".", 1)[0] + f"_mastered.{out_ext}"
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Готово"
        job["result_bytes"] = out_bytes
        job["filename"] = out_name
        job["done_at"] = time.time()
        try:
            log_mastering_job_end(job_id, "done")
        except Exception:  # noqa: BLE001
            pass
        trace_job_done(
            ctx,
            before_lufs=job.get("before_lufs"),
            after_lufs=job.get("after_lufs"),
            out_format=out_format.lower(),
        )
        _after_mastering_success_hooks(job_id)
    except Exception as e:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = ""
        job["error"] = str(e)
        job["done_at"] = time.time()
        try:
            log_mastering_job_end(job_id, "error")
        except Exception:  # noqa: BLE001
            pass
        logger.exception(
            "mastering job failed job_id=%s path=v1 filename=%s error=%s",
            job_id,
            filename,
            str(e)[:500],
        )
        trace_job_error(ctx, e)
        try:
            from ..notifier import notify_mastering_error
            notify_mastering_error(filename, str(e)[:200])
        except Exception:  # noqa: BLE001
            pass


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
    bitrate: Optional[int] = None,
    pro_params: Optional[dict] = None,
    user_id: Optional[int] = None,
) -> None:
    """Мастеринг через MasteringChain (v2) + PRO-модули. Если chain_config is None — используется default_chain."""
    job = _jobs_store.all_jobs()[job_id]
    pro_params = pro_params or {}
    ctx = TraceContext.build(
        job_id,
        filename or "wav",
        "v2",
        style=style,
        user_id=user_id,
        target_lufs=target_lufs,
        pro_params=pro_params,
    )
    trace_job_start(ctx)
    try:
        job["progress"] = 2
        job["message"] = "Загрузка аудио…"
        audio, sr = load_audio_from_bytes(data, filename or "wav")

        if pro_params.get("apply_vocal_isolation") and getattr(settings, "enable_vocal_isolation", False):
            from ..services.vocal_isolation import isolate_vocal, is_demucs_available
            if not is_demucs_available():
                raise RuntimeError(
                    "Изоляция вокала включена, но Demucs не установлен. "
                    "Установите: pip install -r requirements-vocal-isolation.txt"
                )
            job["message"] = "Изоляция вокала (Demucs)…"
            try:
                vocal_bytes = isolate_vocal(data, filename or "audio.wav", getattr(settings, "demucs_model", "htdemucs"))
                audio, sr = load_audio_from_bytes(vocal_bytes, "vocals.wav")
            except RuntimeError as e:
                raise RuntimeError("Изоляция вокала не удалась: %s" % e) from e

        job["progress"] = 4
        job["message"] = "Измерение исходного уровня…"
        job["before_lufs"] = measure_lufs(audio, sr)

        if pro_params.get("rumble_enabled"):
            job["message"] = "Румбл-фильтр…"
            cutoff = float(pro_params.get("rumble_cutoff", 80.0))
            audio = apply_rumble_filter(audio, sr, cutoff_hz=cutoff)
            if _is_debug_mode():
                peak = float(np.max(np.abs(audio))) + 1e-12
                job["message"] = f"Румбл-фильтр… (peak {20 * np.log10(peak):.1f} dB)"

        denoise_strength = pro_params.get("denoise_strength", 0) or 0
        denoise_preset = (pro_params.get("denoise_preset") or "").strip().lower()
        if denoise_preset in DENOISE_PRESETS:
            strength, noise_pct = DENOISE_PRESETS[denoise_preset]
        elif denoise_strength > 0:
            strength = float(denoise_strength)
            noise_pct = float(pro_params.get("denoise_noise_percentile", 15.0))
        else:
            strength = 0.0
            noise_pct = 15.0
        if strength > 0.01:
            job["message"] = "Spectral Denoiser…"
            audio = apply_spectral_denoise(audio, sr, strength=strength, noise_percentile=noise_pct)
            if _is_debug_mode():
                peak = float(np.max(np.abs(audio))) + 1e-12
                job["message"] = f"Spectral Denoiser… (peak {20 * np.log10(peak):.1f} dB)"

        if pro_params.get("deesser_enabled"):
            job["message"] = "De-esser…"
            thr = pro_params.get("deesser_threshold", -6.0)
            freq_hi = float(pro_params.get("deesser_freq_hi", 9000.0))
            audio = apply_deesser(
                audio, sr,
                threshold_db=thr,
                freq_hi=freq_hi,
            )
            if _is_debug_mode():
                peak = float(np.max(np.abs(audio))) + 1e-12
                job["message"] = f"De-esser… (peak {20 * np.log10(peak):.1f} dB)"

        def on_progress(pct: int, msg: str) -> None:
            job["progress"] = pct
            job["message"] = msg

        job["progress"] = 5
        job["message"] = "Мастеринг (v2)…"
        if chain_config:
            chain = MasteringChain.from_config(chain_config)
        else:
            chain = MasteringChain.default_chain(target_lufs=target_lufs, style=style)
        trace_chain_modules(ctx, [getattr(m, "module_id", "?") for m in chain.modules])
        trace_stage(ctx, "v2_pre_chain", audio, sr)
        mastered = chain.process(
            audio,
            sr,
            target_lufs=target_lufs,
            style=style,
            progress_callback=on_progress,
            trace_ctx=ctx,
        )

        ta = pro_params.get("transient_attack")
        ts = pro_params.get("transient_sustain", 1.0)
        if ta is not None and (abs(float(ta) - 1.0) > 0.02 or abs(float(ts) - 1.0) > 0.02):
            job["message"] = "Transient Designer…"
            mastered = apply_transient_designer(
                mastered, sr,
                attack_gain=float(ta),
                sustain_gain=float(ts),
            )
            if _is_debug_mode():
                peak = float(np.max(np.abs(mastered))) + 1e-12
                job["message"] = f"Transient Designer… (peak {20 * np.log10(peak):.1f} dB)"
            trace_stage(ctx, "v2_transient_designer", mastered, sr)

        parallel_mix_val = pro_params.get("parallel_mix", 0)
        if parallel_mix_val is not None and float(parallel_mix_val) > 0:
            job["message"] = "Parallel Compression…"
            mastered = apply_parallel_compression(mastered, sr, mix=float(parallel_mix_val))
            if _is_debug_mode():
                peak = float(np.max(np.abs(mastered))) + 1e-12
                job["message"] = f"Parallel Compression… (peak {20 * np.log10(peak):.1f} dB)"
            trace_stage(ctx, "v2_parallel_compression", mastered, sr)

        if pro_params.get("dynamic_eq_enabled"):
            job["message"] = "Dynamic EQ…"
            mastered = apply_dynamic_eq(mastered, sr)
            if _is_debug_mode():
                peak = float(np.max(np.abs(mastered))) + 1e-12
                job["message"] = f"Dynamic EQ… (peak {20 * np.log10(peak):.1f} dB)"
            trace_stage(ctx, "v2_dynamic_eq", mastered, sr)

        mastered = apply_output_edge_fade_in(mastered, sr, fade_ms=6.0)
        trace_stage(ctx, "v2_output_fade_in", mastered, sr)
        validate_mastered_not_silent(mastered, trace_ctx=ctx, trace_sr=sr)

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
            bitrate=bitrate,
        )
        out_bytes = _apply_output_branding(out_bytes, out_format.lower())
        out_ext = "m4a" if out_format.lower() == "aac" else out_format.lower()
        out_name = (filename or "master").rsplit(".", 1)[0] + f"_mastered.{out_ext}"
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Готово"
        job["result_bytes"] = out_bytes
        job["filename"] = out_name
        job["done_at"] = time.time()
        try:
            log_mastering_job_end(job_id, "done")
        except Exception:  # noqa: BLE001
            pass
        trace_job_done(
            ctx,
            before_lufs=job.get("before_lufs"),
            after_lufs=job.get("after_lufs"),
            out_format=out_format.lower(),
        )
        _after_mastering_success_hooks(job_id)
    except Exception as e:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = ""
        job["error"] = str(e)
        job["done_at"] = time.time()
        try:
            log_mastering_job_end(job_id, "error")
        except Exception:  # noqa: BLE001
            pass
        logger.exception(
            "mastering v2 job failed job_id=%s path=v2 filename=%s error=%s",
            job_id,
            filename,
            str(e)[:500],
        )
        trace_job_error(ctx, e)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/api/master")
async def api_master(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_lufs: Optional[float] = Form(None),
    preset: Optional[str] = Form(None),
    out_format: str = Form("wav"),
    style: str = Form("standard"),
    user: Optional[dict] = Depends(_get_current_user_optional),
    db=Depends(get_db),
):
    """
    Запуск мастеринга: возвращает job_id. Статус — GET /api/master/status/{job_id}.
    Результат — GET /api/master/result/{job_id} после status=done.
    """
    _validate_format(file.filename or "", out_format)
    _check_mastering_rate_limit(user, request)
    _check_and_deduct_paid_tier(user, db)

    if target_lufs is None:
        target_lufs = settings_store.get_setting_float("default_target_lufs", -14.0)
    if preset and preset.lower() in PRESET_LUFS:
        target_lufs = PRESET_LUFS[preset.lower()]

    data = await file.read()
    max_mb = _get_max_upload_mb(file.filename or "", user)
    _validate_upload(data, file.filename or "", max_mb)
    try:
        load_audio_from_bytes(data, file.filename or "wav")
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}")

    if not user and not _is_debug_mode():
        _record_usage(_get_client_ip(request))

    _prune()
    job_id = str(uuid.uuid4())
    style_key = style.lower() if style.lower() in STYLE_CONFIGS else "standard"
    _new_job(
        job_id,
        target_lufs,
        style_key,
        data,
        file.filename or "audio.wav",
        out_format,
        notify_user_id=_user_id_from_user(user),
    )
    try:
        log_mastering_job_start(job_id, _user_id_from_user(user), style_key)
    except Exception:  # noqa: BLE001
        pass

    sem = _jobs_store.sem_priority if _is_priority_user(user) else _jobs_store.sem_normal

    async def run_job() -> None:
        async with sem:
            await asyncio.to_thread(
                _run_mastering_job,
                job_id,
                data,
                file.filename or "audio.wav",
                target_lufs,
                out_format.lower(),
                style_key,
                _user_id_from_user(user),
            )

    background_tasks.add_task(run_job)
    return {"job_id": job_id, "status": "running"}


@router.post("/api/v2/master")
async def api_master_v2(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    target_lufs: Optional[float] = Form(None),
    style: str = Form("standard"),
    out_format: str = Form("wav"),
    dither_type: Optional[str] = Form(None),
    auto_blank_sec: Optional[float] = Form(None),
    bitrate: Optional[str] = Form(None),
    denoise_strength: Optional[float] = Form(None),
    denoise_preset: Optional[str] = Form(None),
    denoise_noise_percentile: Optional[float] = Form(None),
    rumble_enabled: Optional[str] = Form(None),
    rumble_cutoff: Optional[float] = Form(None),
    deesser_enabled: Optional[str] = Form(None),
    deesser_threshold: Optional[float] = Form(None),
    deesser_freq_hi: Optional[float] = Form(None),
    transient_attack: Optional[float] = Form(None),
    transient_sustain: Optional[float] = Form(None),
    parallel_mix: Optional[float] = Form(None),
    dynamic_eq_enabled: Optional[str] = Form(None),
    apply_vocal_isolation: Optional[str] = Form(None),
    user: Optional[dict] = Depends(_get_current_user_optional),
    db=Depends(get_db),
):
    """
    Мастеринг v2: цепочка из JSON-конфига.
    Форма: file (обязательно), config (JSON-строка, опционально), target_lufs, style, out_format,
    dither_type (tpdf | ns_e | ns_itu), auto_blank_sec (сек), bitrate (для MP3: 128/192/256/320, для OPUS: 128/192).
    Ответ: job_id; статус и результат — те же GET /api/master/status/{job_id}, GET /api/master/result/{job_id}.
    """
    try:
        _validate_format(file.filename or "", out_format)
        try:
            bitrate_val = _normalize_bitrate(out_format, int(bitrate) if bitrate and str(bitrate).strip() else None)
        except (ValueError, TypeError):
            bitrate_val = None
        _check_mastering_rate_limit(user, request)
        _check_and_deduct_paid_tier(user, db)

        if target_lufs is None:
            target_lufs = settings_store.get_setting_float("default_target_lufs", -14.0)
        data = await file.read()
        max_mb = _get_max_upload_mb(file.filename or "", user)
        _validate_upload(data, file.filename or "", max_mb)
        try:
            load_audio_from_bytes(data, file.filename or "wav")
        except Exception as e:
            logger.error("v2/master: load_audio failed filename=%s error=%s", file.filename, str(e)[:200])
            raise HTTPException(400, f"Не удалось прочитать аудио: {e}") from e

        chain_config: Optional[dict[str, Any]] = None
        if config and config.strip():
            try:
                chain_config = json.loads(config)
            except json.JSONDecodeError as e:
                raise HTTPException(400, f"Неверный JSON в config: {e}") from e

        if not user and not _is_debug_mode():
            _record_usage(_get_client_ip(request))

        _prune()
        job_id = str(uuid.uuid4())
        style_key = style.lower() if style.lower() in STYLE_CONFIGS else "standard"
        _new_job(
            job_id,
            target_lufs,
            style_key,
            data,
            file.filename or "audio.wav",
            out_format,
            notify_user_id=_user_id_from_user(user),
        )
        try:
            log_mastering_job_start(job_id, _user_id_from_user(user), style_key)
        except Exception:  # noqa: BLE001
            pass

        pro_params: dict = {}
        if denoise_preset and denoise_preset.strip().lower() in (
            "vocal", "light", "medium", "aggressive", "tape_hiss", "room_tone"
        ):
            pro_params["denoise_preset"] = denoise_preset.strip().lower()
        elif denoise_strength is not None and denoise_strength > 0:
            pro_params["denoise_strength"] = float(denoise_strength)
        if denoise_noise_percentile is not None and 5 <= denoise_noise_percentile <= 40:
            pro_params["denoise_noise_percentile"] = float(denoise_noise_percentile)
        if rumble_enabled and rumble_enabled.lower() in ("true", "1", "yes"):
            pro_params["rumble_enabled"] = True
            cutoff = float(rumble_cutoff) if rumble_cutoff is not None else 80.0
            pro_params["rumble_cutoff"] = max(20.0, min(200.0, cutoff))
        if deesser_enabled and deesser_enabled.lower() in ("true", "1", "yes"):
            pro_params["deesser_enabled"] = True
            pro_params["deesser_threshold"] = float(deesser_threshold) if deesser_threshold is not None else -6.0
            f_hi = float(deesser_freq_hi) if deesser_freq_hi is not None else 9000.0
            pro_params["deesser_freq_hi"] = max(5000.0, min(12000.0, f_hi))
        if transient_attack is not None and transient_sustain is not None:
            pro_params["transient_attack"] = float(transient_attack)
            pro_params["transient_sustain"] = float(transient_sustain)
        if parallel_mix is not None and parallel_mix > 0:
            pro_params["parallel_mix"] = float(parallel_mix)
        if dynamic_eq_enabled and dynamic_eq_enabled.lower() in ("true", "1", "yes"):
            pro_params["dynamic_eq_enabled"] = True
        if (
            apply_vocal_isolation and apply_vocal_isolation.lower() in ("true", "1", "yes")
            and getattr(settings, "enable_vocal_isolation", False)
        ):
            pro_params["apply_vocal_isolation"] = True

        sem = _jobs_store.sem_priority if _is_priority_user(user) else _jobs_store.sem_normal

        async def run_job_v2() -> None:
            async with sem:
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
                    bitrate=bitrate_val,
                    pro_params=pro_params,
                    user_id=_user_id_from_user(user),
                )

        background_tasks.add_task(run_job_v2)
        return {"job_id": job_id, "status": "running", "version": "v2"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_master_v2: unhandled error")
        raise HTTPException(500, detail="Ошибка сервера при запуске мастеринга") from e


@router.post("/api/v2/batch")
async def api_v2_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    config: Optional[str] = Form(None),
    target_lufs: Optional[float] = Form(None),
    style: str = Form("standard"),
    out_format: str = Form("wav"),
    dither_type: Optional[str] = Form(None),
    auto_blank_sec: Optional[float] = Form(None),
    bitrate: Optional[str] = Form(None),
    denoise_strength: Optional[float] = Form(None),
    denoise_preset: Optional[str] = Form(None),
    denoise_noise_percentile: Optional[float] = Form(None),
    rumble_enabled: Optional[str] = Form(None),
    rumble_cutoff: Optional[float] = Form(None),
    deesser_enabled: Optional[str] = Form(None),
    deesser_threshold: Optional[float] = Form(None),
    deesser_freq_hi: Optional[float] = Form(None),
    transient_attack: Optional[float] = Form(None),
    transient_sustain: Optional[float] = Form(None),
    parallel_mix: Optional[float] = Form(None),
    dynamic_eq_enabled: Optional[str] = Form(None),
    apply_vocal_isolation: Optional[str] = Form(None),
    user: Optional[dict] = Depends(_get_current_user_optional),
    db=Depends(get_db),
):
    """
    Пакетный мастеринг: несколько файлов с одинаковыми параметрами.
    Возвращает список { job_id, filename }. Статус и результат — как у одиночного.
    Максимум файлов: 10. Free: 1 мастеринг в неделю; Pro/Studio: по токенам и дневному лимиту.
    """
    try:
        _require_feature_batch()
        if not files:
            raise HTTPException(400, "Отправьте хотя бы один файл.")
        if len(files) > _BATCH_MAX_FILES:
            raise HTTPException(400, f"Максимум {_BATCH_MAX_FILES} файлов за один запрос.")

        for f in files:
            if not _allowed_file(f.filename or ""):
                raise HTTPException(400, f"Формат не поддерживается: {f.filename}. Разрешены: WAV, MP3, FLAC.")
        if out_format.lower() not in ("wav", "mp3", "flac", "opus", "aac"):
            raise HTTPException(400, "Формат экспорта: wav, mp3, flac, opus или aac.")
        if out_format.lower() in ("mp3", "opus", "aac") and not shutil.which("ffmpeg"):
            raise HTTPException(400, "Экспорт в MP3/OPUS/AAC требует ffmpeg.")

        try:
            bitrate_val = _normalize_bitrate(out_format, int(bitrate) if bitrate and str(bitrate).strip() else None)
        except (ValueError, TypeError):
            bitrate_val = None

        is_pro = user or _is_debug_mode()
        if not is_pro:
            ip = _get_client_ip(request)
            limit_info = _check_rate_limit(ip)
            if limit_info["remaining"] < len(files):
                raise HTTPException(
                    429,
                    f"Недостаточно лимита. Осталось {limit_info['remaining']} мастеринг(ов) в неделю, файлов — {len(files)}. "
                    f"Сброс: {limit_info['reset_at']}.",
                )
        else:
            _check_and_deduct_paid_tier_batch(user, db, len(files))

        if target_lufs is None:
            target_lufs = settings_store.get_setting_float("default_target_lufs", -14.0)
        style_key = style.lower() if style.lower() in STYLE_CONFIGS else "standard"
        chain_config: Optional[dict[str, Any]] = None
        if config and config.strip():
            try:
                chain_config = json.loads(config)
            except json.JSONDecodeError as e:
                raise HTTPException(400, f"Неверный JSON в config: {e}") from e

        payloads: List[Tuple[bytes, str]] = []
        for f in files:
            data = await f.read()
            max_mb = _get_max_upload_mb(f.filename or "", user)
            if len(data) > max_mb * 1024 * 1024:
                raise HTTPException(400, f"Файл {f.filename} больше {max_mb} МБ.")
            if not _check_audio_magic_bytes(data, f.filename or ""):
                raise HTTPException(400, f"Содержимое файла {f.filename} не соответствует формату. Ожидается WAV, MP3 или FLAC.")
            try:
                load_audio_from_bytes(data, f.filename or "wav")
            except Exception as e:
                raise HTTPException(400, f"Не удалось прочитать аудио {f.filename}: {e}") from e
            payloads.append((data, f.filename or "audio.wav"))

        if not is_pro:
            for _ in range(len(payloads)):
                _record_usage(_get_client_ip(request))

        _prune()
        dt = dither_type or "tpdf"
        ab = float(auto_blank_sec or 0)
        # PRO-параметры (как в одиночном мастеринге)
        batch_pro_params: dict = {}
        if denoise_preset and denoise_preset.strip().lower() in (
            "vocal", "light", "medium", "aggressive", "tape_hiss", "room_tone"
        ):
            batch_pro_params["denoise_preset"] = denoise_preset.strip().lower()
        elif denoise_strength is not None and denoise_strength > 0:
            batch_pro_params["denoise_strength"] = float(denoise_strength)
        if denoise_noise_percentile is not None and 5 <= denoise_noise_percentile <= 40:
            batch_pro_params["denoise_noise_percentile"] = float(denoise_noise_percentile)
        if rumble_enabled and rumble_enabled.lower() in ("true", "1", "yes"):
            batch_pro_params["rumble_enabled"] = True
            cutoff = float(rumble_cutoff) if rumble_cutoff is not None else 80.0
            batch_pro_params["rumble_cutoff"] = max(20.0, min(200.0, cutoff))
        if deesser_enabled and deesser_enabled.lower() in ("true", "1", "yes"):
            batch_pro_params["deesser_enabled"] = True
            batch_pro_params["deesser_threshold"] = float(deesser_threshold) if deesser_threshold is not None else -6.0
            f_hi = float(deesser_freq_hi) if deesser_freq_hi is not None else 9000.0
            batch_pro_params["deesser_freq_hi"] = max(5000.0, min(12000.0, f_hi))
        if transient_attack is not None and transient_sustain is not None:
            batch_pro_params["transient_attack"] = float(transient_attack)
            batch_pro_params["transient_sustain"] = float(transient_sustain)
        if parallel_mix is not None and parallel_mix > 0:
            batch_pro_params["parallel_mix"] = float(parallel_mix)
        if dynamic_eq_enabled and dynamic_eq_enabled.lower() in ("true", "1", "yes"):
            batch_pro_params["dynamic_eq_enabled"] = True
        if (
            apply_vocal_isolation and apply_vocal_isolation.lower() in ("true", "1", "yes")
            and getattr(settings, "enable_vocal_isolation", False)
        ):
            batch_pro_params["apply_vocal_isolation"] = True

        jobs_created: List[dict] = []
        batch_sem = _jobs_store.sem_priority if _is_priority_user(user) else _jobs_store.sem_normal

        for data, filename in payloads:
            job_id = str(uuid.uuid4())
            _jobs_store.all_jobs()[job_id] = {
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
                "notify_user_id": _user_id_from_user(user),
            }
            try:
                log_mastering_job_start(job_id, _user_id_from_user(user), style_key)
            except Exception:  # noqa: BLE001
                pass
            jobs_created.append({"job_id": job_id, "filename": filename})

            uid = _user_id_from_user(user)

            async def run_one(jid: str = job_id, d: bytes = data, fname: str = filename) -> None:
                async with batch_sem:
                    await asyncio.to_thread(
                        _run_mastering_job_v2,
                        jid,
                        d,
                        fname,
                        target_lufs,
                        out_format.lower(),
                        style_key,
                        chain_config,
                        dither_type=dt,
                        auto_blank_sec=ab,
                        bitrate=bitrate_val,
                        pro_params=batch_pro_params,
                        user_id=uid,
                    )

            background_tasks.add_task(run_one)

        return {"version": "v2", "batch": True, "jobs": jobs_created}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_v2_batch: unhandled error")
        raise HTTPException(500, detail="Ошибка сервера при пакетном мастеринге") from e


@router.post("/api/v2/master/auto")
async def api_v2_master_auto(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    out_format: str = Form("wav"),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """
    Авто-мастеринг: анализ трека → AI подбор пресета/настроек → мастеринг.
    Принимает file, опционально out_format. Возвращает job_id.
    """
    try:
        if not file.filename or not _allowed_file(file.filename):
            raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
        if out_format.lower() not in ("wav", "mp3", "flac", "opus", "aac"):
            raise HTTPException(400, "Формат экспорта: wav, mp3, flac, opus или aac.")
        if out_format.lower() in ("mp3", "opus", "aac") and not shutil.which("ffmpeg"):
            raise HTTPException(400, "Экспорт в MP3/OPUS/AAC требует ffmpeg.")

        _check_mastering_rate_limit(user, request)

        tier = _get_tier_for_ai(user, request)
        ident = _get_ai_identifier(request, user)
        ai_limit_info = ai_module.check_ai_rate_limit(ident, tier)
        if not ai_limit_info["ok"]:
            raise HTTPException(
                429,
                f"Лимит AI-запросов исчерпан: {ai_limit_info['limit']}/день. Сброс: {ai_limit_info['reset_at']}.",
            )

        data = await file.read()
        max_mb = _get_max_upload_mb(file.filename or "", user)
        _validate_upload(data, file.filename or "", max_mb)
        fname = file.filename or "audio.wav"
        try:
            audio, sr = load_audio_from_bytes(data, fname)
        except Exception as e:
            raise HTTPException(400, f"Не удалось прочитать аудио: {e}") from e

        try:
            lufs = measure_lufs(audio, sr)
        except Exception:
            lufs = float("nan")
        peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        peak_dbfs = float(20 * np.log10(max(peak, 1e-12)))
        duration_sec = float(len(audio) / sr)
        channels = 1 if audio.ndim == 1 else int(audio.shape[1])
        correlation = None
        if channels == 2 and audio.ndim == 2:
            correlation = measure_stereo_correlation(audio)
        analysis = {
            "lufs": round(lufs, 2) if not np.isnan(lufs) else None,
            "peak_dbfs": round(peak_dbfs, 2),
            "duration_sec": round(duration_sec, 3),
            "sample_rate": sr,
            "channels": channels,
            "stereo_correlation": round(correlation, 4) if correlation is not None else None,
        }
        if audio.size >= 4096:
            try:
                analysis["spectrum_bars"] = compute_spectrum_bars(audio, sr)
            except Exception:
                pass

        rec = ai_module.recommend_preset(analysis)
        style_key = (rec.get("style") or "standard").lower()
        if style_key not in STYLE_CONFIGS:
            style_key = "standard"
        target_lufs = float(rec.get("target_lufs", -14))
        target_lufs = max(-24, min(-6, target_lufs))
        chain_config = rec.get("chain_config")

        if not user and not _is_debug_mode():
            _record_usage(_get_client_ip(request))
        ai_module.record_ai_usage(ident)

        _prune()
        job_id = str(uuid.uuid4())
        _jobs_store.all_jobs()[job_id] = {
            "status": "running",
            "progress": 0,
            "message": "Авто-мастеринг…",
            "created_at": time.time(),
            "result_bytes": None,
            "filename": None,
            "error": None,
            "before_lufs": None,
            "after_lufs": None,
            "target_lufs": target_lufs,
            "style": style_key,
            "notify_user_id": _user_id_from_user(user),
        }
        try:
            log_mastering_job_start(job_id, _user_id_from_user(user), style_key)
        except Exception:  # noqa: BLE001
            pass

        sem = _jobs_store.sem_priority if _is_priority_user(user) else _jobs_store.sem_normal

        async def run_auto_job() -> None:
            async with sem:
                await asyncio.to_thread(
                    _run_mastering_job_v2,
                    job_id,
                    data,
                    fname,
                    target_lufs,
                    out_format.lower(),
                    style_key,
                    chain_config,
                    pro_params={},
                    user_id=_user_id_from_user(user),
                )

        background_tasks.add_task(run_auto_job)
        return {
            "job_id": job_id,
            "status": "running",
            "version": "v2",
            "auto": True,
            "recommendation": {
                "style": style_key,
                "target_lufs": target_lufs,
                "reason": rec.get("reason"),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_v2_master_auto: unhandled error")
        raise HTTPException(500, detail="Ошибка сервера при авто-мастеринге") from e


@router.get("/api/v2/chain/default")
def api_v2_chain_default(style: str = "standard", target_lufs: float = -14.0):
    """
    Конфиг цепочки по умолчанию: полный список модулей с id, label, enabled и параметрами.
    Ответ можно отправить в POST /api/v2/master (поле config).
    """
    config = MasteringChain.default_config(target_lufs=target_lufs, style=style)
    modules = []
    for m in config["modules"]:
        m = dict(m)
        mid = m.get("id")
        if mid:
            m["label"] = CHAIN_MODULE_LABELS.get(mid, mid)
        modules.append(m)
    return {
        "version": "v2",
        "target_lufs": target_lufs,
        "style": style,
        "modules": modules,
        "dither_type": config.get("dither_type", "tpdf"),
        "auto_blank_sec": config.get("auto_blank_sec", 0),
    }


@router.post("/api/v2/analyze")
async def api_v2_analyze(
    file: UploadFile = File(...),
    extended: bool = Form(False),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """
    Загрузить файл и вернуть анализ: LUFS, peak dBFS, длительность, sample rate, stereo_correlation.
    extended=true: дополнительно spectrum_bars, lufs_timeline, vectorscope_points.
    """
    try:
        if not _allowed_file(file.filename or ""):
            raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
        fname = file.filename or ""
        if fname.lower().endswith(".mp3") and not shutil.which("ffmpeg"):
            raise HTTPException(400, "Чтение MP3 требует ffmpeg. Установите: sudo apt-get install -y ffmpeg")
        data = await file.read()
        max_mb = _get_max_upload_mb(fname, user)
        _validate_upload(data, fname, max_mb)
        try:
            audio, sr = load_audio_from_bytes(data, fname)
        except Exception as e:
            raise HTTPException(400, f"Не удалось прочитать аудио: {e}") from e

        try:
            lufs = measure_lufs(audio, sr)
        except Exception:
            lufs = float("nan")
        peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        peak_dbfs = float(20 * np.log10(max(peak, 1e-12)))
        duration_sec = float(len(audio) / sr)
        channels = 1 if audio.ndim == 1 else int(audio.shape[1])
        correlation = None
        if channels == 2 and audio.ndim == 2:
            correlation = measure_stereo_correlation(audio)

        out: dict = {
            "version": "v2",
            "lufs": round(lufs, 2) if not np.isnan(lufs) else None,
            "peak_dbfs": round(peak_dbfs, 2),
            "duration_sec": round(duration_sec, 3),
            "sample_rate": sr,
            "channels": channels,
        }
        if correlation is not None:
            out["stereo_correlation"] = round(correlation, 4)

        if not np.isnan(lufs):
            _streaming_platforms = {
                "Spotify":              -14.0,
                "YouTube":              -14.0,
                "Apple Music":          -16.0,
                "Tidal":                -14.0,
                "Amazon Music":         -14.0,
                "Broadcast (EBU R128)": -23.0,
            }
            streaming_preview = {}
            for platform, target in _streaming_platforms.items():
                penalty = round(max(0.0, lufs - target), 2)
                gain_applied = round(min(0.0, target - lufs), 2)
                if penalty > 6.0:
                    status = "loud"
                elif penalty > 1.0:
                    status = "ok"
                else:
                    status = "optimal"
                streaming_preview[platform] = {
                    "target_lufs": target,
                    "penalty_db": penalty,
                    "gain_applied_db": gain_applied,
                    "status": status,
                }
            out["streaming_preview"] = streaming_preview

        if extended:
            if audio.size >= 4096:
                try:
                    out["spectrum_bars"] = compute_spectrum_bars(audio, sr)
                except Exception:
                    pass
                if channels == 2 and audio.ndim == 2:
                    try:
                        mid = ((audio[:, 0] + audio[:, 1]) * 0.5).astype(np.float32)
                        side = ((audio[:, 0] - audio[:, 1]) * 0.5).astype(np.float32)
                        out["spectrum_bars_mid"] = compute_spectrum_bars(mid, sr)
                        out["spectrum_bars_side"] = compute_spectrum_bars(side, sr)
                    except Exception:
                        pass
            try:
                lufs_timeline, timeline_step_sec = compute_lufs_timeline(audio, sr)
                out["lufs_timeline"] = [_json_safe_float(x) for x in (lufs_timeline or [])]
                out["timeline_step_sec"] = _json_safe_float(timeline_step_sec)
            except Exception:
                pass
            if channels == 2 and audio.ndim == 2:
                try:
                    out["vectorscope_points"] = compute_vectorscope_points(audio, max_points=1000)
                except Exception:
                    pass
        return out
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_v2_analyze: unhandled error")
        raise HTTPException(500, detail="Ошибка сервера при анализе файла") from e


@router.post("/api/v2/reference-match")
async def api_v2_reference_match(
    file: UploadFile = File(...),
    reference: UploadFile = File(...),
    strength: float = Form(0.8),
    out_format: str = Form("wav"),
    bitrate: Optional[str] = Form(None),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """
    Reference Track Mastering: подгоняет спектральный баланс треку к эталону.
    Принимает два файла (file — основной трек, reference — эталон) и параметр strength (0.0–1.0).
    """
    if not _allowed_file(file.filename or ""):
        raise HTTPException(400, "Формат основного файла не поддерживается.")
    if not _allowed_file(reference.filename or ""):
        raise HTTPException(400, "Формат эталонного файла не поддерживается.")
    if out_format.lower() not in ("wav", "mp3", "flac", "opus", "aac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3, flac, opus или aac.")
    if out_format.lower() in ("mp3", "opus", "aac") and not shutil.which("ffmpeg"):
        raise HTTPException(400, "Экспорт в MP3/OPUS/AAC требует ffmpeg.")
    strength = float(np.clip(strength, 0.0, 1.0))
    data_src = await file.read()
    data_ref = await reference.read()
    max_mb_src = _get_max_upload_mb(file.filename or "", user)
    max_mb_ref = _get_max_upload_mb(reference.filename or "", user)
    if len(data_src) > max_mb_src * 1024 * 1024:
        raise HTTPException(400, f"Основной файл больше {max_mb_src} МБ.")
    if len(data_ref) > max_mb_ref * 1024 * 1024:
        raise HTTPException(400, f"Эталонный файл больше {max_mb_ref} МБ.")
    if not _check_audio_magic_bytes(data_src, file.filename or ""):
        raise HTTPException(400, "Содержимое основного файла не соответствует формату.")
    if not _check_audio_magic_bytes(data_ref, reference.filename or ""):
        raise HTTPException(400, "Содержимое эталонного файла не соответствует формату.")
    try:
        audio_src, sr_src = load_audio_from_bytes(data_src, file.filename or "wav")
        audio_ref, sr_ref = load_audio_from_bytes(data_ref, reference.filename or "wav")
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}")
    try:
        result = apply_reference_match(audio_src, sr_src, audio_ref, sr_ref, strength=strength)
    except Exception as e:
        logger.exception("reference_match: processing error")
        raise HTTPException(500, "Ошибка обработки") from e
    channels = 1 if result.ndim == 1 else int(result.shape[1])
    try:
        bitrate_val = _normalize_bitrate(out_format, int(bitrate) if bitrate and str(bitrate).strip() else None)
    except (ValueError, TypeError):
        bitrate_val = None
    out_bytes = export_audio(result, sr_src, channels, out_format.lower(), bitrate=bitrate_val)
    out_ext = "m4a" if out_format.lower() == "aac" else out_format.lower()
    out_name = (file.filename or "track").rsplit(".", 1)[0] + f"_ref_matched.{out_ext}"
    safe_name = _safe_content_disposition_filename(out_name, "ref_matched." + out_ext)
    media = "audio/mp4" if out_format.lower() == "aac" else f"audio/{out_format}"
    return Response(
        content=out_bytes,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


_UPSCALE_ALLOWED_SR = (48000, 96000, 192000)


@router.post("/api/v2/upscale")
async def api_v2_upscale(
    file: UploadFile = File(...),
    target_sr: int = Form(96000),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """
    Upscale: ресемплинг аудио в более высокий sample rate (48k / 96k / 192k).
    Принимает file (WAV/FLAC/MP3), target_sr. Возвращает WAV 16-bit.
    """
    try:
        if not _allowed_file(file.filename or ""):
            raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
        if target_sr not in _UPSCALE_ALLOWED_SR:
            raise HTTPException(400, f"target_sr допускает только: {_UPSCALE_ALLOWED_SR}.")
        fname = file.filename or "audio.wav"
        data = await file.read()
        max_mb = _get_max_upload_mb(fname, user)
        _validate_upload(data, fname, max_mb)
        audio, sr = load_audio_from_bytes(data, fname)
        if target_sr <= sr:
            raise HTTPException(400, f"Upscale: target_sr ({target_sr}) должен быть больше текущего sample rate ({sr}).")
        up = resample_audio(audio, sr, target_sr)
        channels = 1 if up.ndim == 1 else int(up.shape[1])
        out_bytes = export_audio(up, target_sr, channels, "wav", dither_type="tpdf")
        base = (fname or "audio").rsplit(".", 1)[0]
        out_name = f"{base}_upscale_{target_sr // 1000}k.wav"
        safe_name = _safe_content_disposition_filename(out_name, "upscale.wav")
        return Response(
            content=out_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_v2_upscale: unhandled error")
        raise HTTPException(500, detail="Ошибка при upscale") from e


@router.post("/api/v2/isolate-vocal")
async def api_v2_isolate_vocal(
    file: UploadFile = File(...),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """
    Изоляция вокала (задача 9.2). Работает только при ENABLE_VOCAL_ISOLATION=1 и установленном demucs.
    Принимает аудиофайл (WAV/MP3/FLAC), возвращает WAV с дорожкой вокала.
    """
    if not settings.enable_vocal_isolation:
        raise HTTPException(
            503,
            "Изоляция вокала отключена. Включите MAGIC_MASTER_ENABLE_VOCAL_ISOLATION=1 и установите demucs (pip install -r requirements-vocal-isolation.txt).",
        )
    from ..services.vocal_isolation import isolate_vocal, is_demucs_available
    if not is_demucs_available():
        raise HTTPException(
            503,
            "Demucs не установлен. Установите: pip install -r requirements-vocal-isolation.txt",
        )
    if not _allowed_file(file.filename or ""):
        raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
    fname = file.filename or "audio.wav"
    data = await file.read()
    max_mb = _get_max_upload_mb(fname, user)
    _validate_upload(data, fname, max_mb)
    try:
        vocal_bytes = await asyncio.to_thread(
            isolate_vocal, data, fname, settings.demucs_model
        )
    except RuntimeError as e:
        logger.warning("isolate_vocal failed: %s", e)
        raise HTTPException(500, detail="Ошибка при изоляции вокала") from e
    base = (fname or "audio").rsplit(".", 1)[0]
    out_name = f"{base}_vocals.wav"
    safe_name = _safe_content_disposition_filename(out_name, "vocals.wav")
    return Response(
        content=vocal_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


@router.get("/api/master/status/{job_id}")
async def api_master_status(job_id: str):
    """Статус задачи мастеринга: progress 0–100, message."""
    _prune()
    jobs = _jobs_store.all_jobs()
    if job_id not in jobs:
        raise HTTPException(404, "Задача не найдена")
    job = jobs[job_id]
    return {
        "status": job["status"],
        "progress": int(job.get("progress", 0)),
        "message": job.get("message") or "",
        "error": job.get("error"),
        "before_lufs": _json_safe_float(job.get("before_lufs")),
        "after_lufs": _json_safe_float(job.get("after_lufs")),
        "target_lufs": _json_safe_float(job.get("target_lufs")),
        "style": job.get("style", "standard"),
    }


@router.get("/api/master/progress/{job_id}")
async def api_master_progress_sse(job_id: str):
    """SSE-стрим прогресса мастеринга. P37.

    Клиент: new EventSource('/api/master/progress/<job_id>')
    Закрывается сам после status=done|error.
    Формат события: data: {"status":"running","progress":42,"message":"..."}
    """
    import json as _json

    async def event_stream():
        poll_interval = 0.4
        max_wait = 600.0
        elapsed = 0.0
        last_progress = -1

        while elapsed < max_wait:
            jobs = _jobs_store.all_jobs()
            job = jobs.get(job_id)
            if job is None:
                payload = _json.dumps({"status": "error", "progress": 0, "message": "Задача не найдена"})
                yield f"data: {payload}\n\n"
                return

            progress = int(job.get("progress", 0))
            status = job.get("status", "running")
            message = job.get("message") or ""

            if progress != last_progress or status in ("done", "error"):
                payload = _json.dumps({
                    "status": status,
                    "progress": progress,
                    "message": message,
                    "error": job.get("error"),
                    "before_lufs": _json_safe_float(job.get("before_lufs")),
                    "after_lufs": _json_safe_float(job.get("after_lufs")),
                    "target_lufs": _json_safe_float(job.get("target_lufs")),
                    "style": job.get("style", "standard"),
                })
                yield f"data: {payload}\n\n"
                last_progress = progress

                if status in ("done", "error"):
                    return

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        yield f"data: {_json.dumps({'status':'error','progress':0,'message':'Тайм-аут ожидания'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/api/master/share/{job_id}")
async def api_master_share(job_id: str):
    """PNG-карточка для шеринга (пока задача в памяти, до скачивания результата)."""
    jobs = _jobs_store.all_jobs()
    if job_id not in jobs:
        raise HTTPException(404, "Задача не найдена")
    job = jobs[job_id]
    if job.get("status") != "done" or not job.get("result_bytes"):
        raise HTTPException(400, "Результат ещё не готов")
    from ..services.share_card import render_mastering_share_png

    png = render_mastering_share_png(job)
    if not png:
        raise HTTPException(503, "Карточка недоступна (установите Pillow)")
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "no-store", "Content-Disposition": 'inline; filename="magic-master-share.png"'},
    )


@router.get("/api/master/result/{job_id}")
async def api_master_result(job_id: str):
    """Скачать результат мастеринга (после status=done)."""
    jobs = _jobs_store.all_jobs()
    if job_id not in jobs:
        raise HTTPException(404, "Задача не найдена")
    job = jobs[job_id]
    if job["status"] != "done" or not job.get("result_bytes"):
        raise HTTPException(400, "Результат ещё не готов или задача с ошибкой")
    out_bytes = job["result_bytes"]
    filename = job["filename"] or "mastered.wav"
    del jobs[job_id]
    safe_name = _safe_content_disposition_filename(filename)
    return Response(
        content=out_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


@router.get("/api/master/preview/{job_id}")
async def api_master_preview(job_id: str, src: str = "mastered"):
    """
    Стриминг аудио для A/B-сравнения в браузере. P45.
    src=original — оригинал до мастеринга.
    src=mastered  — обработанный вариант (только после status=done).
    """
    jobs = _jobs_store.all_jobs()
    if job_id not in jobs:
        raise HTTPException(404, "Задача не найдена")
    job = jobs[job_id]

    if src == "original":
        audio_bytes = job.get("original_bytes")
        orig_name = job.get("original_filename", "audio.wav")
        ext = Path(orig_name).suffix.lower().lstrip(".")
    else:
        if job.get("status") != "done" or not job.get("result_bytes"):
            raise HTTPException(400, "Обработанный вариант ещё не готов")
        audio_bytes = job.get("result_bytes")
        orig_name = job.get("filename", "mastered.wav")
        ext = job.get("out_format", "wav")

    if not audio_bytes:
        raise HTTPException(404, "Аудио недоступно")

    _mime_map = {
        "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
        "opus": "audio/ogg", "aac": "audio/mp4", "m4a": "audio/mp4",
    }
    media_type = _mime_map.get(ext, "audio/wav")

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(audio_bytes)),
            "Cache-Control": "no-cache",
        },
    )
