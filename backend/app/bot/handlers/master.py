"""Мастеринг через бота."""
import asyncio
import io
import logging
import time

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from ...config import settings
from ...database import (
    DB_AVAILABLE,
    SessionLocal,
    create_mastering_record,
    log_mastering_job_end,
    log_mastering_job_start,
)
from ... import jobs_store as _jobs_store
from ...pipeline import STYLE_CONFIGS
from ..helpers import get_linked_user, precheck_mastering, public_url, refund_token_if_pro
from ..keyboards import presets_inline
from ..services import (
    TELEGRAM_MAX_DOWNLOAD_MB,
    convert_voice_to_wav_if_needed,
    run_mastering_sync,
    validate_telegram_audio,
)
from ..states import MasterStates
from ..texts import lang_for_user, txt

logger = logging.getLogger(__name__)

router = Router(name="master")


def _filename_from_message(message: Message) -> str:
    if message.document and message.document.file_name:
        return message.document.file_name
    if message.audio and message.audio.file_name:
        return message.audio.file_name
    if message.voice:
        return "voice.ogg"
    return "audio.wav"


async def _download_bytes(message: Message, bot) -> tuple[bytes, str]:
    if message.document:
        f = message.document
    elif message.audio:
        f = message.audio
    elif message.voice:
        f = message.voice
    else:
        return b"", ""
    buf = io.BytesIO()
    await bot.download(file=f, destination=buf)
    data = buf.getvalue()
    name = _filename_from_message(message)
    data, name = convert_voice_to_wav_if_needed(data, name)
    return data, name


def _sem_for_user(u) -> object:
    if getattr(settings, "debug_mode", False):
        return _jobs_store.sem_priority
    return _jobs_store.sem_priority if (u.tier or "").lower() in ("pro", "studio") else _jobs_store.sem_normal


@router.message(Command("master"))
@router.message(F.text.in_(["🎛 Master", "🎛 Мастеринг"]))
async def cmd_master(message: Message, state: FSMContext) -> None:
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        lang = lang_for_user(u) if u else "ru"
        if not u:
            await message.answer(txt(lang, "not_linked"))
            return
        if getattr(u, "is_blocked", False):
            await message.answer(txt(lang, "blocked"))
            return
        await state.set_state(MasterStates.waiting_file)
        await message.answer(txt(lang, "send_audio"))
    finally:
        db.close()


@router.message(MasterStates.waiting_file, F.document | F.audio | F.voice)
async def master_got_file(message: Message, state: FSMContext, bot) -> None:
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        lang = lang_for_user(u) if u else "ru"
        if not u:
            await state.clear()
            await message.answer(txt(lang, "not_linked"))
            return
        data, fname = await _download_bytes(message, bot)
        err = validate_telegram_audio(data, fname, TELEGRAM_MAX_DOWNLOAD_MB)
        if err:
            if err.startswith("file_too_large"):
                await message.answer(
                    txt(lang, "file_too_large", mb=TELEGRAM_MAX_DOWNLOAD_MB, url=public_url())
                )
            else:
                await message.answer(txt(lang, "bad_format"))
            return
        await state.update_data(audio_bytes=data, filename=fname)
        await state.set_state(MasterStates.choosing_preset)
        await message.answer(txt(lang, "choose_preset"), reply_markup=presets_inline("p"))
    finally:
        db.close()


@router.callback_query(F.data.startswith("p:m:"), MasterStates.choosing_preset)
async def master_preset_cb(query: CallbackQuery, state: FSMContext, bot) -> None:
    style = (query.data or "").split(":")[-1]
    if style not in STYLE_CONFIGS:
        style = "standard"
    st_data = await state.get_data()
    audio_bytes = st_data.get("audio_bytes")
    filename = st_data.get("filename") or "track.wav"
    if not audio_bytes:
        await query.answer("No file", show_alert=True)
        await state.clear()
        return
    if not DB_AVAILABLE or SessionLocal is None:
        await query.answer()
        return
    db = SessionLocal()
    try:
        u = get_linked_user(db, query.from_user.id)
        lang = lang_for_user(u) if u else "ru"
        if not u:
            await query.answer()
            await query.message.answer(txt(lang, "not_linked"))
            await state.clear()
            return
        errc = precheck_mastering(db, u)
        if errc:
            await query.answer()
            await query.message.answer(txt(lang, errc))
            await state.clear()
            return
        user_id = int(u.id)
        await query.answer()
        await query.message.answer(txt(lang, "processing"))
    finally:
        db.close()

    job_id = f"tg_{query.from_user.id}_{int(time.time() * 1000)}"
    try:
        log_mastering_job_start(job_id, user_id, style)
    except Exception:  # noqa: BLE001
        pass

    db = SessionLocal()
    try:
        u = get_linked_user(db, query.from_user.id)
        sem = _sem_for_user(u) if u else _jobs_store.sem_normal
    finally:
        db.close()

    async def run() -> None:
        async with sem:
            return await asyncio.to_thread(
                run_mastering_sync,
                audio_bytes,
                filename,
                style=style,
                out_format="wav",
            )

    try:
        out_b, out_name, before, after = await run()
    except Exception:  # noqa: BLE001
        logger.exception("telegram master failed")
        db = SessionLocal()
        try:
            u2 = get_linked_user(db, query.from_user.id)
            if u2:
                refund_token_if_pro(db, u2)
        finally:
            db.close()
        try:
            log_mastering_job_end(job_id, "error")
        except Exception:  # noqa: BLE001
            pass
        await query.message.answer(txt(lang, "error"))
        await state.clear()
        return

    try:
        log_mastering_job_end(job_id, "done")
    except Exception:  # noqa: BLE001
        pass

    tgt = float(STYLE_CONFIGS.get(style, STYLE_CONFIGS["standard"]).get("lufs", -14.0))
    db = SessionLocal()
    try:
        u = get_linked_user(db, query.from_user.id)
        if u:
            create_mastering_record(
                db,
                u.id,
                out_name,
                style,
                "wav",
                before,
                after,
                tgt,
                None,
            )
    finally:
        db.close()

    await query.message.answer(
        txt(lang, "done", before=before or 0.0, after=after or 0.0),
    )
    await bot.send_document(
        query.message.chat.id,
        document=BufferedInputFile(out_b, filename=out_name[:200]),
        caption=out_name[:1024],
    )
    await state.clear()
