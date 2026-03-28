"""Анализ аудио."""
import io

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from ...database import DB_AVAILABLE, SessionLocal
from ..helpers import get_linked_user, public_url
from ..services import TELEGRAM_MAX_DOWNLOAD_MB, analyze_audio_sync, validate_telegram_audio
from ..states import AnalyzeStates
from ..texts import lang_for_user, txt

router = Router(name="analyze")


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
    name = (message.document and message.document.file_name) or (
        message.audio and message.audio.file_name
    ) or "voice.ogg"
    from ..services import convert_voice_to_wav_if_needed

    return convert_voice_to_wav_if_needed(data, name or "x.wav")


@router.message(Command("analyze"))
@router.message(F.text.in_(["📊 Analyze", "📊 Анализ"]))
async def cmd_analyze(message: Message, state: FSMContext) -> None:
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        lang = lang_for_user(u) if u else "ru"
        if not u:
            await message.answer(txt(lang, "not_linked"))
            return
        await state.set_state(AnalyzeStates.waiting_file)
        await message.answer(txt(lang, "send_audio"))
    finally:
        db.close()


@router.message(AnalyzeStates.waiting_file, F.document | F.audio | F.voice)
async def analyze_got_file(message: Message, state: FSMContext, bot) -> None:
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
            await state.clear()
            return
        try:
            an = analyze_audio_sync(data, fname)
        except Exception:  # noqa: BLE001
            await message.answer(txt(lang, "error"))
            await state.clear()
            return
        await message.answer(
            txt(
                lang,
                "analyze_result",
                lufs=an.get("lufs"),
                peak=an.get("peak_dbfs"),
                dur=an.get("duration_sec"),
                ch=an.get("channels"),
                sr=an.get("sample_rate"),
            )
        )
        await state.clear()
    finally:
        db.close()
