"""Привязка аккаунта, баланс, история, pricing, ref."""
import re
import secrets
from typing import Optional

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message

from ...database import (
    DB_AVAILABLE,
    SessionLocal,
    User,
    create_telegram_link_code,
    get_user_by_email,
    get_user_history,
    get_user_tokens_balance,
    verify_telegram_link_code,
)
from ...mailer import send_email
from ..helpers import get_linked_user, public_url
from ..keyboards import main_menu_reply
from ..states import LinkStates
from ..texts import lang_for_user, txt
from aiogram.fsm.context import FSMContext

router = Router(name="account")


def _detect_lang(message: Message) -> str:
    lc = (message.from_user.language_code or "ru").lower()[:2]
    return "en" if lc == "en" else "ru"


@router.message(Command("link"))
async def cmd_link(message: Message, state: FSMContext) -> None:
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or "@" not in parts[1]:
        await message.answer(txt(_detect_lang(message), "link_bad_email"))
        return
    email = parts[1].strip().lower()
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        u = get_user_by_email(db, email)
        if not u:
            await message.answer(txt(_detect_lang(message), "link_no_user"))
            return
        code = f"{secrets.randbelow(1000000):06d}"
        create_telegram_link_code(db, email, message.from_user.id, code)
        linked = get_linked_user(db, message.from_user.id)
        lang = lang_for_user(linked) if linked else _detect_lang(message)
        subj = "Magic Master — код привязки Telegram" if lang != "en" else "Magic Master — Telegram link code"
        body = f"<p>Код: <b>{code}</b></p><p>Введите в боте: <code>/code {code}</code></p>"
        if lang == "en":
            body = f"<p>Code: <b>{code}</b></p><p>In bot: <code>/code {code}</code></p>"
        send_email(email, subj, body, text=f"Code: {code}")
        await message.answer(txt(lang, "link_sent", email=email))
        await state.set_state(LinkStates.waiting_code)
    finally:
        db.close()


@router.message(Command("code"))
async def cmd_code(message: Message, state: FSMContext) -> None:
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not re.match(r"^\d{4,8}$", parts[1].strip()):
        await message.answer("Формат: /code 123456")
        return
    if not DB_AVAILABLE or SessionLocal is None:
        return
    tid_for_menu: Optional[int] = None
    db = SessionLocal()
    try:
        user = verify_telegram_link_code(db, message.from_user.id, parts[1].strip())
        if not user:
            await message.answer(txt(_detect_lang(message), "code_bad"))
            return
        lang = _detect_lang(message)
        u2 = db.query(User).filter(User.id == user.id).first()
        if u2:
            u2.telegram_lang = lang
            db.commit()
        await state.clear()
        await message.answer(
            txt(lang, "code_ok", email=user.email),
            reply_markup=main_menu_reply(lang),
        )
        tid_for_menu = int(message.from_user.id)
    finally:
        db.close()
    if tid_for_menu is not None:
        try:
            from ..lifecycle import refresh_menu_for_telegram_chat

            await refresh_menu_for_telegram_chat(tid_for_menu)
        except Exception:
            pass


@router.message(LinkStates.waiting_code, F.text)
async def state_waiting_code_digits(message: Message, state: FSMContext) -> None:
    t = (message.text or "").strip()
    if not re.match(r"^\d{4,8}$", t):
        return
    message.text = f"/code {t}"
    await cmd_code(message, state)


@router.message(Command("unlink"))
async def cmd_unlink(message: Message) -> None:
    if not DB_AVAILABLE or SessionLocal is None:
        return
    tid_for_menu: Optional[int] = None
    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        lang = lang_for_user(u) if u else "ru"
        if not u:
            await message.answer(txt(lang, "unlink_none"))
            return
        tid_for_menu = int(message.from_user.id)
        u.telegram_id = None
        db.commit()
        await message.answer(txt(lang, "unlink_ok"))
    finally:
        db.close()
    if tid_for_menu is not None:
        try:
            from ..lifecycle import refresh_menu_for_telegram_chat

            await refresh_menu_for_telegram_chat(tid_for_menu)
        except Exception:
            pass


@router.message(Command("balance"))
@router.message(F.text.in_(("💰 Balance", "💰 Баланс")))
async def cmd_balance(message: Message) -> None:
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        lang = lang_for_user(u) if u else "ru"
        if not u:
            await message.answer(txt(lang, "not_linked"))
            return
        bal = get_user_tokens_balance(db, u.id)
        await message.answer(txt(lang, "balance", tier=u.tier, tokens=bal))
    finally:
        db.close()


@router.message(Command("history"))
async def cmd_history(message: Message) -> None:
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        lang = lang_for_user(u) if u else "ru"
        if not u:
            await message.answer(txt(lang, "not_linked"))
            return
        rows = get_user_history(db, u.id, limit=10)
        if not rows:
            await message.answer(txt(lang, "history_empty"))
            return
        lines = []
        for r in rows:
            lu = r.after_lufs if r.after_lufs is not None else "—"
            lines.append(
                txt(lang, "history_line", name=(r.filename or "")[:40], style=r.style, lufs=lu)
            )
        await message.answer("\n".join(lines))
    finally:
        db.close()


@router.message(Command("pricing"))
async def cmd_pricing(message: Message) -> None:
    db = SessionLocal() if DB_AVAILABLE else None
    try:
        u = get_linked_user(db, message.from_user.id) if db else None
        lang = lang_for_user(u) if u else "ru"
    finally:
        if db:
            db.close()
    await message.answer(txt(lang, "pricing", url=public_url()))


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    from ... import jobs_store as js

    jobs = js.all_jobs()
    running = sum(1 for j in jobs.values() if j.get("status") == "running")
    db = SessionLocal() if DB_AVAILABLE else None
    try:
        u = get_linked_user(db, message.from_user.id) if db else None
        lang = lang_for_user(u) if u else "ru"
    finally:
        if db:
            db.close()
    await message.answer(txt(lang, "status_ok", st="ok", jobs=running))


@router.message(Command("ref"))
async def cmd_ref(message: Message) -> None:
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        lang = lang_for_user(u) if u else "ru"
        if not u:
            await message.answer(txt(lang, "not_linked"))
            return
        await message.answer(txt(lang, "ref", url=public_url(), uid=u.id))
    finally:
        db.close()
