#!/usr/bin/env python3
"""
Заполнение БД тестовыми данными для админ-панели.
Не менее 44 записей по каждому типу данных, с ретроспективными датами
для проверки всех форм и отчётов (фильтры по датам).

Запуск из каталога backend:
  python scripts/seed_admin_data.py
Или из корня проекта:
  cd backend && python scripts/seed_admin_data.py
"""
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Добавляем backend в path
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

# Минимум 44 записи в каждой ключевой таблице
NUM_USERS = 48
NUM_TRANSACTIONS = 50
NUM_NEWS = 48
NUM_CAMPAIGNS = 48
NUM_AUDIT_ENTRIES = 50
NUM_MASTERING_RECORDS = 50
NUM_MASTERING_JOB_EVENTS = 50
NUM_AI_USAGE_LOGS = 50

# Ретроспектива: данные за последние 120 дней
DAYS_BACK = 120
FAKE_EMAIL_DOMAIN = "test.local"
DEFAULT_PASSWORD = "TestPass123!"

STYLES = ["standard", "podcast", "loud", "vintage", "dynamic", "warm", "transparent"]


def _random_ts_in_range(days_back: int) -> float:
    """Случайная метка времени в последние days_back дней."""
    now = time.time()
    start = now - days_back * 86400
    return start + random.random() * (now - start)


def _day_start_ts(days_ago: int) -> float:
    """Начало дня N дней назад (полночь)."""
    dt = datetime.now() - timedelta(days=days_ago)
    return time.mktime(dt.replace(hour=0, minute=0, second=0, microsecond=0).timetuple())


def run():
    from app.database import (
        SessionLocal,
        create_tables,
        create_user,
        create_transaction,
        create_news_post,
        create_email_campaign,
        write_audit_log,
        get_user_by_email,
        User,
        Transaction,
        NewsPost,
        EmailCampaign,
        AuditLog,
        MasteringRecord,
        MasteringJobEvent,
        AiUsageLog,
        DB_AVAILABLE,
    )
    from app.auth import get_password_hash

    if not DB_AVAILABLE:
        print("БД недоступна (SQLAlchemy не установлен).")
        return

    create_tables()
    db = SessionLocal()
    created = {
        "users": 0,
        "transactions": 0,
        "news": 0,
        "campaigns": 0,
        "audit": 0,
        "mastering_records": 0,
        "mastering_job_events": 0,
        "ai_usage_logs": 0,
    }

    try:
        # ─── 1. Пользователи (48), с ретроспективным created_at ─────────────────
        hashed = get_password_hash(DEFAULT_PASSWORD)
        for i in range(1, NUM_USERS + 1):
            email = f"user{i}@{FAKE_EMAIL_DOMAIN}"
            if get_user_by_email(db, email):
                continue
            tier = random.choice(["free", "free", "pro", "pro", "studio"])
            u = create_user(db, email, hashed, tier=tier)
            if u:
                created["users"] += 1
                u.created_at = _random_ts_in_range(DAYS_BACK)
                db.commit()

        users = db.query(User).order_by(User.id).all()
        if not users:
            print("Нет пользователей в БД. Создайте хотя бы одного вручную или проверьте БД.")
            return
        user_ids = [u.id for u in users]
        admin_id = user_ids[0]
        admin_email = next((u.email for u in users if u.id == admin_id), "admin@test.local")

        # ─── 2. Транзакции (50), ретроспективные, привязаны к пользователям ───
        statuses = ["succeeded", "succeeded", "succeeded", "pending", "failed", "refunded"]
        payments = ["yookassa", "manual"]
        for i in range(NUM_TRANSACTIONS):
            if db.query(Transaction).count() >= NUM_TRANSACTIONS:
                break
            ts = _random_ts_in_range(DAYS_BACK)
            tx = Transaction(
                user_id=random.choice(user_ids),
                amount=round(random.uniform(99, 4999), 2),
                tier=random.choice(["pro", "studio", "pro"]),
                payment_system=random.choice(payments),
                status=random.choice(statuses),
                description=f"Оплата подписки / тест #{i+1}",
                created_at=ts,
            )
            db.add(tx)
            created["transactions"] += 1
        db.commit()

        # ─── 3. Новости (48), ретроспективные, автор — первый пользователь ─────
        titles = [
            "Обновление сервиса мастеринга",
            "Новые пресеты для подкастов",
            "Выход бета-версии пакетной обработки",
            "Изменения в тарифах",
            "Поддержка формата OPUS",
            "Улучшения алгоритма LUFS",
            "Плановые работы 15.03",
            "Акция для новых пользователей",
            "Интеграция с облачным хранилищем",
            "Рекомендации по громкости для стриминга",
            "Чёрная пятница: скидка 30%",
            "Новый стиль мастеринга: Vintage",
            "Отчёт о доступности за март",
            "Обновление политики конфиденциальности",
            "Благодарность сообществу",
            "Поддержка FLAC 24-bit",
            "Новый отчёт AI по анализу треков",
            "Веб-интерфейс: улучшения UX",
            "Тариф Studio: безлимитный мастеринг",
            "Советы по экспорту для Spotify и Apple",
            "Плановые работы 22.04",
            "Обновление зависимостей и стабильность",
            "История изменений (Changelog) за апрель",
            "Верификация email: что изменилось",
            "Новые шаблоны промптов для AI",
            "Итоги первого квартала",
            "Мастер-класс: громкость и динамика",
            "Поддержка формата AAC",
            "Улучшения отчёта по LUFS",
            "Релиз версии 0.5",
            "Техподдержка: новые часы работы",
            "Опрос качества сервиса",
            "Партнёрство со студиями",
            "Обновление документации API",
            "Безопасность: обновление JWT",
            "Новые пресеты для электронной музыки",
            "Экспорт в MP3: настройка битрейта",
            "Итоги месяца: статистика использования",
            "Напоминание: продление подписки",
            "Релиз версии 0.6",
            "Улучшения панели администратора",
            "Журнал аудита действий админов",
            "Отчёты по выручке и мастерингам",
            "Рассылки: таргет по тарифам",
            "Резервное копирование БД",
            "Миграции схемы БД",
            "Тестирование и стабильность",
            "Обратная связь от пользователей",
        ]
        for i, title in enumerate(titles[:NUM_NEWS]):
            if db.query(NewsPost).filter(NewsPost.title == title).first():
                continue
            ts = _random_ts_in_range(DAYS_BACK)
            is_pub = random.choice([True, True, False])
            post = NewsPost(
                title=title,
                body=f"<p>Текст новости для «{title}». Развёрнутое описание изменений или объявление.</p><p>Тестовая запись #{i+1}.</p>",
                author_id=admin_id,
                is_published=is_pub,
                published_at=ts if is_pub else None,
                created_at=ts,
                updated_at=ts,
            )
            db.add(post)
            created["news"] += 1
        db.commit()

        # ─── 4. Email-рассылки (48), часть — отправленные с sent_at в прошлом ────
        subjects = [
            "Новые возможности мастеринга",
            "Напоминание: продление подписки",
            "Специальное предложение Pro",
            "Итоги месяца",
            "Как добиться громкости -14 LUFS",
            "Приглашение на вебинар",
            "Обновление условий использования",
            "Скидка 20% на Studio",
            "Советы по подкастам",
            "Техподдержка: новые часы",
            "Опрос качества сервиса",
            "Релиз версии 0.4",
            "Чёрная пятница: скидки",
            "Новый пресет Podcast",
            "Рекомендации по экспорту",
            "Изменения в тарифах",
            "Поддержка OPUS",
            "Улучшения AI-отчёта",
            "Безопасность и пароли",
            "Итоги квартала",
            "Мастер-класс по LUFS",
            "Новые шаблоны промптов",
            "Обновление API",
            "Верификация email",
            "Статистика за месяц",
            "Акция для Free-пользователей",
            "Переход на Studio",
            "Обратная связь",
            "Документация обновлена",
            "Плановые работы",
            "Релиз 0.5",
            "Улучшения интерфейса",
            "Отчёты в админке",
            "Рассылки по тарифам",
            "Резервное копирование",
            "Новые отчёты",
            "Журнал аудита",
            "Экспорт данных",
            "Итоги года",
            "Благодарность пользователям",
            "Новогодняя акция",
            "Поддержка 24/7",
            "Часто задаваемые вопросы",
            "Обновление политики конфиденциальности",
        ]
        for i, subj in enumerate(subjects[:NUM_CAMPAIGNS]):
            if db.query(EmailCampaign).filter(EmailCampaign.subject == subj).first():
                continue
            ts = _random_ts_in_range(DAYS_BACK)
            status = random.choice(["draft", "draft", "sent", "sent", "sending"])
            total_rec = random.randint(10, 500) if status == "sent" else 0
            sent_cnt = total_rec if status == "sent" else 0
            camp = EmailCampaign(
                subject=subj,
                body_html=f"<h2>{subj}</h2><p>Содержимое рассылки #{i+1}.</p>",
                body_text=f"{subj}\n\nСодержимое рассылки #{i+1}.",
                target_tier=random.choice([None, "free", "pro", "studio"]),
                status=status,
                total_recipients=total_rec,
                sent_count=sent_cnt,
                created_at=ts,
                sent_at=ts + 3600 if status == "sent" else None,
            )
            db.add(camp)
            created["campaigns"] += 1
        db.commit()

        # ─── 5. Журнал аудита (50), ретроспективные даты ───────────────────────
        actions = [
            "settings_update", "patch_user", "login", "view_users", "prompt_update",
            "delete_user", "bulk_set_tier", "view_transactions", "create_news",
            "update_news", "send_campaign", "view_audit", "export_users", "block_user",
        ]
        target_types = ["user", "settings", "prompt", "news", "campaign", ""]
        for i in range(NUM_AUDIT_ENTRIES):
            ts = _random_ts_in_range(DAYS_BACK)
            entry = AuditLog(
                admin_id=admin_id,
                admin_email=admin_email,
                action=random.choice(actions),
                target_type=random.choice(target_types),
                target_id=random.choice(user_ids) if random.random() > 0.5 else None,
                details=f"Тестовая запись журнала #{i+1}",
                ip="127.0.0.1",
                created_at=ts,
            )
            db.add(entry)
            created["audit"] += 1
        db.commit()

        # ─── 6. MasteringRecord (50) — история мастерингов пользователей ─────────
        for i in range(NUM_MASTERING_RECORDS):
            ts = _random_ts_in_range(DAYS_BACK)
            style = random.choice(STYLES)
            before_lufs = round(random.uniform(-22, -10), 2)
            after_lufs = round(random.uniform(-16, -12), 2)
            target_lufs = -14.0
            rec = MasteringRecord(
                user_id=random.choice(user_ids),
                filename=f"track_{i+1}.wav",
                style=style,
                out_format=random.choice(["wav", "mp3", "flac"]),
                before_lufs=before_lufs,
                after_lufs=after_lufs,
                target_lufs=target_lufs,
                duration_sec=round(random.uniform(120, 360), 1),
                created_at=ts,
            )
            db.add(rec)
            created["mastering_records"] += 1
        db.commit()

        # ─── 7. MasteringJobEvent (50) — события мастеринга для отчётов ──────────
        for i in range(NUM_MASTERING_JOB_EVENTS):
            ts = _random_ts_in_range(DAYS_BACK)
            status = random.choice(["done", "done", "done", "error"])
            job_id = f"seed-job-{i+1}-{int(ts)}"
            ev = MasteringJobEvent(
                user_id=random.choice(user_ids) if random.random() > 0.2 else None,
                job_id=job_id,
                style=random.choice(STYLES),
                status=status,
                created_at=ts - 60,
                completed_at=ts if status in ("done", "error") else None,
            )
            db.add(ev)
            created["mastering_job_events"] += 1
        db.commit()

        # ─── 8. AiUsageLog (50) — вызовы AI для отчёта «Использование AI по типу» ─
        ai_types = ["recommend", "report", "nl_config", "chat"]
        for i in range(NUM_AI_USAGE_LOGS):
            ts = _random_ts_in_range(DAYS_BACK)
            log = AiUsageLog(
                user_id=random.choice(user_ids) if random.random() > 0.1 else None,
                type=random.choice(ai_types),
                tier=random.choice(["free", "pro", "studio"]),
                created_at=ts,
            )
            db.add(log)
            created["ai_usage_logs"] += 1
        db.commit()

        # ─── Итог ──────────────────────────────────────────────────────────────
        print("Тестовые данные созданы (ретроспектива: последние {} дней):".format(DAYS_BACK))
        print("  Пользователи:        {} (всего в БД: {})".format(created["users"], db.query(User).count()))
        print("  Транзакции:          {}".format(db.query(Transaction).count()))
        print("  Новости:             {}".format(db.query(NewsPost).count()))
        print("  Рассылки:            {}".format(db.query(EmailCampaign).count()))
        print("  Журнал аудита:      {}".format(db.query(AuditLog).count()))
        print("  Записи мастеринга:  {}".format(db.query(MasteringRecord).count()))
        print("  События задач:      {}".format(db.query(MasteringJobEvent).count()))
        print("  Логи AI:            {}".format(db.query(AiUsageLog).count()))
        print("\nПароль тестовых пользователей (user1@{} … user{}@{}): {}".format(
            FAKE_EMAIL_DOMAIN, NUM_USERS, FAKE_EMAIL_DOMAIN, DEFAULT_PASSWORD))
        print("В админке можно смотреть отчёты с фильтрами date_from / date_to.")
    finally:
        db.close()


if __name__ == "__main__":
    run()
