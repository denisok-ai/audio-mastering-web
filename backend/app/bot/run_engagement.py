"""Запуск из backend: PYTHONPATH=. python -m app.bot.run_engagement"""
from __future__ import annotations

import json
import sys


def main() -> None:
    from .engagement import post_channel_weekly_stats, process_engagement_due

    r = process_engagement_due()
    # Еженедельный пост в канал — раскомментируйте в cron раз в 7 дней:
    # post_channel_weekly_stats()
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
    sys.exit(0)
