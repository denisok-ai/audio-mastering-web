# Production: отклонения сервера от репозитория

Чек-лист для оператора: убедиться, что на VPS нет «забытых» правок в Git, понять что лежит вне репозитория и как обновляться без конфликтов.

## 1. Ожидаемое состояние клона

После обновления:

```bash
cd /opt/magic-master    # или ваш INSTALL_DIR
git fetch origin
git status -sb
```

Норма: ветка `main` совпадает с `origin/main`, **нет** изменённых отслеживаемых файлов (`git diff` пустой).

Допустимо без внимания:

- неотслеживаемые файлы, перечисленные в `.gitignore` (в т.ч. каталоги **`backups/`** и **`backups_db/`** — архивы cron и ручные копии);
- файл **`.env`** (секреты, никогда не в Git).

Если `git status` показывает `?? backups/` до добавления правила в `.gitignore` — после `git pull` с актуальным репозиторием каталог можно оставить; он перестанет мешать статусу.

## 2. Конфигурация вне Git (не обновляется `git pull`)

При смене шаблонов в `deploy/` имеет смысл вручную сверить или переустановить:

| На сервере | Шаблон в репозитории |
|------------|----------------------|
| `/etc/systemd/system/magic-master.service` | [deploy/systemd/magic-master.service](../deploy/systemd/magic-master.service) |
| `/etc/nginx/sites-available/magic-master` (или `magic-master.conf`) | [deploy/nginx/magic-master.conf](../deploy/nginx/magic-master.conf) |
| `/etc/systemd/journald.conf.d/magic-master.conf` | По смыслу как [deploy/journald/99-magic-master-journal.conf](../deploy/journald/99-magic-master-journal.conf) |

Быстрая проверка совпадения nginx-шаблона с установленным файлом (пути подставьте свои):

```bash
diff -q /opt/magic-master/deploy/nginx/magic-master.conf /etc/nginx/sites-available/magic-master
```

Пустой вывод — файлы идентичны.

## 3. Типичные проблемы при `git pull`

**«Your local changes would be overwritten by merge»**

- Посмотреть: `git diff`, `git status`.
- Временно убрать правки: `git stash push -m "ops" -- путь/к/файлу` или `git checkout -- файл` (если правки не нужны).
- Затем: `git pull --ff-only`.

**Устаревший `git stash` с единственным изменением прав на `deploy/deploy.sh`**

В репозитории для `deploy/deploy.sh` зафиксирован исполняемый бит; после `git pull` такой stash не нужен:

```bash
git stash list
git stash show -p stash@{0}   # убедиться, что там только mode change
git stash drop stash@{0}    # если уверены
```

## 4. Бэкапы на диске

- Скрипт [deploy/backup_db.sh](../deploy/backup_db.sh) по умолчанию пишет в **`backups_db/`** в корне проекта.
- На части серверов дополнительно настроены полные архивы в **`backups/`** (tar.gz по cron) — это нормально; каталоги в `.gitignore`, в коммиты не попадают.

Контроль места:

```bash
du -sh /opt/magic-master/backups /opt/magic-master/backups_db 2>/dev/null
```

## 5. Логи и ротация

- Приложение: **journald** (`journalctl -u magic-master`). Ограничение диска — см. [deploy/journald/README.md](../deploy/journald/README.md).
- Nginx: пакет **`logrotate`** должен быть установлен, иначе конфиг в `/etc/logrotate.d/nginx` не выполняется. Проверка: `systemctl status logrotate.timer`.

## 6. Быстрая сверка одной командой (после SSH)

```bash
cd /opt/magic-master && \
  git status -sb && git diff --stat && \
  journalctl --disk-usage && \
  systemctl is-active magic-master logrotate.timer && \
  curl -sS http://127.0.0.1:8000/api/version
```

Подробнее по деплою: [DEPLOY.md](../DEPLOY.md). По сбоям: [RUNBOOK.md](RUNBOOK.md).
