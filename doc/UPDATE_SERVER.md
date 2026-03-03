# Обновление приложения на сервере

Сервер: `root@91.218.244.146`  
Каталог приложения: `/opt/magic-master`

---

## Сначала: отправить сборку на Git (с вашего компьютера)

Выполните в терминале **в каталоге проекта** (WSL или Linux):

```bash
cd /home/denisok/projects/audio-mastering-web

# Добавить все файлы, не коммитить БД
git add -A
git reset HEAD backend/magic_master.db

# Проверить список изменений
git status

# Закоммитить
git commit -m "Полная сборка: отладка без лимитов, кнопки AI после загрузки, стиль жанров, тост лимита, doc UPDATE_SERVER и режим отладки"

# Отправить на удалённый репозиторий
git push origin main
```

Либо одной командой (без проверки статуса):
```bash
cd /home/denisok/projects/audio-mastering-web && git add -A && git reset HEAD backend/magic_master.db && git commit -m "Полная сборка: отладка без лимитов, кнопки AI, стиль жанров, doc UPDATE_SERVER" && git push origin main
```

Скрипт из корня проекта:
```bash
bash push_to_git.sh -f
```

---

## Быстрое обновление (рекомендуется)

Подключитесь к серверу и выполните:

```bash
ssh root@91.218.244.146
cd /opt/magic-master

# Разрешить git работать в этом каталоге (если ещё не делали)
git config --global --add safe.directory /opt/magic-master

# Скачать последние изменения
git pull origin main

# Обновить зависимости Python и перезапустить сервис
sudo ./deploy/deploy.sh update
```

Скрипт `deploy.sh update` сам выполнит:
- `git pull --ff-only` (если запускать из каталога с .git)
- `pip install -r backend/requirements.txt`
- `systemctl restart magic-master`

---

## Обновление вручную (без скрипта)

```bash
ssh root@91.218.244.146
cd /opt/magic-master

git config --global --add safe.directory /opt/magic-master
git pull origin main

# Активировать venv и установить зависимости
/opt/magic-master/venv/bin/pip install -r backend/requirements.txt -q

# Перезапустить приложение
sudo systemctl restart magic-master
```

---

## Проверка после обновления

```bash
# Статус сервиса
sudo systemctl status magic-master

# Проверка API
curl -s http://127.0.0.1:8000/api/health
```

Ожидаемый ответ: `{"status":"ok", ...}`.

---

## Логи (если что-то пошло не так)

```bash
# Последние 100 строк
sudo journalctl -u magic-master -n 100 --no-pager

# В реальном времени
sudo journalctl -u magic-master -f
```

---

## Запуск в режиме отладки на сервере

В режиме отладки отключены лимиты мастеринга (429), счётчик «осталось мастерингов» не применяется, интерфейс показывает «Режим отладки · все функции без входа».

**1. Включить режим отладки**

На сервере создайте или отредактируйте файл `.env` в каталоге приложения и добавьте (или измените) строку:

```bash
ssh root@91.218.244.146
cd /opt/magic-master

# Создать .env из примера, если файла ещё нет
[ -f .env ] || cp .env.example .env

# Включить режим отладки (подставить или добавить строку)
grep -q '^MAGIC_MASTER_DEBUG=1' .env || echo 'MAGIC_MASTER_DEBUG=1' >> .env
# Либо отредактировать вручную:
nano .env
```

В `.env` должна быть строка (без кавычек, без пробелов вокруг `=`):
```env
MAGIC_MASTER_DEBUG=1
```

**2. Перезапустить сервис**

Сервис читает переменные из `/opt/magic-master/.env` (в unit-файле указано `EnvironmentFile=/opt/magic-master/.env`):

```bash
sudo systemctl restart magic-master
sudo systemctl status magic-master
```

**3. Проверить**

Откройте в браузере `http://91.218.244.146:8000` (или ваш домен). В интерфейсе должен отображаться режим отладки, лимиты мастерингов не должны срабатывать.

**4. Выключить режим отладки**

Удалите строку `MAGIC_MASTER_DEBUG=1` из `/opt/magic-master/.env` или замените на `MAGIC_MASTER_DEBUG=0`, затем снова выполните:

```bash
sudo systemctl restart magic-master
```
