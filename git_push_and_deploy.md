# Перенос на Git и обновление на сервере

Выполните команды **в терминале WSL** (или в корне проекта в Ubuntu).

---

## 1. Git: коммит и пуш

```bash
cd /home/denisok/projects/audio-mastering-web

# Посмотреть, что изменилось
git status

# Добавить все изменения
git add -A

# Коммит (подставьте своё сообщение при желании)
git commit -m "v0.5.6+: срез верхов 10%, тесты Dynamic EQ, правка перевода (плотность)"

# Отправить в удалённый репозиторий (origin/main или origin/master — проверьте ветку)
git push origin main
# или, если основная ветка называется master:
# git push origin master
```

Если при `git push` запросят логин/пароль — используйте токен доступа (Personal Access Token) вместо пароля или настройте SSH-ключ.

---

## 2. Сервер: обновление через Git

Если на сервере проект уже склонирован из Git:

```bash
# Подключиться к серверу
ssh user@91.218.244.146
# (подставьте своего пользователя и IP/домен)

# Перейти в каталог проекта (путь может быть другим, например /opt/magic-master)
cd /opt/magic-master
# или: cd /home/user/audio-mastering-web

# Подтянуть изменения
sudo git pull

# Запустить обновление (перезапуск сервиса, при необходимости — обновление venv)
cd deploy
sudo ./deploy.sh update
# или только перезапуск:
# sudo ./deploy.sh restart
```

---

## 3. Сервер: обновление через архив (если не используете Git на сервере)

На **своей машине** (WSL):

```bash
cd /home/denisok/projects/audio-mastering-web
chmod +x pack_for_deploy.sh
./pack_for_deploy.sh
# Появится файл: audio-mastering-web-deploy-YYYYMMDD.tar.gz
```

Скопировать архив на сервер:

```bash
scp audio-mastering-web-deploy-*.tar.gz user@91.218.244.146:/home/user/
```

На **сервере**:

```bash
cd /home/user/audio-mastering-web   # или /opt/magic-master
tar -xzf ~/audio-mastering-web-deploy-*.tar.gz --strip-components=0
# Или распаковать в пустую папку и заменить файлы вручную
sudo systemctl restart magic-master
```

---

После обновления проверьте версию в интерфейсе или: `curl -s http://localhost:8000/api/health | head -5`
