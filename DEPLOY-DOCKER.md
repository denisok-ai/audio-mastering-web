# Запуск Magic Master в Docker на сервере

Сервер: `ssh root@46.19.68.241`.  
На сервере уже есть Docker с **Magic Master** и **другой проект** — они работают автономно, обновляем только Magic Master по имени контейнера.

---

## Диагностика: чтобы понять окружение на сервере

Выполни на сервере и пришли вывод (или сохрани в файл) — по нему будет понятно, как всё стоит и что не трогать.

```bash
ssh root@46.19.68.241
```

Дальше по очереди:

```bash
# Какие контейнеры запущены (имена, образы, порты)
docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}"

# Только образы (чтобы увидеть имя образа Magic Master)
docker images

# Есть ли docker-compose в каталогах (чтобы понять, как поднят другой проект)
ls -la /root/*/docker-compose*.yml /root/*/compose*.yml 2>/dev/null || true
ls -la /opt/*/docker-compose*.yml 2>/dev/null || true

# Как именно запущен контейнер Magic Master (если есть)
docker inspect magic-master --format '{{.Name}} Image:{{.Config.Image}} Ports:{{.NetworkSettings.Ports}}' 2>/dev/null || echo "Контейнер magic-master не найден"
```

По выводу будет видно: имя контейнера Magic Master, образ, порт и что за второй проект — его не трогаем.

---

## Обновление только Magic Master (другой проект не трогаем)

Работаем **только с контейнером и образом Magic Master**. Остальные контейнеры не останавливаем и не пересоздаём.

### Вариант A: образ привозить с машины (scp)

**У себя (сборка и отправка):**

```bash
cd /home/denisok/projects/audio-mastering-web
docker build -t magic-master:latest .
docker save magic-master:latest | gzip > magic-master.tar.gz
scp magic-master.tar.gz root@46.19.68.241:/root/
```

**На сервере (обновление только Magic Master):**

```bash
ssh root@46.19.68.241

# Загрузить новый образ
docker load < /root/magic-master.tar.gz

# Остановить и удалить только контейнер Magic Master (другие не трогаем)
docker stop magic-master
docker rm magic-master

# Запустить заново с теми же портами (подставь свой порт, если не 8001)
docker run -d \
  --name magic-master \
  --restart unless-stopped \
  -p 8001:8000 \
  magic-master:latest

# Проверка
docker ps | grep magic-master
curl -s http://localhost:8001/api/health
```

### Вариант B: сборка на сервере из Git

**На сервере:**

```bash
ssh root@46.19.68.241

cd /root/audio-mastering-web   # или путь, где лежит репо
git pull

docker build -t magic-master:latest .

docker stop magic-master
docker rm magic-master

docker run -d \
  --name magic-master \
  --restart unless-stopped \
  -p 8001:8000 \
  magic-master:latest

docker ps | grep magic-master
curl -s http://localhost:8001/api/health
```

Если у тебя на сервере другой **порт** (не 8001) или другое **имя контейнера** — в командах выше замени `magic-master` и `8001` на свои. После диагностики могу подсказать точные строки под твоё окружение.

---

## 1. На своей машине: собрать образ и отправить на сервер (первый запуск)

```bash
cd /home/denisok/projects/audio-mastering-web
docker build -t magic-master:latest .
docker save magic-master:latest | gzip > magic-master.tar.gz
scp magic-master.tar.gz root@46.19.68.241:/root/
```

Либо через registry (если есть Docker Hub / свой registry):

```bash
docker tag magic-master:latest YOUR_LOGIN/magic-master:latest
docker push YOUR_LOGIN/magic-master:latest
```

---

## 2. На сервере: загрузить образ и запустить

Подключение:

```bash
ssh root@46.19.68.241
```

Если образ передавали через файл:

```bash
docker load < /root/magic-master.tar.gz
```

Запуск контейнера (порт **8001**, чтобы не пересекаться с другим проектом):

```bash
docker run -d \
  --name magic-master \
  --restart unless-stopped \
  -p 8001:8000 \
  magic-master:latest
```

Проверка:

```bash
docker ps
curl -s http://localhost:8001/api/health
```

В браузере: **http://46.19.68.241:8001**

---

## 3. Первый запуск: сборка образа на сервере (git clone)

На сервере (один раз):

```bash
cd /root
git clone https://github.com/YOUR_LOGIN/audio-mastering-web.git
cd audio-mastering-web
docker build -t magic-master:latest .
docker run -d --name magic-master --restart unless-stopped -p 8001:8000 magic-master:latest
```

Дальнейшие обновления — блок **«Обновление только Magic Master»** выше (вариант A или B).

---

## 4. Полезные команды

| Действие              | Команда |
|-----------------------|--------|
| Логи                  | `docker logs -f magic-master` |
| Остановить            | `docker stop magic-master` |
| Запустить снова       | `docker start magic-master` |
| Удалить контейнер     | `docker stop magic-master && docker rm magic-master` |

Порт **8001** выбран чтобы не конфликтовать с другим проектом; при необходимости замените на свой (в команде меняется только `-p 8001:8000`).
