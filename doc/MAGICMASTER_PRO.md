# Домен magicmaster.pro и HTTPS на прод-сервере

IP VPS: `91.218.244.146`. Шаблоны в репозитории: [deploy/nginx/magic-master.conf](../deploy/nginx/magic-master.conf).

## Автоматизация на сервере (один запуск)

После `git pull` на сервере (или при свежем клоне) от **root**:

```bash
chmod +x /opt/magic-master/deploy/apply_magicmaster_pro_on_server.sh
export CERTBOT_EMAIL=ваш@email.com
bash /opt/magic-master/deploy/apply_magicmaster_pro_on_server.sh
```

Скрипт: обновляет `origin/main`, копирует nginx, при необходимости ставит bootstrap HTTP, выпускает Let's Encrypt (webroot), включает полный HTTPS-конфиг, дописывает `MAGIC_MASTER_CORS_ORIGINS` в `.env`, перезапускает `magic-master`.

Если репозиторий не в `/opt/magic-master`: `INSTALL_DIR=/путь/к/проекту bash ...`

Если `dig +short magicmaster.pro A` показывает **другой** IP — на регистраторе ещё не выставлена A-запись на этот сервер; дождитесь TTL или исправьте DNS.

## 1. DNS (у регистратора)

- **A**: `magicmaster.pro` → `91.218.244.146`
- **A** (или CNAME): `www.magicmaster.pro` → тот же IP или на `magicmaster.pro`

Проверка:

```bash
dig +short magicmaster.pro A
dig +short www.magicmaster.pro A
```

## 2. Диагностика на сервере (SSH)

```bash
ls -la /opt/magic-master 2>/dev/null
systemctl status magic-master --no-pager
ss -tlnp | grep -E ':80|:443|:8000'
nginx -t
grep -R "server_name\|listen" /etc/nginx/sites-enabled/ 2>/dev/null
ufw status
# при необходимости:
# ufw allow 80/tcp && ufw allow 443/tcp && ufw reload
```

## 3. SSL: рекомендуется Let's Encrypt (certbot)

Каталог для HTTP-01: `sudo mkdir -p /var/www/certbot`

**Первый выпуск**, если в конфиге уже есть блок `listen 443` и файлов сертификата ещё нет — временно отключите сайт или используйте standalone:

```bash
sudo systemctl stop nginx
sudo certbot certonly --standalone -d magicmaster.pro -d www.magicmaster.pro
sudo systemctl start nginx
```

Скопируйте актуальный [deploy/nginx/magic-master.conf](../deploy/nginx/magic-master.conf) в `/etc/nginx/sites-available/magic-master`, скопируйте [magic-master-proxy.inc](../deploy/nginx/magic-master-proxy.inc) рядом, затем:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

**Обновление сертификата** (webroot, nginx работает):

```bash
sudo certbot renew --webroot -w /var/www/certbot
```

**Альтернатива:** `sudo certbot --nginx -d magicmaster.pro -d www.magicmaster.pro` — certbot сам правит активный конфиг; после этого при обновлении из git сверяйте изменения с шаблоном в репозитории.

## 4. Переменные приложения (`/opt/magic-master/.env`)

```env
MAGIC_MASTER_CORS_ORIGINS=https://magicmaster.pro,https://www.magicmaster.pro
```

При оплате через YooKassa обновите `MAGIC_MASTER_YOOKASSA_RETURN_URL` на `https://magicmaster.pro/pricing?payment=success` (или ваш путь).

```bash
sudo systemctl restart magic-master
```

## 5. Проверка

```bash
curl -sI https://magicmaster.pro/api/health
curl -sI http://magicmaster.pro
openssl s_client -connect magicmaster.pro:443 -servername magicmaster.pro </dev/null 2>/dev/null | openssl x509 -noout -dates -subject
```

В браузере: замочек, отсутствие ошибок CORS в консоли.

## 6. Коммерческий сертификат (не Let's Encrypt)

Положите `fullchain.pem` и `privkey.pem` в например `/etc/ssl/magicmaster.pro/`, в `ssl_certificate` / `ssl_certificate_key` укажите эти пути вместо путей `/etc/letsencrypt/live/...`.

Локально в репозитории в `ssl/` лежат только ключ и CSR — для nginx нужен **выпущенный** сертификат от регистратора.
