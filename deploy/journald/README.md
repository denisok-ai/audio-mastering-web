# Ротация логов на production

Приложение запускается через **systemd** (`magic-master.service`) и **uvicorn**: stdout/stderr попадают в **journald**, а не в отдельные файлы в каталоге приложения.

## Проверка на сервере

```bash
# Использование диска журналом
journalctl --disk-usage

# Логи только сервиса (последний час)
journalctl -u magic-master --since "1 hour ago" -n 200 --no-pager
```

## Ограничение размера journald

Чтобы журнал не разрастался бесконечно, скопируйте фрагмент:

```bash
sudo mkdir -p /etc/systemd/journald.conf.d
sudo cp /opt/magic-master/deploy/journald/99-magic-master-journal.conf /etc/systemd/journald.conf.d/
sudo systemctl restart systemd-journald
```

Путь `/opt/magic-master` замените на ваш `INSTALL_DIR`, если отличается.

## Nginx

Обычно ротация уже настроена пакетом `nginx` в `/etc/logrotate.d/nginx`. Проверка:

```bash
sudo logrotate -d /etc/logrotate.d/nginx
```

## Файловые логи приложения

Если позже включите запись в файлы (например `/var/log/magic-master/*.log`), добавьте запись в `/etc/logrotate.d/magic-master-app` с `copytruncate` или `postrotate` для сигнала процессу.
