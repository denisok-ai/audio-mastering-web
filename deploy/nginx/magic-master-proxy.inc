# Включается из magic-master.conf (server { ... include magic-master-proxy.inc; })
# Проксирование на Uvicorn 127.0.0.1:8000

client_max_body_size 260M;
proxy_connect_timeout 60s;
proxy_send_timeout 300s;
proxy_read_timeout 300s;
proxy_http_version 1.1;
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;

# SSE: прогресс мастеринга — без буферизации, длинный таймаут
location /api/master/progress/ {
    proxy_pass http://magic_master;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
    chunked_transfer_encoding off;
}

# API и приложение
location /api/ {
    proxy_pass http://magic_master;
}

location / {
    proxy_pass http://magic_master;
}

# Service Worker — без кэширования
location = /sw.js {
    proxy_pass http://magic_master;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    add_header Cache-Control "no-cache";
}
