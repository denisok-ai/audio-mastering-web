#!/bin/bash
# Быстрый коммит и пуш. Запуск из корня проекта: ./git_push.sh
set -e
cd "$(dirname "$0")"
git add -A
git status -s
echo ""
read -p "Коммит и пуш? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[YyДд]$ ]]; then
  git commit -m "Лендинг: мобильная кнопка CTA — размер, контраст (граница и тень)"
  git push origin main 2>/dev/null || git push origin master 2>/dev/null || git push
  echo "Готово. На сервере выполните: cd /opt/magic-master && sudo git pull && cd deploy && sudo ./deploy.sh update"
fi
