#!/usr/bin/env bash
# Отправка полной сборки на git. Запуск из корня проекта в WSL:
#   bash push_to_git.sh
# или без подтверждения:
#   bash push_to_git.sh -f
set -e
cd "$(dirname "$0")"
git add -A
git reset HEAD backend/magic_master.db 2>/dev/null || true
echo "Изменённые файлы:"
git status -s
if [[ "${1:-}" = "-f" ]]; then
  DO=1
else
  echo "---"
  read -p "Закоммитить и отправить на origin main? [y/N] " -n 1 -r
  echo
  [[ $REPLY =~ ^[yY]$ ]] && DO=1
fi
if [[ -n "${DO:-}" ]]; then
  git commit -m "Полная сборка: отладка без лимитов на бэке, кнопки AI после загрузки, стиль жанров, тост лимита, doc UPDATE_SERVER и режим отладки" || true
  git push origin main
  echo "Готово. Обновление на сервере: cd /opt/magic-master && git pull origin main && sudo ./deploy/deploy.sh update"
fi
