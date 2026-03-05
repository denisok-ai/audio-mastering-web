# Пресеты сообщества

Файл `presets_community.json` подгружается по `GET /api/presets/community` и отображается в приложении в блоке «Загрузить пресет» (optgroup «Пресеты сообщества»). Доступны без авторизации.

Дополнительные пресеты можно подгрузить через переменную окружения `MAGIC_MASTER_COMMUNITY_PRESETS_EXTRA` (путь к одному `.json`-файлу или к каталогу с `.json`). Подробнее: `doc/PLUGINS_EXTENSIONS.md`.

## Формат записи

Каждый пресет — объект JSON:

| Поле | Тип | Описание |
|------|-----|----------|
| `id` | string | Уникальный идентификатор (латиница, без пробелов). |
| `name` | string | Название для UI (например «Stream (−14 LUFS)»). |
| `target_lufs` | number | Целевая громкость в LUFS (−60…−1). |
| `style` | string | Ключ жанра из `STYLE_CONFIGS` в `pipeline.py`: `standard`, `edm`, `hiphop`, `classical`, `podcast`, `lofi`, `house_basic`, `dry_vocal`. |
| `chain_config` | object \| null | Опционально: полный конфиг цепочки модулей. Если `null`, используется цепочка по умолчанию для выбранного `style` и `target_lufs`. |

## Как добавить пресет

1. Откройте `presets_community.json`.
2. Добавьте объект в массив (с запятой после предыдущего).
3. Убедитесь, что `style` совпадает с одним из ключей в `pipeline.STYLE_CONFIGS`.
4. Перезапуск сервера не обязателен: файл читается при каждом запросе к `/api/presets/community`.

Пример:

```json
{
  "id": "my_preset",
  "name": "Мой пресет (−14 LUFS)",
  "target_lufs": -14,
  "style": "standard",
  "chain_config": null
}
```
