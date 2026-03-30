# Регрессия мастеринга по WAV

## Файл

Положите проблемный трек (например `_Alors On Danse Rem.wav`) как:

`backend/tests/fixtures/mastering_regression/alors_on_danse_rem.wav`

Длительность должна покрывать окна анализа: **не короче ~160 с** (окно 2:34–2:40).

Либо задайте путь через окружение:

```bash
export MM_REGRESSION_WAV=/path/to/track.wav
cd backend && ./venv/bin/python -m pytest tests/test_mastering_regression_windows.py -v
```

## Опциональные пороги

Файл `expected_metrics.json` в этом каталоге (см. пример в репозитории или создайте после калибровки) может задавать верхние границы метрик по окнам. Без файла тесты используют мягкие встроенные проверки.

## Параметры прогона

Как пресет Stream на сайте: **v2 default chain**, `style=standard`, `target_lufs=-14`.

Подробнее: [doc/MASTERING_REGRESSION.md](../../../doc/MASTERING_REGRESSION.md).
