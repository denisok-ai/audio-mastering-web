#!/usr/bin/env python3
"""
Исследовательский скрипт для задачи 9.2 — изоляция вокала (Demucs).
Использование:
  pip install -r requirements-vocal-isolation.txt
  python run_isolate_vocal.py /path/to/track.wav [--output DIR]
Создаёт в DIR (или рядом с файлом) папку с выделенной дорожкой vocals.wav.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Изоляция вокала через Demucs (задача 9.2)")
    parser.add_argument("input", type=Path, help="Входной аудиофайл (WAV, MP3, FLAC)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Папка для результата (по умолчанию рядом с файлом)")
    parser.add_argument("-n", "--model", default="htdemucs", help="Модель Demucs (htdemucs, htdemucs_ft, …)")
    args = parser.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        print("Файл не найден:", inp, file=sys.stderr)
        return 1

    out_dir = args.output.resolve() if args.output else inp.parent / (inp.stem + "_vocal_isolated")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Запуск Demucs: только два стема (vocals / no_vocals), чтобы быстрее и меньше места
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", args.model,
        "--two-stems", "vocals",
        str(inp),
        "-o", str(out_dir),
    ]
    # Проверка наличия demucs до долгого запуска
    try:
        subprocess.run(
            [sys.executable, "-c", "import demucs"],
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print(
            "Demucs не установлен. Установите опциональные зависимости:\n"
            "  pip install -r requirements-vocal-isolation.txt\n"
            "См. doc/PLAN_9_2_VOCAL_ISOLATION.md",
            file=sys.stderr,
        )
        return 1

    print("Запуск:", " ".join(cmd))
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Ошибка Demucs (код %s):" % getattr(e, "returncode", ""), e, file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0

    # Demucs пишет в out_dir/<model>/<trackname>/vocals.wav
    model_dir = out_dir / args.model
    if not model_dir.is_dir():
        print("Ожидалась папка:", model_dir, file=sys.stderr)
        return 1
    track_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not track_dirs:
        print("Не найден подкаталог трека в", model_dir, file=sys.stderr)
        return 1
    vocals_path = track_dirs[0] / "vocals.wav"
    if not vocals_path.is_file():
        print("Не найден vocals.wav в", track_dirs[0], file=sys.stderr)
        return 1

    # Копируем в корень out_dir для удобства
    dest = out_dir / "vocals.wav"
    shutil.copy2(vocals_path, dest)
    print("Готово за %.1f с. Вокал: %s" % (elapsed, dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
