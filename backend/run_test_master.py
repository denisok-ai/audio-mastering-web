#!/usr/bin/env python3
"""
Тестирование мастеринга и выдача тестового мастер-файла.
Генерирует стерео-сигнал (аккорд + бас), прогоняет через пайплайн, сохраняет WAV.
Запуск: ./venv/bin/python run_test_master.py
Выход: ../test_output/test_master.wav (и отчёт в консоль).
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import soundfile as sf
from app.pipeline import (
    load_audio_from_bytes,
    export_audio,
    run_mastering_pipeline,
    measure_lufs,
)


def main():
    sr = 44100
    duration_sec = 5.0
    n = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n, dtype=np.float32)

    # Сигнал: ля-минорный аккорд (A3, C4, E4) + бас A2
    a3 = 220.0
    c4 = 261.63
    e4 = 329.63
    a2 = 110.0
    chord = (
        0.15 * np.sin(2 * np.pi * a3 * t)
        + 0.12 * np.sin(2 * np.pi * c4 * t)
        + 0.12 * np.sin(2 * np.pi * e4 * t)
        + 0.2 * np.sin(2 * np.pi * a2 * t)
    ).astype(np.float32)
    # Стерео (L = R)
    stereo = np.column_stack([chord, chord])

    backend_dir = Path(__file__).resolve().parent
    out_dir = backend_dir.parent / "test_output"
    out_dir.mkdir(exist_ok=True)

    in_path = backend_dir / "test_signal.wav"
    sf.write(in_path, stereo, sr, format="WAV", subtype="PCM_16")
    print("Тестовый сигнал записан:", in_path)

    with open(in_path, "rb") as f:
        data = f.read()
    audio, loaded_sr = load_audio_from_bytes(data, "wav")
    lufs_in = measure_lufs(audio, loaded_sr)
    peak_in = float(np.max(np.abs(audio)))
    print("Вход: shape=%s, LUFS=%.2f, пик=%.4f" % (audio.shape, lufs_in, peak_in))

    target_lufs = -14.0
    mastered = run_mastering_pipeline(audio, loaded_sr, target_lufs=target_lufs)
    lufs_out = measure_lufs(mastered, loaded_sr)
    peak_out = float(np.max(np.abs(mastered)))
    print("После мастеринга: shape=%s, LUFS=%.2f, пик=%.4f" % (mastered.shape, lufs_out, peak_out))

    channels = 1 if mastered.ndim == 1 else mastered.shape[1]
    out_bytes = export_audio(mastered, loaded_sr, channels, "wav")

    master_path = out_dir / "test_master.wav"
    with open(master_path, "wb") as f:
        f.write(out_bytes)
    print("Тестовый мастер-файл записан:", master_path)

    read_back, _ = sf.read(io.BytesIO(out_bytes))
    peak_file = float(np.max(np.abs(read_back)))
    if peak_file < 1e-4:
        print("ОШИБКА: выходной файл тихий.")
        return 1
    print("Проверка WAV: пик=%.4f, длина %.1f с" % (peak_file, len(read_back) / loaded_sr))
    print()
    print("=== ИТОГ ТЕСТИРОВАНИЯ ===")
    print("  Вход:  LUFS %.2f, пик %.4f" % (lufs_in, peak_in))
    print("  Выход: LUFS %.2f (цель %.1f), пик %.4f" % (lufs_out, target_lufs, peak_out))
    print("  Файл: %s" % master_path)
    print("  Воспроизведите test_master.wav для проверки качества.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
