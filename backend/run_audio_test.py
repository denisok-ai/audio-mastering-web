#!/usr/bin/env python3
# Диагностика конвейера мастеринга: тон 440 Hz -> пайплайн -> WAV.
# Запуск из backend: ./venv/bin/python run_audio_test.py
# Создаёт test_sine.wav и test_mastered.wav. Воспроизведите test_mastered.wav.

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import soundfile as sf
from app.pipeline import load_audio_from_bytes, export_audio, run_mastering_pipeline


def main():
    sr = 44100
    duration_sec = 2.0
    freq = 440.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    stereo = np.column_stack([sine, sine])

    backend_dir = Path(__file__).parent
    in_path = backend_dir / "test_sine.wav"
    sf.write(in_path, stereo, sr, format="WAV", subtype="PCM_16")
    print("Записано:", in_path)

    with open(in_path, "rb") as f:
        data = f.read()
    audio, loaded_sr = load_audio_from_bytes(data, "wav")
    print("Загружено: shape=%s, sr=%s" % (audio.shape, loaded_sr))

    mastered = run_mastering_pipeline(audio, loaded_sr, target_lufs=-14.0)
    print("После пайплайна: shape=%s, пик=%.4f" % (mastered.shape, float(np.max(np.abs(mastered)))))

    channels = 1 if mastered.ndim == 1 else mastered.shape[1]
    out_bytes = export_audio(mastered, loaded_sr, channels, "wav")

    out_path = backend_dir / "test_mastered.wav"
    with open(out_path, "wb") as f:
        f.write(out_bytes)
    print("Записано:", out_path)

    read_back, _ = sf.read(io.BytesIO(out_bytes))
    peak = float(np.max(np.abs(read_back)))
    print("Проверка WAV: shape=%s, пик=%.4f" % (read_back.shape, peak))
    if peak < 1e-4:
        print("ОШИБКА: выходной файл тихий.")
        return 1
    print("OK. Воспроизведите test_mastered.wav — тон 440 Hz без писка.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
