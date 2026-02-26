#!/usr/bin/env python3
# Диагностика мастеринга на своём файле.
# Использование: ./venv/bin/python run_diagnose_file.py /path/to/original.wav
# Создаёт в той же папке original_mastered.wav и выводит LUFS до/после, пики.

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import soundfile as sf
from app.pipeline import load_audio_from_bytes, export_audio, run_mastering_pipeline, measure_lufs


def main():
    if len(sys.argv) < 2:
        print("Использование: python run_diagnose_file.py /path/to/audio.wav")
        return 1
    in_path = Path(sys.argv[1]).resolve()
    if not in_path.is_file():
        print("Файл не найден:", in_path)
        return 1

    data = in_path.read_bytes()
    ext = in_path.suffix.lstrip(".").lower() or "wav"
    audio, sr = load_audio_from_bytes(data, ext)
    lufs_in = measure_lufs(audio, sr)
    peak_in = float(np.nanmax(np.abs(audio)))
    print("Вход:  shape=%s sr=%s LUFS=%.1f пик=%.4f" % (audio.shape, sr, lufs_in, peak_in))

    mastered = run_mastering_pipeline(audio, sr, target_lufs=-14.0)
    lufs_out = measure_lufs(mastered, sr)
    peak_out = float(np.nanmax(np.abs(mastered)))
    print("Выход: shape=%s LUFS=%.1f пик=%.4f" % (mastered.shape, lufs_out, peak_out))

    out_path = in_path.parent / (in_path.stem + "_mastered.wav")
    channels = 1 if mastered.ndim == 1 else mastered.shape[1]
    out_bytes = export_audio(mastered, sr, channels, "wav")
    out_path.write_bytes(out_bytes)
    print("Сохранено:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
