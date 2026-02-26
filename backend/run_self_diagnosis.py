#!/usr/bin/env python3
# Самодиагностика конвейера мастеринга (стандарты Sony/Warner).
# Запуск: ./venv/bin/python run_self_diagnosis.py [путь к WAV]
# Без аргумента — тест на сгенерированном тоне.

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import soundfile as sf
from app.pipeline import (
    PRESET_LUFS,
    TRUE_PEAK_LIMIT_DB,
    load_audio_from_bytes,
    export_audio,
    run_mastering_pipeline,
    measure_lufs,
)


def peak_db(peak_linear: float) -> float:
    if peak_linear <= 0:
        return -100.0
    return 20.0 * np.log10(min(peak_linear, 1.0))


def run_diagnosis(source_path: Path, target_lufs: float = -14.0) -> dict:
    data = source_path.read_bytes()
    ext = source_path.suffix.lstrip(".").lower() or "wav"
    audio, sr = load_audio_from_bytes(data, ext)
    lufs_in = measure_lufs(audio, sr)
    peak_in = float(np.nanmax(np.abs(audio)))
    peak_in_db = peak_db(peak_in)

    mastered = run_mastering_pipeline(audio, sr, target_lufs=target_lufs)
    lufs_out = measure_lufs(mastered, sr)
    peak_out = float(np.nanmax(np.abs(mastered)))
    peak_out_db = peak_db(peak_out)

    channels = 1 if mastered.ndim == 1 else mastered.shape[1]
    out_bytes = export_audio(mastered, sr, channels, "wav")
    read_back, _ = sf.read(io.BytesIO(out_bytes))
    peak_file = float(np.nanmax(np.abs(read_back)))
    peak_file_db = peak_db(peak_file)

    has_nan = np.any(np.isnan(mastered)) or np.any(np.isnan(read_back))
    lufs_ok = abs(lufs_out - target_lufs) <= 1.0
    true_peak_ok = peak_file_db <= (TRUE_PEAK_LIMIT_DB + 1.0)

    return {
        "sr": sr,
        "shape_in": audio.shape,
        "shape_out": mastered.shape,
        "lufs_in": lufs_in,
        "lufs_out": lufs_out,
        "target_lufs": target_lufs,
        "peak_in": peak_in,
        "peak_in_db": peak_in_db,
        "peak_out": peak_out,
        "peak_out_db": peak_out_db,
        "peak_file_db": peak_file_db,
        "has_nan": has_nan,
        "lufs_ok": lufs_ok,
        "true_peak_ok": true_peak_ok,
        "out_bytes": len(out_bytes),
    }


def main():
    target_lufs = -14.0
    if len(sys.argv) >= 2:
        source_path = Path(sys.argv[1]).resolve()
        if not source_path.is_file():
            print("Файл не найден:", source_path)
            return 1
        if len(sys.argv) >= 3:
            try:
                target_lufs = float(sys.argv[2])
            except ValueError:
                pass
    else:
        backend_dir = Path(__file__).resolve().parent
        # Генерируем тестовый тон
        sr = 44100
        t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
        sine = 0.4 * np.sin(2 * np.pi * 440 * t)
        stereo = np.column_stack([sine, sine])
        source_path = backend_dir / "test_sine.wav"
        sf.write(source_path, stereo, sr, format="WAV", subtype="PCM_16")
        print("Используется тестовый тон:", source_path)

    r = run_diagnosis(source_path, target_lufs=target_lufs)

    print()
    print("=== САМОДИАГНОСТИКА МАСТЕРИНГА (Sony/Warner-ориентированные стандарты) ===")
    print()
    print("Вход:     shape=%s  sr=%s  LUFS=%.2f  пик=%.4f (%.2f dB)" % (
        r["shape_in"], r["sr"], r["lufs_in"], r["peak_in"], r["peak_in_db"]))
    print("Выход:    shape=%s  LUFS=%.2f  пик=%.4f (%.2f dB)" % (
        r["shape_out"], r["lufs_out"], r["peak_out"], r["peak_out_db"]))
    print("Файл WAV: пик=%.2f dB  размер=%d байт" % (r["peak_file_db"], r["out_bytes"]))
    print()
    print("Целевой LUFS: %.1f" % r["target_lufs"])
    print("True Peak лимит: %.1f dBTP" % TRUE_PEAK_LIMIT_DB)
    print()
    print("Проверки:")
    print("  LUFS в пределах 1 dB от цели:  %s" % ("OK" if r["lufs_ok"] else "ОШИБКА"))
    print("  True Peak ≤ -1 dBTP:           %s" % ("OK" if r["true_peak_ok"] else "ОШИБКА"))
    print("  Нет NaN:                      %s" % ("OK" if not r["has_nan"] else "ОШИБКА"))
    print()
    if r["lufs_ok"] and r["true_peak_ok"] and not r["has_nan"]:
        print("Итог: самодиагностика пройдена.")
        return 0
    print("Итог: есть замечания (см. выше).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
