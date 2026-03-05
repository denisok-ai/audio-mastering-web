"""Сервис изоляции вокала (задача 9.2) — вызов Demucs через subprocess.

Не импортирует demucs/torch, чтобы основной процесс работал без опциональных зависимостей.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def isolate_vocal(audio_bytes: bytes, input_filename: str, model: str = "htdemucs") -> bytes:
    """
    Выделяет вокал из аудиофайла с помощью Demucs (subprocess).
    Возвращает WAV-байты дорожки vocals.
    При недоступности demucs или ошибке запуска бросает RuntimeError.
    """
    with tempfile.TemporaryDirectory(prefix="magic_master_vocal_") as tmp:
        tmp_path = Path(tmp)
        inp = tmp_path / (Path(input_filename).name or "input.wav")
        inp.write_bytes(audio_bytes)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        cmd = [
            sys.executable, "-m", "demucs",
            "-n", model,
            "--two-stems", "vocals",
            str(inp),
            "-o", str(out_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                "Demucs завершился с ошибкой (код %s): %s"
                % (result.returncode, (result.stderr or result.stdout or "")[:500])
            )

        model_dir = out_dir / model
        if not model_dir.is_dir():
            raise RuntimeError("Demucs не создал каталог модели: %s" % model_dir)
        track_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not track_dirs:
            raise RuntimeError("Не найден подкаталог трека в %s" % model_dir)
        vocals_path = track_dirs[0] / "vocals.wav"
        if not vocals_path.is_file():
            raise RuntimeError("Не найден vocals.wav в %s" % track_dirs[0])
        return vocals_path.read_bytes()


def is_demucs_available() -> bool:
    """Проверяет, установлен ли модуль demucs (без загрузки модели)."""
    try:
        subprocess.run(
            [sys.executable, "-c", "import demucs"],
            capture_output=True,
            check=True,
            timeout=15,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
