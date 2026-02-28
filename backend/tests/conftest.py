"""Pytest fixtures for Magic Master API tests."""
import io
import pytest


@pytest.fixture
def minimal_wav_bytes() -> bytes:
    """0.1s of silence — WAV 44100 Hz, mono, float32."""
    try:
        import soundfile as sf
        import numpy as np

        buf = io.BytesIO()
        silence = np.zeros(4410, dtype=np.float32)
        sf.write(buf, silence, 44100, format="WAV", subtype="FLOAT")
        return buf.getvalue()
    except ImportError:
        # Minimal WAV header (44 bytes) + 100 samples of silence (float32 = 400 bytes)
        import struct
        import numpy as np

        samples = np.zeros(4410, dtype=np.float32).tobytes()
        data_size = len(samples)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            3,       # PCM float
            1,       # mono
            44100,   # sample rate
            44100 * 4,  # byte rate
            4,       # block align
            32,      # bits per sample
            b"data",
            data_size,
        )
        return header + samples
