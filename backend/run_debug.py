#!/usr/bin/env python3
"""
run_debug.py — Полная отладка всех компонентов Magic Master.

Запуск с тестовым синусом (сгенерированным):
    cd backend
    ./venv/bin/python run_debug.py

Запуск с реальным треком (рекомендуется):
    ./venv/bin/python run_debug.py ../doc/Pre-Master.wav

Можно передать любой WAV/FLAC-файл.
"""
import io
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS  = "\033[92m[OK  ]\033[0m"
FAIL  = "\033[91m[FAIL]\033[0m"
WARN  = "\033[93m[WARN]\033[0m"
INFO  = "\033[94m[INFO]\033[0m"

errors: list[tuple[str, str]] = []


def check(name: str, fn):
    try:
        result = fn()
        suffix = f"  → {result}" if result is not None else ""
        print(f"  {PASS}  {name}{suffix}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        errors.append((name, traceback.format_exc()))
        return False


def section(title: str):
    print()
    print(f"\033[96m{'='*60}\033[0m")
    print(f"\033[96m  {title}\033[0m")
    print(f"\033[96m{'='*60}\033[0m")


# ──────────────────────────────────────────────────────────────
section("1. Импорт базовых библиотек")
# ──────────────────────────────────────────────────────────────
def _ver(pkg):
    m = __import__(pkg)
    return getattr(m, "__version__", None) or getattr(m, "version", None) or "installed"

check("numpy",       lambda: _ver("numpy"))
check("scipy",       lambda: _ver("scipy"))
check("soundfile",   lambda: _ver("soundfile"))
check("pyloudnorm",  lambda: _ver("pyloudnorm"))
check("pydub",       lambda: _ver("pydub"))
check("fastapi",     lambda: _ver("fastapi"))
check("uvicorn",     lambda: _ver("uvicorn"))

# ──────────────────────────────────────────────────────────────
section("2. Импорт модулей проекта")
# ──────────────────────────────────────────────────────────────
check("app.config",                lambda: __import__("app.config", fromlist=["settings"]).settings.temp_dir)
check("app.pipeline",              lambda: __import__("app.pipeline", fromlist=["measure_lufs"]).measure_lufs)
check("app.modules.base",          lambda: __import__("app.modules.base", fromlist=["BaseModule"]).BaseModule)
check("app.modules.dc_offset",     lambda: __import__("app.modules.dc_offset", fromlist=["DCOffsetModule"]).DCOffsetModule.module_id)
check("app.modules.peak_guard",    lambda: __import__("app.modules.peak_guard", fromlist=["PeakGuardModule"]).PeakGuardModule.module_id)
check("app.modules.equalizer",     lambda: __import__("app.modules.equalizer", fromlist=["TargetCurveModule"]).TargetCurveModule.module_id)
check("app.modules.dynamics",      lambda: __import__("app.modules.dynamics", fromlist=["DynamicsModule"]).DynamicsModule.module_id)
check("app.modules.maximizer",     lambda: __import__("app.modules.maximizer", fromlist=["MaximizerModule"]).MaximizerModule.module_id)
check("app.modules.normalize_lufs",lambda: __import__("app.modules.normalize_lufs", fromlist=["NormalizeLUFSModule"]).NormalizeLUFSModule.module_id)
check("app.modules.exciter",       lambda: __import__("app.modules.exciter", fromlist=["ExciterModule"]).ExciterModule.module_id)
check("app.modules.imaging",       lambda: __import__("app.modules.imaging", fromlist=["ImagerModule"]).ImagerModule.module_id)
check("app.modules.reverb",        lambda: __import__("app.modules.reverb", fromlist=["ReverbModule"]).ReverbModule.module_id)
check("app.chain (MasteringChain)",lambda: __import__("app.chain", fromlist=["MasteringChain"]).MasteringChain)
check("app.main (FastAPI app)",    lambda: __import__("app.main", fromlist=["app"]).app.title)

# ──────────────────────────────────────────────────────────────
section("3. Загрузка аудио")
# ──────────────────────────────────────────────────────────────
import numpy as np
import soundfile as sf
from app.pipeline import load_audio_from_bytes, measure_lufs

# Определяем источник сигнала
wav_path: Path | None = None
if len(sys.argv) >= 2:
    wav_path = Path(sys.argv[1]).resolve()
    if not wav_path.is_file():
        print(f"  {WARN}  Файл не найден: {wav_path}")
        wav_path = None

# Путь по умолчанию — ../doc/Pre-Master.wav относительно backend/
if wav_path is None:
    default_path = Path(__file__).resolve().parent.parent / "doc" / "Pre-Master.wav"
    if default_path.is_file():
        wav_path = default_path
        print(f"  {INFO}  Найден файл по умолчанию: {wav_path}")

if wav_path is not None:
    print(f"  {INFO}  Используется реальный файл: {wav_path}")

    def _load_wav():
        data = wav_path.read_bytes()
        audio, sr = load_audio_from_bytes(data, wav_path.name)
        return audio, sr

    ok = check(f"Чтение {wav_path.name}", _load_wav)
    if ok:
        audio, sr = _load_wav()
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        dur_sec = audio.shape[0] / sr
        peak_db = float(20 * np.log10(max(float(np.max(np.abs(audio))), 1e-12)))
        lufs_in = measure_lufs(audio, sr)
        print(f"  {INFO}  shape={audio.shape}  sr={sr}  ch={channels}  dur={dur_sec:.1f}s")
        print(f"  {INFO}  LUFS вход={lufs_in:.2f}  пик={peak_db:.2f} dBFS")
        stereo = audio if channels == 2 else np.column_stack([audio, audio]) if audio.ndim == 1 else np.column_stack([audio[:, 0], audio[:, 0]])
    else:
        wav_path = None

if wav_path is None:
    # Генерируем тестовый тон
    print(f"  {INFO}  Генерируется тестовый синус 440 Hz / 3 сек / 44100 Hz")
    sr = 44100
    dur_sec = 3.0
    t = np.linspace(0, dur_sec, int(sr * dur_sec), dtype=np.float32)
    sine = 0.35 * np.sin(2 * np.pi * 440 * t)
    stereo = np.column_stack([sine, sine])
    audio = stereo
    lufs_in = measure_lufs(stereo, sr)
    print(f"  {INFO}  LUFS тона={lufs_in:.2f}")

# ──────────────────────────────────────────────────────────────
section("4. Pipeline-функции (по одному шагу)")
# ──────────────────────────────────────────────────────────────
from app.pipeline import (
    remove_dc_offset,
    remove_intersample_peaks,
    apply_target_curve,
    apply_multiband_dynamics,
    apply_maximizer,
    apply_maximizer_transient_aware,
    apply_final_spectral_balance,
    apply_style_eq,
    apply_harmonic_exciter,
    apply_stereo_imager,
    apply_reverb,
    normalize_lufs,
    export_audio,
    compute_lufs_timeline,
    compute_spectrum_bars,
    compute_vectorscope_points,
    measure_stereo_correlation,
    run_mastering_pipeline,
)

check("remove_dc_offset",             lambda: remove_dc_offset(stereo).shape)
check("remove_intersample_peaks",     lambda: remove_intersample_peaks(stereo).shape)
check("apply_target_curve (minimum)", lambda: apply_target_curve(stereo, sr, phase_mode="minimum").shape)
check("apply_target_curve (linear)",  lambda: apply_target_curve(stereo, sr, phase_mode="linear_phase").shape)
check("apply_target_curve (M/S EQ)",  lambda: apply_target_curve(stereo, sr, eq_ms=True).shape)
check("apply_multiband_dynamics",     lambda: apply_multiband_dynamics(stereo, sr).shape)
check("apply_maximizer",              lambda: apply_maximizer(stereo).shape)
check("apply_maximizer_transient",    lambda: apply_maximizer_transient_aware(stereo, sr).shape)
check("apply_final_spectral_balance", lambda: apply_final_spectral_balance(stereo, sr).shape)
check("apply_style_eq standard",      lambda: apply_style_eq(stereo, sr, style="standard").shape)
check("apply_style_eq edm",           lambda: apply_style_eq(stereo, sr, style="edm").shape)
check("apply_harmonic_exciter 1×",    lambda: apply_harmonic_exciter(stereo, sr, exciter_db=0.8, mode="warm", oversample=1).shape)
check("apply_harmonic_exciter 2×",    lambda: apply_harmonic_exciter(stereo, sr, exciter_db=0.8, mode="warm", oversample=2).shape)
check("apply_harmonic_exciter tape",  lambda: apply_harmonic_exciter(stereo, sr, exciter_db=0.8, mode="tape").shape)
check("apply_stereo_imager width=1.2",lambda: apply_stereo_imager(stereo, width=1.2, sr=sr).shape)
check("apply_reverb plate",           lambda: apply_reverb(stereo, sr, reverb_type="plate", mix=0.15).shape)
check("apply_reverb hall",            lambda: apply_reverb(stereo, sr, reverb_type="hall", mix=0.15).shape)
check("apply_reverb M/S",             lambda: apply_reverb(stereo, sr, mix=0.1, mix_mid=0.05, mix_side=0.2).shape)
check("normalize_lufs → -14",         lambda: round(measure_lufs(normalize_lufs(stereo, sr, -14.0), sr), 2))
check("measure_lufs",                 lambda: round(measure_lufs(stereo, sr), 2))
check("measure_stereo_correlation",   lambda: round(measure_stereo_correlation(stereo) or 0.0, 4))
check("compute_spectrum_bars",        lambda: f"{len(compute_spectrum_bars(stereo, sr))} bars")
check("compute_lufs_timeline",        lambda: f"{len(compute_lufs_timeline(stereo, sr)[0])} points")
check("compute_vectorscope_points",   lambda: f"{len(compute_vectorscope_points(stereo))} points")
check("export WAV tpdf",              lambda: f"{len(export_audio(stereo, sr, 2, 'wav', dither_type='tpdf'))} bytes")
check("export WAV ns_e",              lambda: f"{len(export_audio(stereo, sr, 2, 'wav', dither_type='ns_e'))} bytes")
check("export WAV ns_itu",            lambda: f"{len(export_audio(stereo, sr, 2, 'wav', dither_type='ns_itu'))} bytes")
check("export FLAC",                  lambda: f"{len(export_audio(stereo, sr, 2, 'flac'))} bytes")

# ──────────────────────────────────────────────────────────────
section("5. Модули цепочки (BaseModule subclasses)")
# ──────────────────────────────────────────────────────────────
from app.modules.dc_offset import DCOffsetModule
from app.modules.peak_guard import PeakGuardModule
from app.modules.equalizer import TargetCurveModule, FinalSpectralBalanceModule, StyleEQModule
from app.modules.dynamics import DynamicsModule
from app.modules.maximizer import MaximizerModule
from app.modules.normalize_lufs import NormalizeLUFSModule
from app.modules.exciter import ExciterModule
from app.modules.imaging import ImagerModule
from app.modules.reverb import ReverbModule

check("DCOffsetModule",            lambda: DCOffsetModule().process(stereo, sr).shape)
check("PeakGuardModule",           lambda: PeakGuardModule().process(stereo, sr).shape)
check("TargetCurveModule",         lambda: TargetCurveModule().process(stereo, sr).shape)
check("TargetCurveModule M/S",     lambda: TargetCurveModule(eq_ms=True).process(stereo, sr).shape)
check("TargetCurveModule linear",  lambda: TargetCurveModule(phase_mode="linear_phase").process(stereo, sr).shape)
check("FinalSpectralBalance",      lambda: FinalSpectralBalanceModule().process(stereo, sr).shape)
check("StyleEQModule edm",         lambda: StyleEQModule(style="edm").process(stereo, sr).shape)
check("DynamicsModule",            lambda: DynamicsModule().process(stereo, sr).shape)
check("MaximizerModule",           lambda: MaximizerModule().process(stereo, sr).shape)
check("NormalizeLUFSModule -14",   lambda: round(measure_lufs(NormalizeLUFSModule(target_lufs=-14.0).process(stereo, sr), sr), 2))
check("ExciterModule warm 2×",     lambda: ExciterModule(exciter_db=0.8, mode="warm", oversample=2).process(stereo, sr).shape)
check("ImagerModule width=1.2",    lambda: ImagerModule(width=1.2).process(stereo, sr).shape)
check("ReverbModule disabled",     lambda: ReverbModule(enabled=False).process(stereo, sr).shape)
check("ReverbModule plate",        lambda: ReverbModule(enabled=True, mix=0.1).process(stereo, sr).shape)
check("ReverbModule M/S",          lambda: ReverbModule(enabled=True, mix_mid=0.05, mix_side=0.2).process(stereo, sr).shape)
check("Module amount=0 (bypass)",  lambda: np.allclose(DCOffsetModule(amount=0.0).process(stereo, sr), stereo))
check("Module amount=0.5 (blend)", lambda: DCOffsetModule(amount=0.5).process(stereo, sr).shape)

# ──────────────────────────────────────────────────────────────
section("6. MasteringChain (v2) — полная цепочка")
# ──────────────────────────────────────────────────────────────
from app.chain import MasteringChain

def _chain_standard():
    chain = MasteringChain.default_chain(target_lufs=-14.0, style="standard")
    out = chain.process(stereo, sr, target_lufs=-14.0, style="standard")
    lufs = measure_lufs(out, sr)
    assert abs(lufs - (-14.0)) <= 2.5, f"LUFS {lufs:.2f} далеко от цели -14.0"
    return f"LUFS={lufs:.2f}  shape={out.shape}"

def _chain_edm():
    chain = MasteringChain.default_chain(target_lufs=-9.0, style="edm")
    out = chain.process(stereo, sr, target_lufs=-9.0, style="edm")
    lufs = measure_lufs(out, sr)
    return f"LUFS={lufs:.2f}  shape={out.shape}"

def _chain_from_config():
    cfg = MasteringChain.default_config(target_lufs=-14.0, style="standard")
    chain = MasteringChain.from_config(cfg)
    out = chain.process(stereo, sr, target_lufs=-14.0)
    return f"shape={out.shape}  modules={len(chain.modules)}"

def _chain_custom():
    """Кастомная цепочка: только EQ + нормализация."""
    cfg = {
        "modules": [
            {"id": "dc_offset",     "enabled": True,  "amount": 1.0},
            {"id": "target_curve",  "enabled": True,  "phase_mode": "minimum", "eq_ms": False, "amount": 1.0},
            {"id": "normalize_lufs","enabled": True,  "target_lufs": -14.0, "amount": 1.0},
        ]
    }
    chain = MasteringChain.from_config(cfg)
    out = chain.process(stereo, sr, target_lufs=-14.0)
    return f"shape={out.shape}"

check("default_chain standard → LUFS -14", _chain_standard)
check("default_chain edm → LUFS -9",       _chain_edm)
check("chain from_config",                  _chain_from_config)
check("custom chain (3 modules)",           _chain_custom)
check("default_config структура",           lambda: f"{len(MasteringChain.default_config()['modules'])} модулей")

# ──────────────────────────────────────────────────────────────
section("7. Полный мастеринг реального файла (если передан)")
# ──────────────────────────────────────────────────────────────
if wav_path is not None:
    def _full_master():
        target = -14.0
        chain = MasteringChain.default_chain(target_lufs=target, style="standard")
        out = chain.process(audio, sr, target_lufs=target, style="standard")
        lufs_out = measure_lufs(out, sr)
        peak_out_db = float(20 * np.log10(max(float(np.max(np.abs(out))), 1e-12)))
        has_nan = bool(np.any(np.isnan(out)))
        assert not has_nan, "Обнаружены NaN в выходном сигнале!"
        assert abs(lufs_out - target) <= 2.5, f"LUFS {lufs_out:.2f} далеко от цели {target}"
        # Сохранить мастер для прослушивания
        out_dir = Path(__file__).resolve().parent.parent / "test_output"
        out_dir.mkdir(exist_ok=True)
        ch = 1 if out.ndim == 1 else out.shape[1]
        wav_bytes = export_audio(out, sr, ch, "wav", dither_type="tpdf")
        out_file = out_dir / "Pre-Master_mastered.wav"
        out_file.write_bytes(wav_bytes)
        return (
            f"LUFS: {lufs_in:.2f} → {lufs_out:.2f}  "
            f"пик: {peak_out_db:.2f} dBFS  "
            f"сохранён: {out_file}"
        )

    check("Полный мастеринг Pre-Master.wav → -14 LUFS", _full_master)

    # Анализ (extended)
    def _analyze():
        from app.pipeline import compute_lufs_timeline, compute_spectrum_bars, compute_vectorscope_points
        timeline, step = compute_lufs_timeline(audio, sr)
        bars = compute_spectrum_bars(audio, sr)
        vscope = compute_vectorscope_points(audio)
        corr = measure_stereo_correlation(audio)
        return (
            f"timeline={len(timeline)}pts  "
            f"spectrum={len(bars)}bars  "
            f"vectorscope={len(vscope)}pts  "
            f"correlation={corr:.4f}"
        )

    check("Extended анализ (timeline + spectrum + vectorscope)", _analyze)

else:
    print(f"\n  {INFO}  Секция пропущена: нет реального WAV-файла.")
    print(f"  {INFO}  Передайте путь: python run_debug.py ../doc/Pre-Master.wav")

# ──────────────────────────────────────────────────────────────
section("8. run_mastering_pipeline v1 (legacy)")
# ──────────────────────────────────────────────────────────────
def _v1():
    out = run_mastering_pipeline(stereo, sr, target_lufs=-14.0, style="standard")
    lufs = measure_lufs(out, sr)
    assert not np.any(np.isnan(out)), "NaN в выходе v1!"
    return f"LUFS={lufs:.2f}  shape={out.shape}"

check("run_mastering_pipeline (standard)", _v1)

def _v1_house():
    out = run_mastering_pipeline(stereo, sr, target_lufs=-10.0, style="house_basic")
    return f"LUFS={measure_lufs(out, sr):.2f}"

check("run_mastering_pipeline (house_basic)", _v1_house)

# ──────────────────────────────────────────────────────────────
section("Итог")
# ──────────────────────────────────────────────────────────────
print()
total = sum(1 for _ in errors)  # количество ошибок уже есть
if errors:
    print(f"\033[91m  Обнаружено ошибок: {len(errors)}\033[0m\n")
    for name, tb in errors:
        print(f"\033[91m  ─── {name} ───\033[0m")
        # Показываем только последние строки трейсбека
        lines = tb.strip().split("\n")
        for line in lines[-5:]:
            print(f"    {line}")
        print()
    sys.exit(1)
else:
    print(f"\033[92m  Все проверки пройдены успешно! Система готова к работе.\033[0m")
    sys.exit(0)
