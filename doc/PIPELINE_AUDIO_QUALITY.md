# Аудит пайплайна и качества звука

Обновлено: 2026-03

## 1. Порядок модулей

### run_mastering_pipeline (pipeline.py)

Линейный конвейер (v1):

1. **remove_dc_offset** — удаление DC-смещения по каналам  
2. **remove_intersample_peaks** — защита от пиков (headroom 0.5 dB)  
3. **apply_spectral_denoise** (опционально, при denoise_strength > 0)  
4. **apply_target_curve** — студийный EQ (Ozone 5 Equalizer)  
5. **apply_deesser** — De-esser 5–9 kHz  
6. **apply_dynamics** — многополосная динамика + максимайзер (Ozone 5 Dynamics/Maximizer)  
7. **apply_parallel_compression** (опционально, при parallel_mix > 0)  
8. **normalize_lufs** — нормализация по целевым LUFS  
9. **apply_final_spectral_balance** — финальная частотная коррекция  
10. **apply_reference_match** (опционально)  
11. **apply_style_eq** — жанровый EQ  
12. **apply_transient_designer** (опционально)  
13. **apply_harmonic_exciter** (опционально)  
14. **apply_stereo_imager** (опционально)  
15. **remove_intersample_peaks** — финальная защита пиков  
16. **clip** −1…+1, очистка NaN/Inf  

### MasteringChain (chain.py, v2)

Порядок из `default_config`: dc_offset → peak_guard → target_curve → dynamics → **normalize_lufs** → final_spectral_balance → style_eq → exciter → imager → reverb → peak_guard.  
Модуль **maximizer** в цепочке по умолчанию не добавлен; лимитирование входит в dynamics/peak_guard.

### Дизеринг

- Дизеринг применяется **один раз** — при экспорте в 16-bit WAV в `export_audio` → `_write_wav_16bit_dithered`.
- Типы: `tpdf` (по умолчанию), `ns_e`, `ns_itu`. Уровень шума TPDF — 1 LSB.
- Для FLAC/24-bit дизеринг не используется (запись 24-bit без редукции битности в этом месте).

## 2. Пресеты Denoiser

В `pipeline.py` заданы `DENOISE_PRESETS`:

- **light:** (strength=0.20, noise_percentile=22) — мягкое подавление, меньше риска «металлического» фона на вокале.  
- **medium:** (0.5, 15.0)  
- **aggressive:** (0.75, 10.0)  

В `apply_spectral_denoise` нижняя граница усиления Wiener `min_gain = 0.25`, чтобы не обнулять ячейки и не давать артефактов на тихих/классических записях.

## 3. Компрессор и лимитер

- **remove_intersample_peaks:** headroom_db=0.5, true-peak стиль.  
- **apply_dynamics:** многополосная компрессия (crossovers из стиля), лимитер с порогом из конфига.  
- В `_apply_limiter_numpy` порог по умолчанию −1.0 dB; клиппинг через `np.clip` после масштабирования.

Рекомендация: при появлении «качания» или щелчков проверить атаку/релиз в dynamics и пороги лимитера в конфиге стиля.

## 4. Тесты качества

В `tests/test_pipeline.py` добавлены:

- **test_run_mastering_pipeline_sine** — синус на входе; на выходе нет клиппинга (peak ≤ 1.0), LUFS в разумном диапазоне, нет NaN/Inf.  
- **test_run_mastering_pipeline_vocal_like** — короткий «вокало-подобный» сигнал (микс синусов 200–2k Hz + малый шум); пайплайн не падает, выход конечный и без NaN/Inf.  
- **test_export_audio_wav_no_nan** — экспорт WAV 16-bit; сэмплы в допустимых границах, без NaN/Inf.
