/* ═══════ DOM ═══════ */
const API         = '';
const wdeco       = document.getElementById('wdeco');
const drop        = document.getElementById('drop');
const fileInput   = document.getElementById('fileInput');
const fileInfo    = document.getElementById('fileInfo');
const fileName    = document.getElementById('fileName');
const fileMeta    = document.getElementById('fileMeta');
const btnReset    = document.getElementById('btnReset');
const playerEl    = document.getElementById('player');
const waveWrap    = document.getElementById('waveWrap');
const waveCanvas  = document.getElementById('waveCanvas');
const btnPP       = document.getElementById('btnPP');
const iconPlay    = document.getElementById('iconPlay');
const iconPause   = document.getElementById('iconPause');
const tElapsed    = document.getElementById('tElapsed');
const tTotal      = document.getElementById('tTotal');
const tScrub      = document.getElementById('tScrub');
const tFill       = document.getElementById('tFill');
const audioMeta   = document.getElementById('audioMeta');
const amDur       = document.getElementById('amDur');
const amSr        = document.getElementById('amSr');
const amCh        = document.getElementById('amCh');
const amPeak      = document.getElementById('amPeak');
const vu          = document.getElementById('vu');
const meterVal    = document.getElementById('meterVal');
const btnMeasure  = document.getElementById('btnMeasure');
const stMeasure   = document.getElementById('stMeasure');
const meterCorrelation = document.getElementById('meterCorrelation');
const lufsTimelineWrap   = document.getElementById('lufsTimelineWrap');
const lufsTimelineCanvas = document.getElementById('lufsTimelineCanvas');
const outFormat   = document.getElementById('outFormat');
const styleGrid     = document.getElementById('styleGrid');
const targetLufsInput= document.getElementById('targetLufsInput');
const abWrap        = document.getElementById('abWrap');
const btnABa        = document.getElementById('btnAB-a');
const btnABb        = document.getElementById('btnAB-b');
const btnMaster     = document.getElementById('btnMaster');
const btnMasterTxt  = document.getElementById('btnMasterTxt');

let selectedStyle   = 'standard';
let masteredBuffer  = null;  // decoded AudioBuffer of mastered file
let abMode          = 'a';   // 'a' = original, 'b' = mastered
const pipeline    = document.getElementById('pipeline');
const progWrap    = document.getElementById('progWrap');
const progFill    = document.getElementById('progFill');
const progPct     = document.getElementById('progPct');
const progMsg     = document.getElementById('progMsg');
const resultPanel = document.getElementById('resultPanel');
const rBefore     = document.getElementById('rBefore');
const rAfter      = document.getElementById('rAfter');
const rDelta      = document.getElementById('rDelta');
const stMaster    = document.getElementById('stMaster');
const dawCard     = document.getElementById('dawCard');
const dawRuler    = document.getElementById('dawRuler');
const dawCanvasA  = document.getElementById('dawCanvasA');
const dawCanvasB  = document.getElementById('dawCanvasB');
const dawStatLufsA= document.getElementById('dawStatLufsA');
const dawStatLufsB= document.getElementById('dawStatLufsB');
const abLufsA     = document.getElementById('abLufsA');
const abLufsB     = document.getElementById('abLufsB');
const spectrumCard   = document.getElementById('spectrumCard');
const spectrumCanvas = document.getElementById('spectrumCanvas');
const vectorscopeCard   = document.getElementById('vectorscopeCard');
const vectorscopeCanvas = document.getElementById('vectorscopeCanvas');
const chainModulesHead   = document.getElementById('chainModulesHead');
const chainModulesBody   = document.getElementById('chainModulesBody');
const chainModulesList   = document.getElementById('chainModulesList');
const chainModulesChevron= document.getElementById('chainModulesChevron');
const chainPresetsWrap   = document.getElementById('chainPresetsWrap');
const chainPresetName    = document.getElementById('chainPresetName');
const btnChainPresetSave = document.getElementById('btnChainPresetSave');
const chainPresetSelect = document.getElementById('chainPresetSelect');
const btnChainPresetLoad = document.getElementById('btnChainPresetLoad');
const btnChainPresetDelete = document.getElementById('btnChainPresetDelete');

let currentFile = null;
/** Текущий конфиг цепочки (из GET /api/v2/chain/default), для отправки в POST /api/v2/master при изменённом порядке */
let chainModulesConfig = null;
/** Данные для графика LUFS по времени (после расширенного замера) */
let lastLufsTimelineData = null;
/** Последний результат расширенного анализа (для экспорта отчёта P13) */
let lastAnalyzeReport = null;

/* ═══════ CANVAS RESIZE OBSERVER ═══════ */
const canvasRO = new ResizeObserver(() => {
  if (audioBuffer) drawWaveform(getCurrentFrac());
  if (dawCard.classList.contains('visible') && lastDawState) {
    showDawComparison(lastDawState.origBuf, lastDawState.masteredBuf, lastDawState.lufsA, lastDawState.lufsB);
  }
  if (spectrumCard && spectrumCard.classList.contains('visible') && audioBuffer) drawSpectrum();
  if (vectorscopeCard && vectorscopeCard.classList.contains('visible') && audioBuffer) drawVectorscope();
  if (lufsTimelineWrap && lufsTimelineWrap.classList.contains('visible') && lastLufsTimelineData) {
    drawLufsTimeline(lastLufsTimelineData.timeline, lastLufsTimelineData.stepSec, lastLufsTimelineData.durationSec);
  }
});
canvasRO.observe(waveCanvas);
if (dawCard) canvasRO.observe(dawCard);
if (spectrumCard) canvasRO.observe(spectrumCard);
if (vectorscopeCard) canvasRO.observe(vectorscopeCard);
if (lufsTimelineWrap) canvasRO.observe(lufsTimelineWrap);

function getCurrentFrac() {
  if (!audioBuffer) return 0;
  if (isPlaying) return Math.min(1, (getCtx().currentTime - startedAt) / audioBuffer.duration);
  return pauseOffset > 0 ? Math.min(1, pauseOffset / audioBuffer.duration) : 0;
}

/* ═══════ TOASTS ═══════ */
const toastWrap = document.getElementById('toastWrap');

/**
 * Преобразует техническое сообщение об ошибке в читаемый текст.
 * Для ошибок ffprobe/ffmpeg добавляет инструкцию по установке.
 */
/** Безопасно прочитать ответ как JSON; при не-JSON (например 500 HTML) вернуть текст и status. */
async function safeResponseJson(res) {
  const text = await res.text();
  if (!text.trim()) return {};
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new Error(text.slice(0, 200) || res.statusText || 'Ошибка сервера');
  }
}

function friendlyError(msg) {
  if (!msg) return msg;
  if (msg.includes('Internal Server Error') || msg.includes('Internal S')) {
    return 'Ошибка на сервере при обработке. Попробуйте отключить часть модулей «Дополнительная обработка» или другой файл.';
  }
  if (msg.includes('ffprobe') || msg.includes('ffmpeg') || msg.includes('ffmpeg')) {
    const base = msg.replace(/\[Errno \d+\][^.]*\.\s*/g, '').trim();
    return base + ' → <code style="font-family:\'JetBrains Mono\',monospace;font-size:.85em;background:rgba(255,255,255,0.08);padding:1px 5px;border-radius:4px">sudo apt-get install -y ffmpeg</code>';
  }
  return msg;
}

function toast(msg, type = 'inf', dur = 3000) {
  const icons = {
    ok:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    err: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    inf: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="8"/><line x1="12" y1="12" x2="12" y2="16"/></svg>',
  };
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = icons[type] + `<span>${msg}</span>`;
  toastWrap.appendChild(el);
  const remove = () => {
    el.classList.add('out');
    setTimeout(() => el.remove(), 260);
  };
  setTimeout(remove, dur);
  el.addEventListener('click', remove);
}

/* ═══════ PAGE-WIDE DRAG & DROP ═══════ */
const dropOverlay = document.getElementById('dropOverlay');
let dragDepth = 0;
document.addEventListener('dragenter', e => {
  e.preventDefault();
  dragDepth++;
  if (dragDepth === 1) dropOverlay.classList.add('visible');
});
document.addEventListener('dragleave', () => {
  dragDepth--;
  if (dragDepth <= 0) { dragDepth = 0; dropOverlay.classList.remove('visible'); }
});
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  e.preventDefault();
  dragDepth = 0;
  dropOverlay.classList.remove('visible');
  const f = e.dataTransfer?.files?.[0];
  if (f && /\.(wav|mp3|flac)$/i.test(f.name)) {
    setFile(f);
    toast('Файл загружен: ' + f.name, 'ok');
  } else if (f) {
    toast('Формат не поддерживается. Используйте WAV, MP3, FLAC', 'err');
  }
});

/* ═══════ HISTORY ═══════ */
const HIST_KEY  = 'mm_history_v1';
const HIST_MAX  = 8;
const histList  = document.getElementById('histList');
const histCount = document.getElementById('histCount');
const histBody  = document.getElementById('histBody');
const histHead  = document.getElementById('histHead');
const histChev  = document.getElementById('histChevron');
const histClear = document.getElementById('histClear');

function getHistory() {
  try { return JSON.parse(localStorage.getItem(HIST_KEY) || '[]'); }
  catch { return []; }
}
function saveToHistory(entry) {
  const h = getHistory();
  h.unshift(entry);
  if (h.length > HIST_MAX) h.length = HIST_MAX;
  try { localStorage.setItem(HIST_KEY, JSON.stringify(h)); } catch {}
  renderHistory();
}
function renderHistory() {
  const h = getHistory();
  histCount.textContent = h.length;
  if (!h.length) {
    histList.innerHTML = '<div class="hist-empty">Нет обработанных файлов</div>';
    return;
  }
  histList.innerHTML = h.map(e => {
    const delta = e.after != null && e.before != null
      ? (e.after - e.before).toFixed(1)
      : null;
    const sign = delta > 0 ? '+' : '';
    return `
    <div class="hist-entry">
      <div>
        <div class="he-name">${e.name}</div>
        <div class="he-meta">${e.fmt?.toUpperCase() || 'WAV'} · ${e.size || ''} · Цель ${e.target} LUFS</div>
      </div>
      <div>
        <div class="he-lufs">${e.after != null ? e.after.toFixed(1)+' LUFS' : '—'}${delta != null ? ' <span style="color:var(--text-dim);font-weight:400">('+sign+delta+' dB)</span>' : ''}</div>
        <div class="he-date">${e.date || ''}</div>
      </div>
    </div>`;
  }).join('');
}
function fmtDate(ts) {
  const d = new Date(ts);
  return d.toLocaleDateString('ru-RU', {day:'2-digit', month:'2-digit'}) + ' ' +
         d.toLocaleTimeString('ru-RU', {hour:'2-digit', minute:'2-digit'});
}

histHead.addEventListener('click', () => {
  const open = histBody.classList.toggle('open');
  histChev.classList.toggle('open', open);
});
histClear.addEventListener('click', e => {
  e.stopPropagation();
  try { localStorage.removeItem(HIST_KEY); } catch {}
  renderHistory();
  toast('История очищена', 'inf');
});

// Init
renderHistory();
if (chainModulesList) loadChainModules();

/* ═══════ Загрузка файла, переданного с лендинга через sessionStorage ═══════ */
(function() {
  try {
    const data = sessionStorage.getItem('mm_pending_file_data');
    const name = sessionStorage.getItem('mm_pending_file_name');
    const type = sessionStorage.getItem('mm_pending_file_type') || 'audio/wav';
    if (!data || !name) return;
    sessionStorage.removeItem('mm_pending_file_data');
    sessionStorage.removeItem('mm_pending_file_name');
    sessionStorage.removeItem('mm_pending_file_type');
    // data — base64 dataURL
    const byteStr = atob(data.split(',')[1]);
    const ab = new ArrayBuffer(byteStr.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteStr.length; i++) ia[i] = byteStr.charCodeAt(i);
    const file = new File([ab], name, { type });
    if (/\.(wav|mp3|flac)$/i.test(file.name)) {
      setFile(file);
      toast('Файл загружен с главной страницы: ' + file.name, 'ok');
    }
  } catch (e) {
    console.warn('sessionStorage file restore failed:', e);
  }
})();

/* ═══════ WEB AUDIO API ═══════ */
let audioCtx    = null;
let audioBuffer = null;   // decoded AudioBuffer
let srcNode     = null;   // current AudioBufferSourceNode
let isPlaying   = false;
let startedAt   = 0;      // audioCtx.currentTime when play started
let pauseOffset = 0;      // offset in seconds where we paused
let rafId       = null;

function getCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

/* ─── Decode and draw ─── */
async function loadAudio(file) {
  try {
    const ctx = getCtx();
    const ab  = await file.arrayBuffer();
    audioBuffer = await ctx.decodeAudioData(ab);
    pauseOffset = 0;
    // Wait two frames so the player element has rendered and canvas has a size
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    drawWaveform(0);
    updateTransport();
    // show meta row
    const dur = audioBuffer.duration;
    amDur.textContent  = fmtTime(dur);
    amSr.textContent   = (audioBuffer.sampleRate / 1000).toFixed(1) + ' kHz';
    amCh.textContent   = audioBuffer.numberOfChannels === 1 ? 'Mono' : 'Stereo';
    // peak dBFS from channel data
    let peak = 0;
    for (let c = 0; c < audioBuffer.numberOfChannels; c++) {
      const d = audioBuffer.getChannelData(c);
      for (let i = 0; i < d.length; i++) { const a = Math.abs(d[i]); if (a > peak) peak = a; }
    }
    amPeak.textContent = peak > 1e-9 ? (20 * Math.log10(peak)).toFixed(1) + ' dB' : '—';
    audioMeta.classList.add('visible');
    tTotal.textContent = fmtTime(dur);
    // Update file badge with duration
    fileMeta.textContent = fmtSize(file.size) + '  ·  ' + fmtTime(dur);
    if (spectrumCard) {
      spectrumCard.classList.add('visible');
      requestAnimationFrame(() => drawSpectrum());
    }
    if (vectorscopeCard) {
      vectorscopeCard.classList.add('visible');
      requestAnimationFrame(() => drawVectorscope());
    }
  } catch(e) {
    console.warn('Audio decode failed:', e);
  }
}

/* ─── Waveform drawing ─── */
function drawWaveform(playPosFrac) {
  if (!audioBuffer) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = waveCanvas.getBoundingClientRect();
  const W = Math.floor(rect.width  * dpr) || 460 * dpr;
  const H = Math.floor(rect.height * dpr) || 58  * dpr;
  if (waveCanvas.width !== W || waveCanvas.height !== H) {
    waveCanvas.width  = W;
    waveCanvas.height = H;
  }
  const ctx = waveCanvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  // Average channels
  const n  = audioBuffer.numberOfChannels;
  const len = audioBuffer.length;
  const spp = Math.max(1, Math.floor(len / W)); // samples per pixel
  const mid = H / 2;

  // Gradient: played = brighter accent, unplayed = dimmer
  const playedX = Math.floor(playPosFrac * W);

  for (let x = 0; x < W; x++) {
    const s0 = x * spp;
    let maxAmp = 0;
    for (let i = 0; i < spp; i++) {
      let sum = 0;
      for (let c = 0; c < n; c++) sum += Math.abs(audioBuffer.getChannelData(c)[s0 + i] || 0);
      const a = sum / n;
      if (a > maxAmp) maxAmp = a;
    }
    const h = Math.max(1, maxAmp * mid * 0.93);
    const played = x < playedX;
    ctx.fillStyle = played
      ? `rgba(108,75,255,${0.55 + maxAmp * 0.45})`
      : `rgba(108,75,255,${0.18 + maxAmp * 0.22})`;
    ctx.fillRect(x, mid - h, 1, h * 2);
  }

  // Playhead line
  if (playPosFrac > 0 && playPosFrac < 1) {
    const px = Math.floor(playPosFrac * W);
    ctx.fillStyle = 'rgba(255,255,255,0.75)';
    ctx.fillRect(px, 0, Math.max(1, dpr), H);
    // glow
    ctx.fillStyle = 'rgba(108,75,255,0.35)';
    ctx.fillRect(Math.max(0, px - 2 * dpr), 0, 4 * dpr, H);
  }
}

/* ─── Spectrum (FFT, log scale 20 Hz – 20 kHz) ─── */
const FFT_SIZE = 4096;
const SPECTRUM_BARS = 64;
const MIN_FREQ = 20;
const MAX_FREQ = 20000;

function hannWindow(n, N) {
  return 0.5 * (1 - Math.cos(2 * Math.PI * n / (N - 1)));
}

function fftRadix2(real, imag) {
  const N = real.length;
  if (N <= 1) return;
  const evenRe = [], evenIm = [], oddRe = [], oddIm = [];
  for (let i = 0; i < N / 2; i++) {
    evenRe[i] = real[i * 2]; evenIm[i] = imag[i * 2];
    oddRe[i]  = real[i * 2 + 1]; oddIm[i]  = imag[i * 2 + 1];
  }
  fftRadix2(evenRe, evenIm);
  fftRadix2(oddRe, oddIm);
  for (let k = 0; k < N / 2; k++) {
    const angle = -2 * Math.PI * k / N;
    const tRe = Math.cos(angle) * oddRe[k] - Math.sin(angle) * oddIm[k];
    const tIm = Math.cos(angle) * oddIm[k] + Math.sin(angle) * oddRe[k];
    real[k] = evenRe[k] + tRe; imag[k] = evenIm[k] + tIm;
    real[k + N / 2] = evenRe[k] - tRe; imag[k + N / 2] = evenIm[k] - tIm;
  }
}

function getSpectrumBars(buffer) {
  const sr = buffer.sampleRate;
  const numCh = buffer.numberOfChannels;
  const len = buffer.length;
  const start = len >= FFT_SIZE ? Math.max(0, Math.floor(len / 2) - Math.floor(FFT_SIZE / 2)) : 0;
  const re = new Float32Array(FFT_SIZE);
  const im = new Float32Array(FFT_SIZE);
  for (let i = 0; i < FFT_SIZE; i++) {
    const idx = start + i;
    let v = 0;
    if (idx < len) {
      for (let c = 0; c < numCh; c++) v += buffer.getChannelData(c)[idx] || 0;
      v /= numCh;
    }
    re[i] = v * hannWindow(i, FFT_SIZE);
    im[i] = 0;
  }
  fftRadix2(re, im);
  const numBins = FFT_SIZE / 2 + 1;
  const mag = new Float32Array(numBins);
  for (let k = 0; k < numBins; k++) {
    mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]) * (2 / FFT_SIZE);
  }
  const bars = [];
  const nyquist = sr / 2;
  for (let b = 0; b < SPECTRUM_BARS; b++) {
    const fLog = MIN_FREQ * Math.pow(MAX_FREQ / MIN_FREQ, b / (SPECTRUM_BARS - 1));
    const fNext = MIN_FREQ * Math.pow(MAX_FREQ / MIN_FREQ, (b + 1) / (SPECTRUM_BARS - 1));
    const k0 = Math.max(0, Math.floor((fLog / nyquist) * (FFT_SIZE / 2)));
    const k1 = Math.min(numBins - 1, Math.ceil((fNext / nyquist) * (FFT_SIZE / 2)));
    let maxMag = 0;
    for (let k = k0; k <= k1; k++) if (mag[k] > maxMag) maxMag = mag[k];
    const db = maxMag > 1e-12 ? 20 * Math.log10(maxMag) : -100;
    bars.push(db);
  }
  return bars;
}

function drawSpectrum(barsArg) {
  if (!spectrumCanvas || (!audioBuffer && !barsArg)) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = spectrumCanvas.getBoundingClientRect();
  const W = Math.floor(rect.width * dpr) || 400 * dpr;
  const H = Math.floor(rect.height * dpr) || 80 * dpr;
  if (spectrumCanvas.width !== W || spectrumCanvas.height !== H) {
    spectrumCanvas.width = W;
    spectrumCanvas.height = H;
  }
  const ctx = spectrumCanvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const bars = barsArg || getSpectrumBars(audioBuffer);
  const barW = W / bars.length;
  const refDb = 0;
  const minDb = -60;
  // Цвет полос зависит от активного режима (MID=пурпурный, SIDE=янтарный, MONO=циан)
  const mode = lastSpectrumData ? lastSpectrumData.active : 'mono';
  const colorMap = {
    mono:  ['rgba(34,211,238,0.35)',  'rgba(34,211,238,0.9)'],
    mid:   ['rgba(139,92,246,0.35)',  'rgba(139,92,246,0.9)'],
    side:  ['rgba(245,158,11,0.35)',  'rgba(245,158,11,0.9)'],
  };
  const [c0, c1] = colorMap[mode] || colorMap.mono;
  for (let i = 0; i < bars.length; i++) {
    const db = bars[i];
    const norm = Math.max(0, Math.min(1, (db - minDb) / (refDb - minDb)));
    const h = norm * H * 0.92;
    const x = i * barW;
    const grd = ctx.createLinearGradient(x, H, x, H - h);
    grd.addColorStop(0, c0);
    grd.addColorStop(1, c1);
    ctx.fillStyle = grd;
    ctx.fillRect(x, H - h, Math.max(1, barW - 1), h);
  }
}

/* ─── Vectorscope (Lissajous L vs R) ─── */
const VECTORSCOPE_MAX_POINTS = 6000;

function drawVectorscope() {
  if (!vectorscopeCanvas || !audioBuffer) return;
  const numCh = audioBuffer.numberOfChannels;
  const len = audioBuffer.length;
  const dpr = window.devicePixelRatio || 1;
  const rect = vectorscopeCanvas.getBoundingClientRect();
  const size = Math.min(rect.width, rect.height);
  const W = Math.floor(size * dpr) || 200 * dpr;
  const H = W;
  if (vectorscopeCanvas.width !== W || vectorscopeCanvas.height !== H) {
    vectorscopeCanvas.width = W;
    vectorscopeCanvas.height = H;
  }
  const ctx = vectorscopeCanvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const cx = W / 2;
  const cy = H / 2;
  const r = Math.min(cx, cy) * 0.92;
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, 2 * Math.PI);
  ctx.stroke();
  const ch0 = audioBuffer.getChannelData(0);
  const ch1 = numCh >= 2 ? audioBuffer.getChannelData(1) : ch0;
  const step = Math.max(1, Math.floor(len / VECTORSCOPE_MAX_POINTS));
  ctx.fillStyle = 'rgba(168,85,247,0.4)';
  for (let i = 0; i < len; i += step) {
    const L = ch0[i] || 0;
    const R = ch1[i] || 0;
    const x = cx + L * r;
    const y = cy - R * r;
    ctx.fillRect(Math.floor(x), Math.floor(y), 2, 2);
  }
}

/* ─── LUFS timeline (extended analyze) ─── */
const LUFS_TIMELINE_RANGE = { min: -50, max: 0 };

function drawLufsTimeline(timeline, stepSec, durationSec) {
  if (!lufsTimelineCanvas || !timeline || timeline.length === 0) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = lufsTimelineCanvas.getBoundingClientRect();
  const W = Math.floor((rect.width || 400) * dpr);
  const H = Math.floor(56 * dpr);
  if (lufsTimelineCanvas.width !== W || lufsTimelineCanvas.height !== H) {
    lufsTimelineCanvas.width = W;
    lufsTimelineCanvas.height = H;
  }
  const ctx = lufsTimelineCanvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const { min: lufsMin, max: lufsMax } = LUFS_TIMELINE_RANGE;
  const duration = durationSec > 0 ? durationSec : (timeline.length * (stepSec || 0.4));
  const pad = 2;
  const gw = W - pad * 2;
  const gh = H - pad * 2;
  const points = [];
  for (let i = 0; i < timeline.length; i++) {
    const v = timeline[i];
    if (v != null && !isNaN(v)) {
      const t = duration > 0 ? (i * (stepSec || 0.4)) / duration : i / Math.max(1, timeline.length - 1);
      const x = pad + t * gw;
      const norm = (v - lufsMin) / (lufsMax - lufsMin);
      const y = pad + (1 - Math.max(0, Math.min(1, norm))) * gh;
      points.push({ x, y, lufs: v });
    }
  }
  if (points.length < 2) return;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  const grd = ctx.createLinearGradient(0, 0, 0, H);
  grd.addColorStop(0, 'rgba(34,211,238,0.25)');
  grd.addColorStop(1, 'rgba(108,75,255,0.06)');
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.lineTo(points[points.length - 1].x, H - pad);
  ctx.lineTo(points[0].x, H - pad);
  ctx.closePath();
  ctx.fillStyle = grd;
  ctx.fill();
  ctx.strokeStyle = 'rgba(108,75,255,0.9)';
  ctx.lineWidth = Math.max(1, dpr * 1.5);
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.stroke();
}

/* ─── DAW comparison (ruler, waveforms, stats) ─── */
let lastDawState = null;

function dawDrawWaveform(ctx, buffer, w, h, rgb) {
  if (!buffer || buffer.length === 0) return;
  const [r, g, b] = rgb;
  const n = buffer.numberOfChannels;
  const len = buffer.length;
  const spp = Math.max(1, Math.floor(len / w));
  const mid = h / 2;
  for (let x = 0; x < w; x++) {
    const s0 = x * spp;
    let maxAmp = 0;
    for (let i = 0; i < spp; i++) {
      let sum = 0;
      for (let c = 0; c < n; c++) sum += Math.abs(buffer.getChannelData(c)[s0 + i] || 0);
      maxAmp = Math.max(maxAmp, sum / n);
    }
    const barH = Math.max(1, maxAmp * mid * 0.92);
    ctx.fillStyle = `rgba(${r},${g},${b},${0.2 + maxAmp * 0.6})`;
    ctx.fillRect(x, mid - barH, 1, barH * 2);
  }
}

function dawDrawRuler(canvas, duration) {
  if (!duration || duration <= 0) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.textBaseline = 'top';
  const step = duration <= 30 ? 5 : duration <= 120 ? 15 : 30;
  for (let t = 0; t <= duration; t += step) {
    const x = (t / duration) * w;
    const m = Math.floor(t / 60), s = Math.floor(t % 60);
    const label = m + ':' + String(s).padStart(2, '0');
    ctx.fillText(label, x + 3, 2);
  }
}

function dawComputeStats(lufsA, lufsB) {
  dawStatLufsA.textContent = lufsA != null ? lufsA.toFixed(1) + ' LUFS' : '—';
  dawStatLufsB.textContent = lufsB != null ? lufsB.toFixed(1) + ' LUFS' : '—';
}

function showDawComparison(origBuf, masteredBuf, lufsA, lufsB) {
  if (!origBuf || !masteredBuf) return;
  lastDawState = { origBuf, masteredBuf, lufsA, lufsB };
  const wrap = dawRuler.parentElement;
  const width = wrap.getBoundingClientRect().width;
  const dpr = window.devicePixelRatio || 1;
  const W = Math.floor(width * dpr) || 400;
  const trackH = 52;
  const rulerH = 20;
  dawRuler.width = W;
  dawRuler.height = Math.floor(rulerH * dpr);
  dawCanvasA.width = W;
  dawCanvasA.height = Math.floor(trackH * dpr);
  dawCanvasB.width = W;
  dawCanvasB.height = Math.floor(trackH * dpr);

  const duration = Math.max(origBuf.duration, masteredBuf.duration);
  dawDrawRuler(dawRuler, duration);

  const ctxA = dawCanvasA.getContext('2d');
  ctxA.clearRect(0, 0, W, Math.floor(trackH * dpr));
  dawDrawWaveform(ctxA, origBuf, W, Math.floor(trackH * dpr), [108, 75, 255]);

  const ctxB = dawCanvasB.getContext('2d');
  ctxB.clearRect(0, 0, W, Math.floor(trackH * dpr));
  dawDrawWaveform(ctxB, masteredBuf, W, Math.floor(trackH * dpr), [52, 211, 153]);

  dawComputeStats(lufsA, lufsB);
  dawCard.classList.add('visible');
}

/* ─── Time formatting ─── */
function fmtTime(sec) {
  const s = Math.floor(sec);
  return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
}

/* ─── Playback ─── */
function playFrom(offset) {
  if (!audioBuffer) return;
  const ctx = getCtx();
  if (ctx.state === 'suspended') ctx.resume();
  if (srcNode) { try { srcNode.stop(); } catch(e){} srcNode = null; }
  srcNode = ctx.createBufferSource();
  srcNode.buffer = audioBuffer;
  srcNode.connect(ctx.destination);
  srcNode.start(0, offset);
  startedAt   = ctx.currentTime - offset;
  isPlaying   = true;
  srcNode.onended = () => { if (isPlaying) stopAll(); };
  startRaf();
  updateTransport();
}

function pauseAudio() {
  if (!isPlaying) return;
  pauseOffset = getCtx().currentTime - startedAt;
  if (srcNode) { try { srcNode.stop(); } catch(e){} srcNode = null; }
  isPlaying = false;
  stopRaf();
  updateTransport();
}

function stopAll() {
  pauseOffset = 0; isPlaying = false;
  if (srcNode) { try { srcNode.stop(); } catch(e){} srcNode = null; }
  stopRaf();
  drawWaveform(0);
  tElapsed.textContent = '0:00';
  tFill.style.width = '0%';
  updateTransport();
}

function togglePlay() {
  if (!audioBuffer) return;
  if (isPlaying) pauseAudio(); else playFrom(pauseOffset);
}

function updateTransport() {
  iconPlay.style.display  = isPlaying ? 'none'  : '';
  iconPause.style.display = isPlaying ? '' : 'none';
}

/* ─── RAF loop ─── */
function startRaf() {
  stopRaf();
  function loop() {
    if (!isPlaying || !audioBuffer) return;
    const pos = (getCtx().currentTime - startedAt);
    const dur = audioBuffer.duration;
    const frac = Math.min(1, pos / dur);
    drawWaveform(frac);
    tElapsed.textContent = fmtTime(Math.min(pos, dur));
    tFill.style.width = (frac * 100).toFixed(2) + '%';
    waveWrap.setAttribute('data-time', fmtTime(Math.min(pos, dur)));
    rafId = requestAnimationFrame(loop);
  }
  rafId = requestAnimationFrame(loop);
}
function stopRaf() { if (rafId) { cancelAnimationFrame(rafId); rafId = null; } }

/* ─── Click on waveform to seek ─── */
waveWrap.addEventListener('click', e => {
  if (!audioBuffer) return;
  const rect = waveWrap.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  const newOffset = frac * audioBuffer.duration;
  pauseOffset = newOffset;
  if (isPlaying) playFrom(newOffset);
  else { drawWaveform(frac); tElapsed.textContent = fmtTime(newOffset); tFill.style.width = (frac*100)+'%'; }
});

/* ─── Click on scrub bar ─── */
tScrub.addEventListener('click', e => {
  if (!audioBuffer) return;
  const rect = tScrub.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  const newOffset = frac * audioBuffer.duration;
  pauseOffset = newOffset;
  if (isPlaying) playFrom(newOffset);
  else { drawWaveform(frac); tElapsed.textContent = fmtTime(newOffset); tFill.style.width = (frac*100)+'%'; }
});

btnPP.addEventListener('click', e => { e.stopPropagation(); togglePlay(); });

/* ─── Keyboard shortcuts ─── */
document.addEventListener('keydown', e => {
  // Don't fire when typing in inputs
  if (['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)) return;
  if (e.code === 'Space') { e.preventDefault(); if (audioBuffer) togglePlay(); }
  if (e.code === 'Enter' && !btnMaster.disabled) { e.preventDefault(); btnMaster.click(); }
});

/* ═══════ VU Bar (28 segs) ═══════ */
const VU_N = 28;
for (let i=0;i<VU_N;i++) {
  const s = document.createElement('div');
  s.className = 'vu-s';
  vu.appendChild(s);
}
function setVu(lufs) {
  const segs = vu.children;
  const pct = lufs==null ? 0 : Math.min(1, Math.max(0,(lufs+60)/60));
  const lit = Math.round(pct*VU_N);
  Array.from(segs).forEach((s,i)=>{
    s.classList.remove('g','a','r');
    const pos=(i+1)/VU_N;
    if(i<lit){
      if(pos<.72)      s.classList.add('g');
      else if(pos<.88) s.classList.add('a');
      else             s.classList.add('r');
    }
  });
}
function setMeter(lufs) {
  if(lufs==null||isNaN(lufs)){
    meterVal.textContent='— LUFS'; meterVal.className='meter-val empty'; setVu(null);
    if (meterCorrelation) { meterCorrelation.classList.remove('visible'); meterCorrelation.textContent = ''; }
    if (lufsTimelineWrap) lufsTimelineWrap.classList.remove('visible');
    lastLufsTimelineData = null;
    lastAnalyzeReport = null;
    const reportActions = document.getElementById('reportActions');
    if (reportActions) reportActions.style.display = 'none';
    return;
  }
  meterVal.textContent = lufs.toFixed(1)+' LUFS';
  meterVal.className = 'meter-val '+(lufs>-9?'hot':lufs>-14?'warn':'good');
  setVu(lufs);
}
function setCorrelation(corr) {
  if (!meterCorrelation) return;
  if (corr == null || typeof corr !== 'number') {
    meterCorrelation.classList.remove('visible');
    meterCorrelation.textContent = '';
    return;
  }
  meterCorrelation.textContent = 'Корреляция L/R: ' + corr.toFixed(2) + ' (−1…+1)';
  meterCorrelation.classList.add('visible');
}

/* ═══════ Status ═══════ */
function setStatus(el,txt,type){
  el.textContent=txt; el.className='status'+(type?' '+type:'');
}

/* ═══════ Progress ═══════ */
function setProgress(pct,msg){
  progFill.style.width=pct+'%';
  progPct.textContent=Math.round(pct);
  progMsg.textContent=msg||'Обработка…';
  updatePipelineSteps(pct,msg);
}

/* ═══════ Pipeline step visualiser ═══════ */
const PIPE_MAP = [
  {step:'dc',      from:5,  to:20,  keys:['dc','смещени']},
  {step:'eq',      from:20, to:35,  keys:['eq','частот','equalizer']},
  {step:'dyn',     from:35, to:70,  keys:['динамик','максимайз','полос','dynamics']},
  {step:'lufs',    from:70, to:78,  keys:['lufs','нормализ']},
  {step:'final',   from:78, to:88,  keys:['финальн','спектр','жанр']},
  {step:'exciter', from:88, to:91,  keys:['эксайтер','exciter']},
  {step:'imager',  from:91, to:95,  keys:['стерео','imager','расширен']},
];
function updatePipelineSteps(pct,msg){
  const msgL=(msg||'').toLowerCase();
  PIPE_MAP.forEach(({step,from,to,keys})=>{
    const el=pipeline.querySelector(`[data-step="${step}"]`);
    if(!el) return;
    const matchMsg = keys.some(k=>msgL.includes(k));
    const inRange  = pct>=from && pct<to;
    const done     = pct>=to;
    el.classList.toggle('active', (inRange||matchMsg) && !done);
    el.classList.toggle('done',   done);
  });
}
function resetPipelineSteps(){
  pipeline.querySelectorAll('.pipe-step').forEach(el=>{
    el.classList.remove('active','done');
  });
}
function updateOzoneSteps(style){
  const isHouse = style === 'house_basic';
  const exciterEl = document.getElementById('pipeExciter');
  const imagerEl  = document.getElementById('pipeImager');
  if(exciterEl) exciterEl.style.display = isHouse ? '' : 'none';
  if(imagerEl)  imagerEl.style.display  = isHouse ? '' : 'none';
}

/* ═══════ Dropzone ═══════ */
drop.addEventListener('click', ()=>fileInput.click());
drop.addEventListener('dragover', e=>{e.preventDefault(); drop.classList.add('over')});
drop.addEventListener('dragleave', ()=>drop.classList.remove('over'));
drop.addEventListener('drop', e=>{
  e.preventDefault(); drop.classList.remove('over');
  const f=e.dataTransfer.files[0];
  if(f&&/\.(wav|mp3|flac)$/i.test(f.name)){ fileInput.files=e.dataTransfer.files; setFile(f); }
});
fileInput.addEventListener('change',()=>{ if(fileInput.files[0]) setFile(fileInput.files[0]); });

btnReset.addEventListener('click', e=>{
  e.stopPropagation();
  resetAll();
});

function fmtSize(bytes){
  if(bytes<1024*1024) return (bytes/1024).toFixed(0)+' KB';
  return (bytes/1024/1024).toFixed(1)+' MB';
}

function setFile(f){
  currentFile=f;
  fileName.textContent=f.name;
  fileMeta.textContent=fmtSize(f.size);
  fileInfo.classList.add('visible');
  drop.classList.add('has-file');
  btnMeasure.disabled=false;
  btnMaster.disabled=false;
  setMeter(null);
  setStatus(stMeasure,'');
  setStatus(stMaster,'');
  progWrap.classList.remove('on');
  pipeline.classList.remove('visible');
  resultPanel.classList.remove('visible');
  if (abLufsA) abLufsA.textContent = '—';
  if (abLufsB) abLufsB.textContent = '—';
  resetPipelineSteps();
  updateOzoneSteps(selectedStyle);
  // reset audio state
  stopAll();
  audioBuffer=null;
  masteredBuffer=null;
  abMode='a';
  abWrap.classList.remove('visible');
  dawCard.classList.remove('visible');
  if (spectrumCard) spectrumCard.classList.remove('visible');
  if (vectorscopeCard) vectorscopeCard.classList.remove('visible');
  lastDawState = null;
  btnABa.classList.add('active'); btnABb.classList.remove('active');
  audioMeta.classList.remove('visible');
  tTotal.textContent='0:00';
  playerEl.classList.add('visible');
  // decode async
  loadAudio(f);
}

function resetAll(){
  currentFile=null;
  fileInput.value='';
  fileName.textContent='';
  fileMeta.textContent='';
  fileInfo.classList.remove('visible');
  drop.classList.remove('has-file');
  btnMeasure.disabled=true;
  btnMaster.disabled=true;
  setMeter(null);
  setStatus(stMeasure,'');
  setStatus(stMaster,'');
  progWrap.classList.remove('on');
  pipeline.classList.remove('visible');
  resultPanel.classList.remove('visible');
  if (abLufsA) abLufsA.textContent = '—';
  if (abLufsB) abLufsB.textContent = '—';
  resetPipelineSteps();
  updateOzoneSteps(selectedStyle);
  // reset player
  stopAll();
  audioBuffer=null;
  masteredBuffer=null;
  abMode='a';
  abWrap.classList.remove('visible');
  dawCard.classList.remove('visible');
  if (spectrumCard) spectrumCard.classList.remove('visible');
  if (vectorscopeCard) vectorscopeCard.classList.remove('visible');
  lastDawState = null;
  btnABa.classList.add('active'); btnABb.classList.remove('active');
  playerEl.classList.remove('visible');
  audioMeta.classList.remove('visible');
  // clear canvas
  const ctx2=waveCanvas.getContext('2d');
  ctx2.clearRect(0,0,waveCanvas.width,waveCanvas.height);
}

/* ═══════ Style cards ═══════ */
styleGrid.addEventListener('click', e => {
  const card = e.target.closest('.style-card');
  if (!card) return;
  if (card.classList.contains('locked')) return; // заблокированная — игнорируем
  styleGrid.querySelectorAll('.style-card').forEach(c => c.classList.remove('active'));
  card.classList.add('active');
  selectedStyle = card.dataset.style;
  targetLufsInput.value = card.dataset.lufs;
  updateOzoneSteps(selectedStyle);
  loadChainModules();
});

if (chainModulesHead && chainModulesBody && chainModulesChevron) {
  chainModulesHead.addEventListener('click', () => {
    chainModulesBody.classList.toggle('open');
    chainModulesChevron.classList.toggle('open', chainModulesBody.classList.contains('open'));
  });
}
if (targetLufsInput && chainModulesList) {
  targetLufsInput.addEventListener('change', () => loadChainModules());
}

function renderChainModulesList(modules) {
  if (!chainModulesList) return;
  const mods = modules || [];
  chainModulesList.innerHTML = mods.map((m, i) => {
    const amt = Math.round((m.amount != null ? m.amount : 1) * 100);
    const phaseMode = (m.phase_mode || 'minimum').toLowerCase();
    const isTargetCurve = m.id === 'target_curve';
    const eqMsChecked = !!m.eq_ms;
    const eqMsCheckbox = isTargetCurve
      ? `<label class="cm-eq-ms-wrap" title="Применять кривую отдельно к Mid и Side"><input type="checkbox" class="cm-eq-ms" data-index="${i}" ${eqMsChecked ? 'checked' : ''}> M/S</label>`
      : '';
    const phaseSelect = isTargetCurve
      ? `<select class="cm-phase-mode" data-index="${i}" title="Режим фазы EQ">
          <option value="minimum" ${phaseMode === 'linear_phase' ? '' : 'selected'}>Min. фаза</option>
          <option value="linear_phase" ${phaseMode === 'linear_phase' ? 'selected' : ''}>Linear phase</option>
        </select>`
      : '';
    const reverbType = (m.reverb_type || 'plate').toLowerCase();
    const isReverb = m.id === 'reverb';
    const mixMidPct = m.mix_mid != null ? Math.round(m.mix_mid * 100) : '';
    const mixSidePct = m.mix_side != null ? Math.round(m.mix_side * 100) : '';
    const reverbSelect = isReverb
      ? `<select class="cm-reverb-type" data-index="${i}" title="Тип ревербератора">
          <option value="plate" ${reverbType === 'plate' ? 'selected' : ''}>Plate</option>
          <option value="room" ${reverbType === 'room' ? 'selected' : ''}>Room</option>
          <option value="hall" ${reverbType === 'hall' ? 'selected' : ''}>Hall</option>
          <option value="theater" ${reverbType === 'theater' ? 'selected' : ''}>Theater</option>
          <option value="cathedral" ${reverbType === 'cathedral' ? 'selected' : ''}>Cathedral</option>
        </select>
        <label class="cm-ms-wrap" title="Mix Mid (M/S)">M:<input type="number" class="cm-reverb-mid" data-index="${i}" min="0" max="100" step="5" value="${mixMidPct}" placeholder="—"></label>
        <label class="cm-ms-wrap" title="Mix Side (M/S)">S:<input type="number" class="cm-reverb-side" data-index="${i}" min="0" max="100" step="5" value="${mixSidePct}" placeholder="—"></label>`
      : '';
    const osVal = Math.max(1, Math.min(4, parseInt(m.oversample, 10) || 1));
    const isExciter = m.id === 'exciter';
    const oversampleSelect = isExciter
      ? `<select class="cm-oversample" data-index="${i}" title="Передискретизация (антиалиас)">
          <option value="1" ${osVal === 1 ? 'selected' : ''}>1×</option>
          <option value="2" ${osVal === 2 ? 'selected' : ''}>2×</option>
          <option value="4" ${osVal === 4 ? 'selected' : ''}>4×</option>
        </select>`
      : '';
    return `<div class="chain-mod-item ${m.enabled ? 'enabled' : ''}" draggable="true" data-index="${i}">
      <span class="cm-num">${i + 1}</span>
      <span class="cm-label">${m.label || m.id}</span>
      <label class="cm-amount-wrap">
        <input type="range" class="cm-amount" min="0" max="100" value="${amt}" data-index="${i}" title="Amount">
        <span class="cm-amount-val">${amt}%</span>
      </label>
      ${phaseSelect}
      ${eqMsCheckbox}
      ${reverbSelect}
      ${oversampleSelect}
    </div>`;
  }).join('');
  chainModulesList.querySelectorAll('.chain-mod-item').forEach((el, i) => {
    el.addEventListener('dragstart', (e) => {
      if (e.target.classList.contains('cm-amount') || e.target.closest('.cm-amount-wrap') || e.target.classList.contains('cm-phase-mode') || e.target.classList.contains('cm-eq-ms') || e.target.closest('.cm-eq-ms-wrap') || e.target.classList.contains('cm-reverb-type') || e.target.classList.contains('cm-oversample') || e.target.classList.contains('cm-reverb-mid') || e.target.classList.contains('cm-reverb-side')) return;
      e.dataTransfer.setData('text/plain', String(i));
      e.dataTransfer.effectAllowed = 'move';
    });
    el.addEventListener('dragover', (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; el.classList.add('chain-mod-drag-over'); });
    el.addEventListener('dragleave', () => el.classList.remove('chain-mod-drag-over'));
    el.addEventListener('drop', (e) => {
      e.preventDefault();
      el.classList.remove('chain-mod-drag-over');
      const from = parseInt(e.dataTransfer.getData('text/plain'), 10);
      if (!chainModulesConfig?.modules || from === i) return;
      const arr = [...chainModulesConfig.modules];
      const [moved] = arr.splice(from, 1);
      arr.splice(i, 0, moved);
      chainModulesConfig.modules = arr;
      renderChainModulesList(arr);
    });
  });
  chainModulesList.querySelectorAll('.cm-amount').forEach((input) => {
    input.addEventListener('input', () => {
      const idx = parseInt(input.dataset.index, 10);
      const val = parseInt(input.value, 10) / 100;
      if (chainModulesConfig?.modules && chainModulesConfig.modules[idx] != null) {
        chainModulesConfig.modules[idx].amount = val;
        const v = input.closest('.chain-mod-item').querySelector('.cm-amount-val');
        if (v) v.textContent = input.value + '%';
      }
    });
  });
  chainModulesList.querySelectorAll('.cm-phase-mode').forEach((sel) => {
    sel.addEventListener('change', () => {
      const idx = parseInt(sel.dataset.index, 10);
      if (chainModulesConfig?.modules && chainModulesConfig.modules[idx] != null) {
        chainModulesConfig.modules[idx].phase_mode = sel.value;
      }
    });
  });
  chainModulesList.querySelectorAll('.cm-eq-ms').forEach((cb) => {
    cb.addEventListener('change', () => {
      const idx = parseInt(cb.dataset.index, 10);
      if (chainModulesConfig?.modules && chainModulesConfig.modules[idx] != null) {
        chainModulesConfig.modules[idx].eq_ms = cb.checked;
      }
    });
  });
  chainModulesList.querySelectorAll('.cm-reverb-type').forEach((sel) => {
    sel.addEventListener('change', () => {
      const idx = parseInt(sel.dataset.index, 10);
      if (chainModulesConfig?.modules && chainModulesConfig.modules[idx] != null) {
        chainModulesConfig.modules[idx].reverb_type = sel.value;
      }
    });
  });
  chainModulesList.querySelectorAll('.cm-reverb-mid').forEach((inp) => {
    inp.addEventListener('change', () => {
      const idx = parseInt(inp.dataset.index, 10);
      if (!chainModulesConfig?.modules || chainModulesConfig.modules[idx] == null) return;
      const v = inp.value.trim();
      if (v === '') delete chainModulesConfig.modules[idx].mix_mid;
      else { const pct = parseInt(v, 10); chainModulesConfig.modules[idx].mix_mid = isNaN(pct) ? undefined : Math.max(0, Math.min(100, pct)) / 100; }
    });
  });
  chainModulesList.querySelectorAll('.cm-reverb-side').forEach((inp) => {
    inp.addEventListener('change', () => {
      const idx = parseInt(inp.dataset.index, 10);
      if (!chainModulesConfig?.modules || chainModulesConfig.modules[idx] == null) return;
      const v = inp.value.trim();
      if (v === '') delete chainModulesConfig.modules[idx].mix_side;
      else { const pct = parseInt(v, 10); chainModulesConfig.modules[idx].mix_side = isNaN(pct) ? undefined : Math.max(0, Math.min(100, pct)) / 100; }
    });
  });
  chainModulesList.querySelectorAll('.cm-oversample').forEach((sel) => {
    sel.addEventListener('change', () => {
      const idx = parseInt(sel.dataset.index, 10);
      if (chainModulesConfig?.modules && chainModulesConfig.modules[idx] != null) {
        chainModulesConfig.modules[idx].oversample = parseInt(sel.value, 10) || 1;
      }
    });
  });
}

async function loadChainModules() {
  if (!chainModulesList) return;
  const style = selectedStyle || 'standard';
  const targetLufs = parseFloat(targetLufsInput?.value) || -14;
  try {
    const r = await fetch(API + `/api/v2/chain/default?style=${encodeURIComponent(style)}&target_lufs=${targetLufs}`);
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    chainModulesConfig = data;
    renderChainModulesList(data.modules || []);
  } catch (e) {
    chainModulesConfig = null;
    chainModulesList.innerHTML = `<div class="chain-mod-item"><span class="cm-label">Не удалось загрузить цепочку</span></div>`;
  }
}

/* ═══════ P10: сохранённые пресеты цепочки (только для залогиненных) ═══════ */
async function loadSavedPresetsList() {
  if (!chainPresetSelect) return;
  try {
    const r = await fetch(API + '/api/auth/presets', { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    const list = data.presets || [];
    chainPresetSelect.innerHTML = '<option value="">— Загрузить пресет —</option>' +
      list.map(p => `<option value="${p.id}">${escapeHtml(p.name || 'Без имени')}</option>`).join('');
  } catch (e) { /* ignore */ }
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

if (btnChainPresetSave && chainPresetName) {
  btnChainPresetSave.addEventListener('click', async () => {
    if (!isLoggedIn()) return;
    const name = (chainPresetName.value || '').trim();
    if (!name) { toast('Введите имя пресета', 'warn'); return; }
    if (!chainModulesConfig || !chainModulesConfig.modules) { toast('Нет цепочки для сохранения', 'warn'); return; }
    try {
      const r = await fetch(API + '/api/auth/presets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({
          name,
          config: { modules: chainModulesConfig.modules },
          style: selectedStyle || 'standard',
          target_lufs: parseFloat(targetLufsInput?.value) || -14,
        }),
      });
      if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.detail || r.statusText); }
      chainPresetName.value = '';
      loadSavedPresetsList();
      toast('Пресет сохранён', 'inf');
    } catch (e) {
      toast(e.message || 'Ошибка сохранения', 'err');
    }
  });
}

if (btnChainPresetLoad && chainPresetSelect) {
  btnChainPresetLoad.addEventListener('click', async () => {
    if (!isLoggedIn()) return;
    const id = chainPresetSelect.value;
    if (!id) { toast('Выберите пресет', 'warn'); return; }
    try {
      const r = await fetch(API + '/api/auth/presets/' + id, { headers: authHeaders() });
      if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.detail || r.statusText); }
      const p = await r.json();
      const modules = (p.config && p.config.modules) ? p.config.modules : (p.config || []);
      if (!Array.isArray(modules) || modules.length === 0) { toast('Пресет пустой', 'warn'); return; }
      chainModulesConfig = { modules };
      if (p.style) {
        selectedStyle = p.style;
        const styleCard = document.querySelector('.style-card[data-style="' + p.style + '"]');
        if (styleCard && !styleCard.classList.contains('locked')) {
          styleGrid.querySelectorAll('.style-card').forEach(c => c.classList.remove('active'));
          styleCard.classList.add('active');
        }
        updateOzoneSteps(selectedStyle);
      }
      if (p.target_lufs != null && targetLufsInput) targetLufsInput.value = String(p.target_lufs);
      renderChainModulesList(modules);
      toast('Пресет загружен', 'inf');
    } catch (e) {
      toast(e.message || 'Ошибка загрузки пресета', 'err');
    }
  });
}

if (btnChainPresetDelete && chainPresetSelect) {
  btnChainPresetDelete.addEventListener('click', async () => {
    if (!isLoggedIn()) return;
    const id = chainPresetSelect.value;
    if (!id) { toast('Выберите пресет для удаления', 'warn'); return; }
    try {
      const r = await fetch(API + '/api/auth/presets/' + id, { method: 'DELETE', headers: authHeaders() });
      if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.detail || r.statusText); }
      loadSavedPresetsList();
      toast('Пресет удалён', 'inf');
    } catch (e) {
      toast(e.message || 'Ошибка удаления', 'err');
    }
  });
}

/* ═══════ A/B comparison ═══════ */
function setAB(mode) {
  abMode = mode;
  btnABa.classList.toggle('active', mode === 'a');
  btnABb.classList.toggle('active', mode === 'b');
  const buf = mode === 'b' && masteredBuffer ? masteredBuffer : audioBuffer;
  if (!buf) return;
  const wasPlaying = isPlaying;
  const pos = wasPlaying ? (getCtx().currentTime - startedAt) : pauseOffset;
  stopAll();
  // swap buffer reference temporarily for drawing
  const orig = audioBuffer;
  audioBuffer = buf;
  drawWaveform(buf.duration > 0 ? Math.min(pos / buf.duration, 1) : 0);
  pauseOffset = Math.min(pos, buf.duration);
  if (wasPlaying) playFrom(pauseOffset);
  else audioBuffer = orig; // restore; playFrom will use correct buf
  // keep the swapped buffer active for playback
  audioBuffer = buf;
  // обновить спектр и векторскоп под выбранный буфер (A — исходник, B — мастер)
  requestAnimationFrame(() => {
    if (spectrumCard && spectrumCard.classList.contains('visible')) drawSpectrum();
    if (vectorscopeCard && vectorscopeCard.classList.contains('visible')) drawVectorscope();
  });
}
btnABa.addEventListener('click', () => setAB('a'));
btnABb.addEventListener('click', () => setAB('b'));

/* ═══════ Measure ═══════ */
btnMeasure.addEventListener('click', async()=>{
  if(!currentFile) return;
  btnMeasure.disabled=true;
  wdeco.classList.add('active');
  setStatus(stMeasure,'Измерение уровня…');
  const form=new FormData();
  form.append('file',currentFile);
  form.append('extended','true');
  try{
    const r=await fetch(API+'/api/v2/analyze',{method:'POST',body:form,headers:authHeaders()});
    const d=await safeResponseJson(r);
    if(!r.ok) throw new Error(d.detail||r.statusText);
    setMeter(d.lufs);
    setCorrelation(d.stereo_correlation != null ? d.stereo_correlation : null);
    lastAnalyzeReport = { ...d, filename: currentFile ? currentFile.name : '' };
    const reportActions = document.getElementById('reportActions');
    if (reportActions) reportActions.style.display = 'inline-flex';
    if (lufsTimelineWrap && lufsTimelineCanvas && Array.isArray(d.lufs_timeline) && d.lufs_timeline.length > 0) {
      lastLufsTimelineData = {
        timeline: d.lufs_timeline,
        stepSec: d.timeline_step_sec || 0,
        durationSec: d.duration_sec != null ? d.duration_sec : d.duration
      };
      drawLufsTimeline(lastLufsTimelineData.timeline, lastLufsTimelineData.stepSec, lastLufsTimelineData.durationSec);
      lufsTimelineWrap.classList.add('visible');
    } else {
      lastLufsTimelineData = null;
      if (lufsTimelineWrap) lufsTimelineWrap.classList.remove('visible');
    }
    // Streaming Loudness Preview
    renderStreamingPreview(d.streaming_preview || null);
    const durationSec = d.duration_sec != null ? d.duration_sec : d.duration;
    // Update audio meta with server-side data
    if(d.peak_dbfs!=null) amPeak.textContent = d.peak_dbfs.toFixed(1)+' dB';
    if(d.channels!=null)  amCh.textContent   = d.channels===1?'Mono':'Stereo';
    if(d.sample_rate!=null) amSr.textContent  = (d.sample_rate/1000).toFixed(1)+' kHz';
    if(durationSec!=null) { amDur.textContent=fmtTime(durationSec); audioMeta.classList.add('visible'); }
    const peakTxt = d.peak_dbfs!=null ? `  ·  Peak ${d.peak_dbfs.toFixed(1)} dB` : '';
    setStatus(stMeasure, 'Измерение завершено'+peakTxt, 'ok');
    // M/S spectrum tabs
    if (d.spectrum_bars_mid || d.spectrum_bars_side) {
      updateSpectrumTabs(d);
    }
  }catch(e){
    const msg = friendlyError(e.message||'Ошибка измерения');
    setStatus(stMeasure, msg, 'err');
    toast(msg, 'err', msg.includes('ffmpeg') ? 8000 : 3000);
    setMeter(null);
  }
  btnMeasure.disabled=false;
  wdeco.classList.remove('active');
});

/* ═══════ P13: экспорт отчёта анализа ═══════ */
function buildAnalyzeReportText(data) {
  const lines = [];
  lines.push('Magic Master — Отчёт анализа');
  lines.push('============================');
  lines.push('');
  if (data.filename) lines.push('Файл: ' + data.filename);
  lines.push('Дата: ' + new Date().toLocaleString('ru-RU'));
  lines.push('');
  lines.push('--- Уровень громкости ---');
  if (data.lufs != null) lines.push('LUFS (интегрированный): ' + data.lufs.toFixed(2) + ' dB');
  if (data.peak_dbfs != null) lines.push('Peak dBFS: ' + data.peak_dbfs.toFixed(2) + ' dB');
  lines.push('');
  lines.push('--- Метаданные ---');
  const dur = data.duration_sec != null ? data.duration_sec : data.duration;
  if (dur != null) lines.push('Длительность: ' + (typeof fmtTime === 'function' ? fmtTime(dur) : Math.floor(dur/60)+':'+String(Math.floor(dur%60)).padStart(2,'0')));
  if (data.sample_rate != null) lines.push('Sample rate: ' + data.sample_rate + ' Hz');
  if (data.channels != null) lines.push('Каналы: ' + (data.channels === 1 ? 'Mono' : 'Stereo'));
  if (data.stereo_correlation != null) lines.push('Стерео-корреляция L/R: ' + data.stereo_correlation.toFixed(4) + ' (−1…+1)');
  if (Array.isArray(data.lufs_timeline) && data.lufs_timeline.length > 0) {
    const arr = data.lufs_timeline;
    const min = Math.min.apply(null, arr);
    const max = Math.max.apply(null, arr);
    const avg = arr.reduce((a,b)=>a+b,0)/arr.length;
    lines.push('');
    lines.push('--- LUFS по времени (краткосрочный) ---');
    lines.push('Мин: ' + min.toFixed(2) + ' dB  Макс: ' + max.toFixed(2) + ' dB  Среднее: ' + avg.toFixed(2) + ' dB');
    if (data.timeline_step_sec != null) lines.push('Шаг: ' + data.timeline_step_sec.toFixed(3) + ' с');
  }
  if (Array.isArray(data.spectrum_bars) && data.spectrum_bars.length > 0) {
    lines.push('');
    lines.push('--- Спектр ---');
    lines.push('Полос: ' + data.spectrum_bars.length + ' (логарифмическая шкала 20 Hz – 20 kHz)');
  }
  if (Array.isArray(data.vectorscope_points) && data.vectorscope_points.length > 0) {
    lines.push('');
    lines.push('--- Векторскоп ---');
    lines.push('Точек: ' + data.vectorscope_points.length);
  }
  lines.push('');
  lines.push('— сгенерировано Magic Master');
  return lines.join('\n');
}

const btnDownloadReport = document.getElementById('btnDownloadReport');
if (btnDownloadReport) {
  btnDownloadReport.addEventListener('click', () => {
    if (!lastAnalyzeReport) return;
    const text = buildAnalyzeReportText(lastAnalyzeReport);
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const name = (lastAnalyzeReport.filename || 'report').replace(/\.[^.]+$/, '') + '_analyze_report.txt';
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
    toast('Отчёт сохранён', 'inf');
  });
}

const btnDownloadReportJson = document.getElementById('btnDownloadReportJson');
if (btnDownloadReportJson) {
  btnDownloadReportJson.addEventListener('click', () => {
    if (!lastAnalyzeReport) return;
    const json = JSON.stringify(lastAnalyzeReport, null, 2);
    const blob = new Blob([json], { type: 'application/json;charset=utf-8' });
    const name = (lastAnalyzeReport.filename || 'report').replace(/\.[^.]+$/, '') + '_analyze_report.json';
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
    toast('Отчёт JSON сохранён', 'inf');
  });
}

/* ═══════ Master ═══════ */
btnMaster.addEventListener('click', async()=>{
  if(!currentFile) return;
  let targetLufs = parseFloat(targetLufsInput.value);
  if(isNaN(targetLufs)) targetLufs=-14;

  btnMaster.disabled=true;
  btnMaster.classList.add('processing');
  btnMasterTxt.textContent='Обработка…';
  wdeco.classList.add('active');
  setStatus(stMaster,'');
  resultPanel.classList.remove('visible');
  if (abLufsA) abLufsA.textContent = '—';
  if (abLufsB) abLufsB.textContent = '—';
  progWrap.classList.add('on');
  pipeline.classList.add('visible');
  resetPipelineSteps();
  updateOzoneSteps(selectedStyle);
  setProgress(0,'Инициализация…');

  const form=new FormData();
  form.append('file',currentFile);
  form.append('target_lufs',targetLufs);
  form.append('out_format',outFormat.value);
  form.append('style',selectedStyle);
  const ditherTypeEl = document.getElementById('ditherType');
  const autoBlankSecEl = document.getElementById('autoBlankSec');
  if (ditherTypeEl) form.append('dither_type', ditherTypeEl.value);
  if (autoBlankSecEl) form.append('auto_blank_sec', autoBlankSecEl.value);
  if (chainModulesConfig?.modules?.length) {
    const modules = chainModulesConfig.modules.map(m => { const { label, ...rest } = m; return rest; });
    form.append('config', JSON.stringify({ modules }));
  }
  collectProModuleParams(form);

  try{
    const startRes=await fetch(API+'/api/v2/master',{method:'POST',body:form,headers:authHeaders()});
    const startData=await safeResponseJson(startRes);
    if(!startRes.ok) throw new Error(startData.detail||startRes.statusText);
    const job_id=startData.job_id;
    const poll=async ()=>{ const r=await fetch(API+'/api/master/status/'+job_id); return safeResponseJson(r); };

    let data;
    do{
      await new Promise(r=>setTimeout(r,300));
      data=await poll();
      setProgress(data.progress||0, data.message||'Обработка…');
      if(data.status==='error') throw new Error(data.error||'Ошибка мастеринга');
    } while(data.status==='running');

    if(data.status!=='done') throw new Error('Неизвестный статус');

    // Download
    const r=await fetch(API+'/api/master/result/'+job_id);
    if(!r.ok) throw new Error('Не удалось скачать результат');
    const blob=await r.blob();
    const name=(currentFile.name.replace(/\.[^.]+$/,'')||'master')+'_mastered.'+outFormat.value;
    // Download
    const dlUrl=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=dlUrl; a.download=name; a.click();
    // Decode mastered for A/B player
    try {
      const mastAB = await blob.arrayBuffer();
      masteredBuffer = await getCtx().decodeAudioData(mastAB);
      abWrap.classList.add('visible');
      // Switch to B (mastered) automatically
      const origBuf = audioBuffer;
      audioBuffer = masteredBuffer;
      await new Promise(rr=>requestAnimationFrame(()=>requestAnimationFrame(rr)));
      drawWaveform(0);
      audioBuffer = origBuf;
      abMode='b'; btnABa.classList.remove('active'); btnABb.classList.add('active');
    } catch(e) { console.warn('A/B decode failed', e); }
    URL.revokeObjectURL(dlUrl);

    // Mark all pipeline steps done
    pipeline.querySelectorAll('.pipe-step').forEach(el=>{ el.classList.remove('active'); el.classList.add('done'); });
    setProgress(100,'Готово!');

    // Show before/after panel
    if(data.before_lufs!=null && data.after_lufs!=null){
      rBefore.textContent = data.before_lufs.toFixed(1)+' LUFS';
      rAfter.textContent  = data.after_lufs.toFixed(1)+' LUFS';
      if (abLufsA) abLufsA.textContent = data.before_lufs.toFixed(1);
      if (abLufsB) abLufsB.textContent = data.after_lufs.toFixed(1);
      const delta = data.after_lufs - data.before_lufs;
      const sign  = delta>0?'+':'';
      rDelta.innerHTML = `Изменение: <strong>${sign}${delta.toFixed(1)} dB</strong> · Цель: <strong>${targetLufs} LUFS</strong>`;
      resultPanel.classList.add('visible');
      setMeter(data.after_lufs);
      if (audioBuffer && masteredBuffer) showDawComparison(audioBuffer, masteredBuffer, data.before_lufs, data.after_lufs);
    }

    setStatus(stMaster,'Скачано: '+name,'ok');
    toast('Готово! Файл скачан: '+name, 'ok', 4000);

    // Reference Track matching — если пользователь загрузил эталонный трек
    if (refFile) {
      try {
        setStatus(stMaster, 'Применяю Reference Track…');
        const refForm = new FormData();
        refForm.append('file', blob, name);
        refForm.append('reference', refFile, refFile.name);
        const refStrEl = document.getElementById('refStrength');
        const refStrVal = refStrEl ? (parseFloat(refStrEl.value) / 100).toFixed(2) : '0.80';
        refForm.append('strength', refStrVal);
        refForm.append('out_format', outFormat.value);
        const refRes = await fetch(API + '/api/v2/reference-match', {
          method: 'POST', body: refForm, headers: authHeaders()
        });
        if (refRes.ok) {
          const refBlob = await refRes.blob();
          const refName = (currentFile.name.replace(/\.[^.]+$/, '') || 'master') + '_ref-matched.' + outFormat.value;
          const refUrl = URL.createObjectURL(refBlob);
          const refA = document.createElement('a');
          refA.href = refUrl; refA.download = refName; refA.click();
          URL.revokeObjectURL(refUrl);
          setStatus(stMaster, 'Готово! Скачано: ' + name + ' + ' + refName, 'ok');
          toast('Reference Track применён! Скачан: ' + refName, 'ok', 5000);
        } else {
          const refErr = await refRes.json().catch(() => ({}));
          toast('Reference Track: ' + (refErr.detail || 'ошибка'), 'err', 4000);
          setStatus(stMaster, 'Скачано: ' + name + ' (Reference Track — ошибка)', 'ok');
        }
      } catch (refErr) {
        toast('Reference Track: ' + (refErr.message || 'ошибка'), 'err', 4000);
      }
    }
    refreshTierAfterMaster();
    // Сохраняем в localStorage (локальная история)
    saveToHistory({
      name: currentFile.name,
      size: fmtSize(currentFile.size),
      before: data.before_lufs,
      after:  data.after_lufs,
      target: targetLufs,
      fmt:    outFormat.value,
      date:   fmtDate(Date.now()),
      ts:     Date.now(),
    });
    // Сохраняем на сервере (если залогинен)
    if (isLoggedIn()) {
      fetch(API + '/api/auth/record', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({
          filename: currentFile.name,
          style: selectedStyle || 'standard',
          out_format: outFormat.value,
          before_lufs: data.before_lufs,
          after_lufs: data.after_lufs,
          target_lufs: targetLufs,
        }),
      }).catch(() => {}); // silent — не критично
    }
    setTimeout(()=>progWrap.classList.remove('on'), 2500);

  }catch(e){
    const msg = friendlyError(e.message||'Ошибка мастеринга');
    setStatus(stMaster, msg, 'err');
    toast(msg, 'err', msg.includes('ffmpeg') ? 10000 : 3000);
    setProgress(0,'');
    progWrap.classList.remove('on');
    pipeline.classList.remove('visible');
    resetPipelineSteps();
  }

  btnMaster.disabled=false;
  btnMaster.classList.remove('processing');
  btnMasterTxt.textContent='Запустить мастеринг';
  wdeco.classList.remove('active');
});

/* ═══════ Версия приложения в футере ═══════ */
(function() {
  const el = document.getElementById('appVersion');
  if (!el) return;
  fetch(API + '/api/version').then(function(r) { return r.ok ? r.json() : null; }).then(function(d) {
    if (d && d.version) el.textContent = 'v' + d.version;
  }).catch(function() {});
})();

/* ═══════ Auth helpers ═══════ */
function getAuthToken() { return localStorage.getItem('mm_token'); }
function getAuthEmail() { return localStorage.getItem('mm_user_email') || ''; }
function getAuthTier()  { return localStorage.getItem('mm_user_tier') || 'free'; }
function isLoggedIn()   { return !!getAuthToken(); }

function authHeaders() {
  const token = getAuthToken();
  return token ? { 'Authorization': 'Bearer ' + token } : {};
}

function logout() {
  localStorage.removeItem('mm_token');
  localStorage.removeItem('mm_user_email');
  localStorage.removeItem('mm_user_tier');
  window.location.reload();
}

/* ═══════ Тариф и лимиты (Free tier / Pro) + режим отладки ═══════ */
let _tierInfo = { tier: 'free', remaining: 3, limit: 3, used: 0, reset_at: '' };
let _debugMode = typeof window.__MAGIC_MASTER_DEBUG__ !== 'undefined' && window.__MAGIC_MASTER_DEBUG__ === true;

if (_debugMode) {
  _tierInfo = { tier: 'pro', remaining: -1, limit: -1, used: 0, reset_at: null, debug: true };
}

async function loadTierLimits() {
  if (_debugMode) {
    applyTierUI();
    return;
  }
  try {
    const debugRes = await fetch(API + '/api/debug-mode');
    if (debugRes.ok) {
      const debugData = await debugRes.json();
      _debugMode = !!debugData.debug;
      if (_debugMode) {
        _tierInfo = { tier: 'pro', remaining: -1, limit: -1, used: 0, reset_at: null, debug: true };
        applyTierUI();
        return;
      }
    }
  } catch(e) { /* ignore */ }
  try {
    const r = await fetch(API + '/api/limits', { headers: authHeaders() });
    if (!r.ok) return;
    _tierInfo = await r.json();
    if (_tierInfo.debug) _debugMode = true;
  } catch(e) { /* оффлайн — оставляем дефолт */ }
  applyTierUI();
}

function applyTierUI() {
  const isPro = _debugMode || isLoggedIn() || _tierInfo.tier === 'pro' || _tierInfo.tier === 'studio';

  // Показываем правильные строки auth/tier
  const tierRow      = document.getElementById('tierRow');
  const authRow      = document.getElementById('authRow');
  const authRowGuest = document.getElementById('authRowGuest');

  if (isPro && isLoggedIn() && !_debugMode) {
    // Залогинен (не debug) — показываем email + logout и блок сохранённых пресетов (P10)
    if (tierRow)      tierRow.style.display = 'none';
    if (authRow)      authRow.style.display = 'flex';
    if (authRowGuest) authRowGuest.style.display = 'none';
    const emailEl = document.getElementById('authEmail');
    if (emailEl) emailEl.textContent = getAuthEmail();
    const priorityHint = document.getElementById('authPriorityHint');
    if (priorityHint) priorityHint.style.display = _tierInfo.priority_queue ? 'inline' : 'none';
    if (chainPresetsWrap) { chainPresetsWrap.style.display = 'block'; loadSavedPresetsList(); }
  } else {
    // Гость или режим отладки — показываем tier row или auth row
    if (_debugMode) {
      if (tierRow)      tierRow.style.display = 'flex';
      if (authRow)      authRow.style.display = 'none';
      if (authRowGuest) authRowGuest.style.display = 'none';
      const badge = document.getElementById('tierBadge');
      if (badge) {
        badge.classList.add('debug-badge');
        badge.classList.remove('warn', 'empty');
        badge.innerHTML = '<span class="tb-dot"></span> Режим отладки · все функции без входа';
      }
      const upgradeLink = document.getElementById('upgradeLink');
      if (upgradeLink) upgradeLink.style.display = 'none';
    } else {
      if (tierRow)      tierRow.style.display = 'flex';
      if (authRow)      authRow.style.display = 'none';
      if (authRowGuest) authRowGuest.style.display = 'flex';
      if (chainPresetsWrap) chainPresetsWrap.style.display = 'none';
      const badge    = document.getElementById('tierBadge');
      const remaining= document.getElementById('tierRemaining');
      if (badge) {
        badge.classList.remove('debug-badge');
        badge.innerHTML = '<span class="tb-dot"></span> Free · <span id="tierRemaining">' + (_tierInfo.remaining >= 0 ? _tierInfo.remaining : '∞') + '</span> мастеринга осталось';
        badge.classList.remove('warn', 'empty');
        if (_tierInfo.remaining === 0)       badge.classList.add('empty');
        else if (_tierInfo.remaining === 1)  badge.classList.add('warn');
      }
      const upgradeLink = document.getElementById('upgradeLink');
      if (upgradeLink) upgradeLink.style.display = '';
    }
  }

  // Разблокируем Pro-карточки если авторизован или режим отладки
  document.querySelectorAll('.style-card[data-tier="pro"]').forEach(c => {
    if (isPro) {
      c.classList.remove('locked');
    } else {
      c.classList.add('locked');
    }
  });

  // Разблокируем MP3/FLAC в select
  Array.from(outFormat.options).forEach(function(opt) {
    opt.disabled = !isPro && opt.dataset.tier === 'pro';
  });

  // Пакетная обработка — только для Pro / debug
  const batchSection = document.getElementById('batchSection');
  if (batchSection) batchSection.style.display = isPro ? 'block' : 'none';
}

// Logout button
(function() {
  const btn = document.getElementById('btnLogout');
  if (btn) btn.addEventListener('click', function() { logout(); });
})();

/* Обработчик смены формата — блокировать MP3/FLAC для Free (не для Pro / debug) */
outFormat.addEventListener('change', function() {
  const opt = outFormat.options[outFormat.selectedIndex];
  const isPro = _debugMode || isLoggedIn() || _tierInfo.tier === 'pro' || _tierInfo.tier === 'studio';
  if (!isPro && opt && opt.dataset.tier === 'pro') {
    showUpgradeModal(
      '🎵',
      'MP3 и FLAC — функция Pro',
      'Экспорт в MP3 (320 kbps) и FLAC (без потерь) доступен в тарифах Pro и Studio. На Free — экспорт WAV. Зарегистрируйтесь бесплатно для Pro доступа!'
    );
    outFormat.value = 'wav';
  }
});

/* ═══════ Upgrade Modal ═══════ */
function showUpgradeModal(icon, title, desc) {
  const overlay = document.getElementById('upgradeOverlay');
  if (!overlay) return;
  const iconEl = document.getElementById('upgradeModalIcon');
  const titleEl= document.getElementById('upgradeTitle');
  const descEl = document.getElementById('upgradeDesc');
  if (iconEl)  iconEl.textContent  = icon;
  if (titleEl) titleEl.textContent = title;
  if (descEl)  descEl.textContent  = desc;
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function hideUpgradeModal() {
  const overlay = document.getElementById('upgradeOverlay');
  if (!overlay) return;
  overlay.classList.remove('open');
  document.body.style.overflow = '';
}

(function() {
  const closeBtn  = document.getElementById('upgradeClose');
  const cancelBtn = document.getElementById('upgradeCancel');
  const overlay   = document.getElementById('upgradeOverlay');
  if (closeBtn)  closeBtn.addEventListener('click',  hideUpgradeModal);
  if (cancelBtn) cancelBtn.addEventListener('click', hideUpgradeModal);
  if (overlay)   overlay.addEventListener('click', function(e) {
    if (e.target === overlay) hideUpgradeModal();
  });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') hideUpgradeModal();
  });
})();

/* Перехватываем клик по заблокированной style-карточке (не показываем модалку в режиме отладки / Pro) */
styleGrid.addEventListener('click', function(e) {
  const card = e.target.closest('.style-card.locked');
  if (!card) return;
  const isPro = _debugMode || _tierInfo.tier === 'pro' || _tierInfo.tier === 'studio' || isLoggedIn();
  if (isPro) {
    return; // режим отладки или Pro — не блокируем, даём клику дойти до выбора стиля
  }
  const name = card.querySelector('.sc-name')?.textContent || '';
  showUpgradeModal('⚡', `Жанр "${name}" — функция Pro`,
    'Жанровые пресеты EDM, Hip-Hop, Classical, Podcast, Lo-fi, House доступны в тарифах Pro и Studio.\nЗарегистрируйтесь бесплатно — в бета-период все пользователи получают Pro!');
}, true); // capture — раньше обычного обработчика

/* Обновляем лимиты после успешного мастеринга */
function refreshTierAfterMaster() {
  if (_debugMode || _tierInfo.remaining === -1) return; // безлимит или режим отладки
  _tierInfo.used = (_tierInfo.used || 0) + 1;
  _tierInfo.remaining = Math.max(0, (_tierInfo.remaining || 0) - 1);
  applyTierUI();
  if (_tierInfo.remaining === 0) {
    toast('Лимит Free-тарифа исчерпан (3/3). Сброс: ' + _tierInfo.reset_at + '. Перейти на Pro →', 'err', 7000);
  }
}

/* ═══════ Пакетная обработка (Batch) ═══════ */
(function() {
  const batchSection = document.getElementById('batchSection');
  const batchFileInput = document.getElementById('batchFileInput');
  const batchFileList = document.getElementById('batchFileList');
  const batchCountEl = document.getElementById('batchCount');
  const btnSelectBatch = document.getElementById('btnSelectBatch');
  const btnBatchMaster = document.getElementById('btnBatchMaster');
  const batchPanel = document.getElementById('batchPanel');
  const batchJobsEl = document.getElementById('batchJobs');
  if (!batchSection || !batchFileInput || !btnBatchMaster) return;

  btnSelectBatch.addEventListener('click', function() { batchFileInput.click(); });

  batchFileInput.addEventListener('change', function() {
    const files = Array.from(batchFileInput.files || []).filter(function(f) {
      return /\.(wav|mp3|flac)$/i.test(f.name);
    });
    if (batchFileList) {
      batchFileList.innerHTML = files.length ? files.map(function(f) { return '<span>' + f.name + '</span>'; }).join('') : '';
    }
    if (batchCountEl) batchCountEl.textContent = files.length;
    if (btnBatchMaster) {
      btnBatchMaster.style.display = files.length ? 'inline-flex' : 'none';
      btnBatchMaster.disabled = files.length === 0;
    }
  });

  btnBatchMaster.addEventListener('click', async function() {
    const files = Array.from(batchFileInput.files || []).filter(function(f) { return /\.(wav|mp3|flac)$/i.test(f.name); });
    if (files.length === 0) return;
    btnBatchMaster.disabled = true;

    const form = new FormData();
    files.forEach(function(f) { form.append('files', f); });
    form.append('style', selectedStyle || 'standard');
    form.append('out_format', outFormat.value);
    form.append('target_lufs', targetLufsInput.value);
    if (chainModulesConfig && chainModulesConfig.modules) {
      form.append('config', JSON.stringify({ modules: chainModulesConfig.modules }));
    }
    if (document.getElementById('ditherType')) form.append('dither_type', document.getElementById('ditherType').value || 'tpdf');
    var autoBlank = document.querySelector('[name="auto_blank_sec"]') || document.getElementById('autoBlankSec');
    if (autoBlank) form.append('auto_blank_sec', autoBlank.value || '0');

    try {
      const r = await fetch(API + '/api/v2/batch', { method: 'POST', body: form, headers: authHeaders() });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || r.statusText || 'Ошибка batch');

      batchPanel.style.display = 'block';
      batchJobsEl.innerHTML = data.jobs.map(function(j) {
        return '<div class="batch-job-row" data-job-id="' + j.job_id + '">' +
          '<span class="batch-job-name">' + (j.filename || '—') + '</span>' +
          '<span class="batch-job-progress">0%</span>' +
          '<a class="batch-job-dl" href="#" style="display:none">Скачать</a></div>';
      }).join('');

      data.jobs.forEach(function(job) {
        (function poll(jobId, filename) {
          fetch(API + '/api/master/status/' + jobId, { headers: authHeaders() }).then(function(res) { return res.json(); }).then(function(st) {
            const row = batchJobsEl.querySelector('[data-job-id="' + jobId + '"]');
            if (!row) return;
            const prog = row.querySelector('.batch-job-progress');
            const dl = row.querySelector('.batch-job-dl');
            if (prog) prog.textContent = st.progress + '%';
            if (st.status === 'done') {
              if (prog) prog.textContent = '100%';
              if (dl) {
                dl.style.display = '';
                dl.href = API + '/api/master/result/' + jobId;
                dl.download = (filename || 'master').replace(/\.[^.]+$/, '') + '_mastered.' + outFormat.value;
                dl.target = '_blank';
                dl.classList.add('done');
              }
              return;
            }
            if (st.status === 'error') {
              if (prog) prog.textContent = 'Ошибка';
              return;
            }
            setTimeout(function() { poll(jobId, filename); }, 1200);
          }).catch(function() {
            const row = batchJobsEl.querySelector('[data-job-id="' + jobId + '"]');
            if (row && row.querySelector('.batch-job-progress')) row.querySelector('.batch-job-progress').textContent = '—';
          });
        })(job.job_id, job.filename);
      });

      toast('Пакет запущен: ' + data.jobs.length + ' файлов', 'ok', 3000);
      batchFileInput.value = '';
      batchFileList.innerHTML = '';
      batchCountEl.textContent = '0';
      btnBatchMaster.style.display = 'none';
    } catch (e) {
      toast(e.message || 'Ошибка пакетной обработки', 'err', 4000);
    }
    btnBatchMaster.disabled = false;
  });
})();

/* ═══════ Streaming Loudness Preview ═══════ */
function renderStreamingPreview(data) {
  const panel = document.getElementById('streamingPreview');
  const grid  = document.getElementById('streamingPreviewGrid');
  if (!panel || !grid) return;
  if (!data || typeof data !== 'object' || Object.keys(data).length === 0) {
    panel.style.display = 'none';
    return;
  }
  const statusLabel = { optimal: '✓ Оптимально', ok: '↓ Понизят', loud: '↓↓ Громко' };
  grid.innerHTML = Object.entries(data).map(([name, info]) => {
    const pen = info.penalty_db > 0 ? `−${info.penalty_db.toFixed(1)} dB` : '—';
    return `<div class="sp-item ${info.status}">
      <span class="sp-platform">${name}</span>
      <span class="sp-lufs">Цель: ${info.target_lufs} LUFS</span>
      <span class="sp-penalty">${pen} ${statusLabel[info.status] || ''}</span>
    </div>`;
  }).join('');
  panel.style.display = 'block';
}

/* ═══════ Reference Track ═══════ */
let refFile = null;

(function initReferenceTrack() {
  const refInput     = document.getElementById('refFileInput');
  const refFileName  = document.getElementById('refFileName');
  const btnRefClear  = document.getElementById('btnRefClear');
  const refStrWrap   = document.getElementById('refStrengthWrap');
  const refStrength  = document.getElementById('refStrength');
  const refStrVal    = document.getElementById('refStrengthVal');
  if (!refInput) return;

  refInput.addEventListener('change', () => {
    if (refInput.files && refInput.files[0]) {
      refFile = refInput.files[0];
      refFileName.textContent = refFile.name;
      if (btnRefClear) btnRefClear.style.display = 'inline';
      if (refStrWrap) refStrWrap.style.display = 'flex';
    }
  });
  if (btnRefClear) btnRefClear.addEventListener('click', () => {
    refFile = null;
    refInput.value = '';
    refFileName.textContent = 'не выбран';
    btnRefClear.style.display = 'none';
    if (refStrWrap) refStrWrap.style.display = 'none';
  });
  if (refStrength && refStrVal) {
    refStrength.addEventListener('input', () => {
      refStrVal.textContent = refStrength.value + '%';
    });
  }
})();

/* ═══════ PRO Processing Modules — аккордеон + слайдеры ═══════ */
(function initProModules() {
  // Аккордеон секции
  const head  = document.getElementById('proSectionHead');
  const body  = document.getElementById('proSectionBody');
  const chev  = document.getElementById('proSectionChevron');
  if (head && body) {
    head.addEventListener('click', () => {
      const open = body.classList.toggle('open');
      if (chev) chev.classList.toggle('open', open);
    });
  }

  // Раскрытие тела при включении модуля
  function wireModule(toggleId, bodyId, moduleId) {
    const tog = document.getElementById(toggleId);
    const bd  = document.getElementById(bodyId);
    const mod = document.getElementById(moduleId);
    if (!tog) return;
    tog.addEventListener('change', () => {
      if (bd) bd.style.display = tog.checked ? 'block' : 'none';
      if (mod) mod.classList.toggle('active', tog.checked);
    });
  }
  wireModule('denoiserEnabled',  'denoiserBody',  'modDenoiser');
  wireModule('deesserEnabled',   'deesserBody',   'modDeesser');
  wireModule('transientEnabled', 'transientBody', 'modTransient');
  wireModule('parallelEnabled',  'parallelBody',  'modParallel');
  wireModule('dynEQEnabled',     'dynEQBody',     'modDynEQ');

  // Слайдеры — обновление значений
  function wireSlider(sliderId, valId, fmt) {
    const sl = document.getElementById(sliderId);
    const vl = document.getElementById(valId);
    if (!sl || !vl) return;
    sl.addEventListener('input', () => { vl.textContent = fmt(sl.value); });
  }
  wireSlider('denoiserStrength',  'denoiserStrVal',     v => v + '%');
  wireSlider('deesserThreshold',  'deesserThrVal',      v => '−' + Math.abs(v) + ' dB');
  wireSlider('transientAttack',   'transientAttackVal', v => (v / 100).toFixed(2) + '×');
  wireSlider('transientSustain',  'transientSustainVal',v => (v / 100).toFixed(2) + '×');
  wireSlider('parallelMix',       'parallelMixVal',     v => v + '%');

  // Раскрытие тела модуля по клику на заголовок (toggle expand)
  document.querySelectorAll('.pro-module-head').forEach(h => {
    h.addEventListener('click', e => {
      if (e.target.closest('.pro-module-toggle')) return; // не мешать toggle
      h.closest('.pro-module').classList.toggle('expanded');
    });
  });
})();

/* Собираем параметры PRO-модулей для FormData */
function collectProModuleParams(form) {
  const gv = id => { const el = document.getElementById(id); return el ? el.value : null; };
  const gc = id => { const el = document.getElementById(id); return el ? el.checked : false; };

  // Spectral Denoiser
  if (gc('denoiserEnabled')) {
    form.append('denoise_strength', (parseFloat(gv('denoiserStrength') || 40) / 100).toFixed(2));
  }
  // De-esser
  if (gc('deesserEnabled')) {
    form.append('deesser_enabled', 'true');
    form.append('deesser_threshold', gv('deesserThreshold') || '-10');
  }
  // Transient Designer
  if (gc('transientEnabled')) {
    form.append('transient_attack',  (parseFloat(gv('transientAttack')  || 100) / 100).toFixed(2));
    form.append('transient_sustain', (parseFloat(gv('transientSustain') || 100) / 100).toFixed(2));
  }
  // Parallel Compression
  if (gc('parallelEnabled')) {
    form.append('parallel_mix', (parseFloat(gv('parallelMix') || 30) / 100).toFixed(2));
  }
  // Dynamic EQ
  if (gc('dynEQEnabled')) {
    form.append('dynamic_eq_enabled', 'true');
  }
}

/* ═══════ M/S Spectrum (Mid/Side tabs in spectrum card) ═══════ */
let lastSpectrumData = { mono: null, mid: null, side: null, active: 'mono' };

function updateSpectrumTabs(data) {
  if (!data) return;
  if (data.spectrum_bars)     lastSpectrumData.mono = data.spectrum_bars;
  if (data.spectrum_bars_mid) lastSpectrumData.mid  = data.spectrum_bars_mid;
  if (data.spectrum_bars_side)lastSpectrumData.side = data.spectrum_bars_side;

  const hasMidSide = lastSpectrumData.mid || lastSpectrumData.side;
  const head = document.getElementById('spectrumHead');
  if (!head) return;

  // Inject tabs if missing and M/S data is available
  if (hasMidSide && !head.querySelector('.spec-tab-wrap')) {
    const tabWrap = document.createElement('div');
    tabWrap.className = 'spec-tab-wrap';
    const modes = [
      { id: 'mono', label: 'MONO' },
      { id: 'mid',  label: 'MID' },
      { id: 'side', label: 'SIDE' },
    ];
    modes.forEach(({ id, label }) => {
      const btn = document.createElement('button');
      btn.className = 'spec-tab' + (id === lastSpectrumData.active ? ' active' : '');
      btn.dataset.mode = id;
      btn.textContent = label;
      btn.addEventListener('click', () => {
        head.querySelectorAll('.spec-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        lastSpectrumData.active = id;
        const bars = lastSpectrumData[id] || lastSpectrumData.mono;
        if (bars) drawSpectrum(bars);
      });
      tabWrap.appendChild(btn);
    });
    head.appendChild(tabWrap);
  } else if (!hasMidSide) {
    // Если M/S нет — убрать табы если были
    const wrap = head.querySelector('.spec-tab-wrap');
    if (wrap) wrap.remove();
  }

  // Render active tab
  const bars = lastSpectrumData[lastSpectrumData.active] || lastSpectrumData.mono;
  if (bars) drawSpectrum(bars);
}

/* Hook into existing spectrum rendering after measure */
const _origSetMeter = typeof setMeter === 'function' ? setMeter : null;

/* ═══════ Spectrum tabs after analyze ═══════ */
document.addEventListener('analyzeComplete', (e) => {
  if (e.detail) updateSpectrumTabs(e.detail);
});

/* Запускаем при загрузке: в режиме отладки сразу разблокируем Pro-карточки */
if (_debugMode) {
  if (typeof applyTierUI === 'function') applyTierUI();
}
loadTierLimits();
