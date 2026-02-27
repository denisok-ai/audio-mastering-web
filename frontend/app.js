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
const spectrumCard   = document.getElementById('spectrumCard');
const spectrumCanvas = document.getElementById('spectrumCanvas');
const vectorscopeCard   = document.getElementById('vectorscopeCard');
const vectorscopeCanvas = document.getElementById('vectorscopeCanvas');
const chainModulesHead   = document.getElementById('chainModulesHead');
const chainModulesBody   = document.getElementById('chainModulesBody');
const chainModulesList   = document.getElementById('chainModulesList');
const chainModulesChevron= document.getElementById('chainModulesChevron');

let currentFile = null;
/** Текущий конфиг цепочки (из GET /api/v2/chain/default), для отправки в POST /api/v2/master при изменённом порядке */
let chainModulesConfig = null;
/** Данные для графика LUFS по времени (после расширенного замера) */
let lastLufsTimelineData = null;

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
function friendlyError(msg) {
  if (!msg) return msg;
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

function drawSpectrum() {
  if (!spectrumCanvas || !audioBuffer) return;
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
  const bars = getSpectrumBars(audioBuffer);
  const barW = W / bars.length;
  const refDb = 0;
  const minDb = -60;
  for (let i = 0; i < bars.length; i++) {
    const db = bars[i];
    const norm = Math.max(0, Math.min(1, (db - minDb) / (refDb - minDb)));
    const h = norm * H * 0.92;
    const x = i * barW;
    const grd = ctx.createLinearGradient(x, H, x, H - h);
    grd.addColorStop(0, 'rgba(34,211,238,0.35)');
    grd.addColorStop(1, 'rgba(34,211,238,0.9)');
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
    const r=await fetch(API+'/api/v2/analyze',{method:'POST',body:form});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||r.statusText);
    setMeter(d.lufs);
    setCorrelation(d.stereo_correlation != null ? d.stereo_correlation : null);
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
    const durationSec = d.duration_sec != null ? d.duration_sec : d.duration;
    // Update audio meta with server-side data
    if(d.peak_dbfs!=null) amPeak.textContent = d.peak_dbfs.toFixed(1)+' dB';
    if(d.channels!=null)  amCh.textContent   = d.channels===1?'Mono':'Stereo';
    if(d.sample_rate!=null) amSr.textContent  = (d.sample_rate/1000).toFixed(1)+' kHz';
    if(durationSec!=null) { amDur.textContent=fmtTime(durationSec); audioMeta.classList.add('visible'); }
    const peakTxt = d.peak_dbfs!=null ? `  ·  Peak ${d.peak_dbfs.toFixed(1)} dB` : '';
    setStatus(stMeasure, 'Измерение завершено'+peakTxt, 'ok');
  }catch(e){
    const msg = friendlyError(e.message||'Ошибка измерения');
    setStatus(stMeasure, msg, 'err');
    toast(msg, 'err', msg.includes('ffmpeg') ? 8000 : 3000);
    setMeter(null);
  }
  btnMeasure.disabled=false;
  wdeco.classList.remove('active');
});

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

  try{
    const startRes=await fetch(API+'/api/v2/master',{method:'POST',body:form});
    if(!startRes.ok){
      const err=await startRes.json().catch(()=>({}));
      throw new Error(err.detail||startRes.statusText);
    }
    const {job_id}=await startRes.json();
    const poll=()=>fetch(API+'/api/master/status/'+job_id).then(r=>r.json());

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
      const delta = data.after_lufs - data.before_lufs;
      const sign  = delta>0?'+':'';
      rDelta.innerHTML = `Изменение: <strong>${sign}${delta.toFixed(1)} dB</strong> · Цель: <strong>${targetLufs} LUFS</strong>`;
      resultPanel.classList.add('visible');
      setMeter(data.after_lufs);
      if (audioBuffer && masteredBuffer) showDawComparison(audioBuffer, masteredBuffer, data.before_lufs, data.after_lufs);
    }

    setStatus(stMaster,'Скачано: '+name,'ok');
    toast('Готово! Файл скачан: '+name, 'ok', 4000);
    // Save to history
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
