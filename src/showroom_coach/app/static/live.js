/* Showroom Coach: live camera page. Vanilla JS (ES2020), no build step, no network except the backend.
 *
 * What it does. The camera goes into a hidden <video>. About five times a second the current picture is drawn onto an
 * offscreen canvas (at most 960 px wide) and sent over /ws/live as a JPEG (quality 0.7), one at a time: the next frame
 * goes out only after the server's overlay for the previous one comes back (stop-and-wait). The overlay says who is
 * where; the visible canvas then shows THAT SAME frame with every face region pixelated and blurred (the face of each
 * entry in `people` and every rect in the overlay's `faces`, the top of every detected box), boxes coloured by role (staff blue, customer orange; a
 * person with `state: null` is not confirmed yet and gets a thin dashed box), labels, waiting timers and a line between
 * people who are talking.
 *
 * Privacy. The visible canvas never shows the live <video> directly. It only ever shows a frame the server has analysed,
 * drawn together with that frame's own face rectangles, and the faces are covered before anything else is drawn. If
 * covering them fails, the frame is not shown at all. Nothing is stored; the offscreen frame is overwritten by the next
 * one. (People the detector misses are not covered: see the project notes.)
 *
 * Reconnects. When the socket drops the page keeps the camera on and retries after 1 s, 2 s, 4 s, 8 s, then every 10 s.
 * The server builds a fresh session per connection, so person numbers and scores start again after a reconnect.
 *
 * Demo mode: ?demo=/some/video.mp4 (same origin only) plays that file instead of the camera, for screenshots and tests.
 * All text from the server is shown with textContent or canvas fillText, never as markup.
 */
'use strict';
(() => {

const Q = new URLSearchParams(location.search);
const MAX_W = 960;                 // widest frame sent
const JPEG_QUALITY = 0.7;
const MAX_FPS = 5;
const MIN_GAP_MS = 1000 / MAX_FPS;
const BACKOFF_MS = [1000, 2000, 4000, 8000, 10000];   // then 10 s for ever
const SUMMARY_MS = 5000;           // mini scorecard refresh
const WATCHDOG_MS = 30000;         // no reply for this long: treat the connection as dead
const FIRST_REPLY_MS = 120000;     // ...but the first reply also waits for the models to load
const FEED_MAX = 100;
const TOAST_MS = 3200;
const STAFF = 'rgb(20,120,230)';   // same RGB as the rendered video and the rest of the UI
const CUSTOMER = 'rgb(255,140,0)';
const ALERT = '#C8321F';
const INK = '#1B2430';
const FONT = '"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,Arial,sans-serif';
// The only errors that are the reply to a FRAME (they stand in for its overlay, so they free the in-flight slot).
// "bad frame" is in the contract; "frame could not be processed" is what live_ws sends when the engine fails on a frame.
// Every other error ("unknown person", "bad control", ...) is an EXTRA message about a control and frees nothing.
const FRAME_ERRORS = new Set(['bad frame', 'frame could not be processed']);
const FALLBACK_PRESETS = {
  short: { label: 'Short clip (10–60 s)', thresholds: {} },
  dealership: { label: 'Dealership (SKAIVISION thresholds)', thresholds: {} },
};

const $ = (id) => document.getElementById(id);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const plural = (n, one, many = one + 's') => `${n} ${n === 1 ? one : many}`;

function mmss(sec) {
  if (sec == null || !Number.isFinite(sec)) return '—';
  const s = Math.max(0, Math.round(sec));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}` : `${m}:${String(r).padStart(2, '0')}`;
}

function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid != null && kid !== false) n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
}

/* ------------------------------------------------------------------ state and elements */

const view = $('view');
const ctx = view.getContext('2d');
const cam = $('cam');
const frameCv = document.createElement('canvas');          // the frame in flight: what the next overlay describes
const frameCtx = frameCv.getContext('2d');
const tinyCv = document.createElement('canvas');           // scratch for pixelating a face
const tinyCtx = tinyCv.getContext('2d');
const CAN_FILTER = typeof ctx.filter === 'string';         // canvas blur filter (not in older Safari)

const S = {
  running: false, starting: false,
  stream: null, demo: false,
  ws: null, wsToken: 0, attempt: 0, reconnectTimer: 0, watchdog: 0, loopTimer: 0, summaryTimer: 0, toastTimer: 0,
  inflight: false, lastSend: 0, gotOverlay: false, sessions: 0,
  presets: FALLBACK_PRESETS, preset: 'short', waitAlert: null,
  overlay: null, scale: { sx: 1, sy: 1 }, arrivals: [],      // arrival times of recent overlays (for the 'shown' rate)
  who: {},                       // pid -> label for events
  peopleKey: '', alertsKey: '', feedN: 0,
  serverError: null,
};

/* ------------------------------------------------------------------ small UI helpers */

function toast(text) {
  const t = $('toast');
  t.textContent = text;
  t.hidden = false;
  clearTimeout(S.toastTimer);
  S.toastTimer = setTimeout(() => { t.hidden = true; }, TOAST_MS);
}

function setConn(state, text) {
  const chip = $('connChip');
  chip.dataset.state = state;
  $('connText').textContent = text;
}

function showStage(title, text, problem = false) {
  const m = $('stageMsg');
  $('stageTitle').textContent = title;
  $('stageText').textContent = text;
  m.classList.toggle('is-problem', problem);
  m.setAttribute('role', problem ? 'alert' : 'status');
  m.hidden = false;
}
const hideStage = () => { $('stageMsg').hidden = true; };

function setButton(mode) {
  const b = $('btnStart');
  b.disabled = mode === 'starting';
  b.setAttribute('aria-busy', mode === 'starting' ? 'true' : 'false');
  b.classList.toggle('btn--primary', mode !== 'running');
  b.classList.toggle('btn--stop', mode === 'running');
  b.textContent = mode === 'running' ? 'Stop' : mode === 'starting' ? 'Starting…' : 'Start camera';
}

function clearCanvas() {
  ctx.fillStyle = '#11161D';
  ctx.fillRect(0, 0, view.width, view.height);
  view.classList.remove('is-stale', 'is-over-person');
  S.overlay = null;
}

/* ------------------------------------------------------------------ presets */

async function loadPresets() {
  const sel = $('preset');
  try {
    const res = await fetch('/api/presets');
    if (!res.ok) throw new Error(String(res.status));
    S.presets = await res.json();
  } catch {
    S.presets = FALLBACK_PRESETS;          // page opened without the backend: names only, no thresholds
  }
  const keys = Object.keys(S.presets);
  sel.replaceChildren(...keys.map((k) => el('option', { value: k, text: S.presets[k].label || k })));
  S.preset = keys.includes('short') ? 'short' : keys[0];
  sel.value = S.preset;
  sel.disabled = false;
  renderFacts();
}

function renderFacts() {
  const th = (S.presets[S.preset] || {}).thresholds || {};
  S.waitAlert = Number.isFinite(th.wait_alert) ? th.wait_alert : null;
  const rows = [];
  if (Number.isFinite(th.greet_dist) && Number.isFinite(th.greet_time)) rows.push(['Greeted', `within ${th.greet_dist} m for ${th.greet_time} s`]);
  if (Number.isFinite(th.engage_dist) && Number.isFinite(th.engage_time)) rows.push(['Talking', `within ${th.engage_dist} m for ${th.engage_time} s`]);
  if (S.waitAlert != null) rows.push(['Long-wait alert', `after ${S.waitAlert} s`]);
  $('presetFacts').replaceChildren(...rows.map(([k, v]) => el('div', {}, el('dt', { text: k }), el('dd', { text: v }))));
}

/* ------------------------------------------------------------------ camera */

function cameraProblem(err) {
  const name = (err && err.name) || '';
  if (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError') {
    return ['Camera access was blocked', 'Allow the camera for this site (the camera icon in the address bar), then press Start camera again.'];
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError' || name === 'OverconstrainedError') {
    return ['No camera was found', 'Connect a webcam, or check that it is not switched off, then press Start camera again.'];
  }
  if (name === 'NotReadableError' || name === 'TrackStartError' || name === 'AbortError') {
    return ['The camera is busy', 'Another program (a video call, for example) may be using it. Close that program, then press Start camera again.'];
  }
  return ['The camera could not be started', `${name || 'Error'}${err && err.message ? ': ' + err.message : ''}`];
}

function waitForVideo(timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    const tick = () => {
      if (cam.videoWidth > 0 && cam.readyState >= 2) resolve();
      else if (performance.now() - t0 > timeoutMs) reject(Object.assign(new Error('the camera sent no picture'), { name: 'NotReadableError' }));
      else setTimeout(tick, 50);
    };
    tick();
  });
}

async function openCamera() {
  const demo = Q.get('demo');
  if (demo) {
    const url = new URL(demo, location.href);
    if (url.origin !== location.origin) {
      showStage('Demo video refused', 'The demo video must come from this same server.', true);
      return false;
    }
    S.demo = true;
    cam.loop = true;
    cam.src = url.href;
    try { await cam.play(); await waitForVideo(); } catch (err) {
      showStage('The demo video could not be played', `${err.name || 'Error'}: ${err.message || 'unsupported video'}`, true);
      return false;
    }
    return true;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showStage('The camera is not available on this page',
      `Browsers only allow camera access on https:// pages and on http://localhost. Open this page as http://localhost${location.port ? ':' + location.port : ''}/live instead of using the computer's network address.`, true);
    return false;
  }
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 } }, audio: false });
    cam.srcObject = S.stream;
    await cam.play();
    await waitForVideo();
  } catch (err) {
    releaseCamera();
    const [title, text] = cameraProblem(err);
    showStage(title, text, true);
    return false;
  }
  return true;
}

function releaseCamera() {
  if (S.stream) for (const t of S.stream.getTracks()) t.stop();
  S.stream = null;
  cam.pause();
  cam.srcObject = null;
  cam.removeAttribute('src');
  cam.load();
  S.demo = false;
}

/* ------------------------------------------------------------------ start, stop */

async function start() {
  if (S.running || S.starting) return;
  S.starting = true;
  setButton('starting');
  setConn('connecting', 'Starting the camera…');
  showStage('Starting the camera…', 'If the browser asks for permission, choose Allow.');
  const ok = await openCamera();
  S.starting = false;
  if (!ok) {
    setButton('idle');
    setConn('problem', 'Camera unavailable');
    return;
  }
  S.running = true;
  S.attempt = 0;
  resetResults();
  setButton('running');
  setConn('connecting', 'Connecting…');
  showStage('Connecting…', 'The first frame can take a few seconds while the models load.');
  connect();
  S.summaryTimer = setInterval(askSummary, SUMMARY_MS);
}

function stop(note) {
  S.running = false;
  S.wsToken++;                                    // late events from the old socket are ignored
  clearTimeout(S.reconnectTimer);
  clearTimeout(S.watchdog);
  clearTimeout(S.loopTimer);
  clearInterval(S.summaryTimer);
  const ws = S.ws;
  S.ws = null;
  S.inflight = false;
  if (ws) { try { ws.close(1000); } catch { /* already closed */ } }
  releaseCamera();
  clearCanvas();
  $('hud').hidden = true;
  setButton('idle');
  setConn('off', 'Camera off');
  showStage(note ? note[0] : 'The camera is off', note ? note[1] : 'Press Start camera to begin again. The results below stay until you start a new session.', !!(note && note[2]));
}

/* ------------------------------------------------------------------ websocket */

function wsUrl() {
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live?preset=${encodeURIComponent(S.preset)}`;
}

function connect() {
  if (!S.running) return;
  const token = ++S.wsToken;
  let ws;
  try { ws = new WebSocket(wsUrl()); } catch { scheduleReconnect(); return; }
  S.ws = ws;
  S.inflight = false;
  S.gotOverlay = false;
  S.serverError = null;
  ws.onopen = () => {
    if (token !== S.wsToken) return;
    setConn('connecting', 'Starting analysis…');
    sendNextFrame();
  };
  ws.onmessage = (ev) => {
    if (token !== S.wsToken) return;
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    handleMessage(msg);
  };
  ws.onclose = (ev) => {
    if (token !== S.wsToken) return;
    S.ws = null;
    S.inflight = false;
    clearTimeout(S.watchdog);
    clearTimeout(S.loopTimer);
    if (!S.running) return;
    // 1008 (unknown preset) is always final. 1011 (the server cannot run live mode) is final only if no overlay ever came
    // during this run; once live mode has worked, a 1011 is just another dropped connection and gets the usual retries.
    if (ev.code === 1008 || (ev.code === 1011 && S.sessions === 0)) {
      const why = S.serverError || (ev.code === 1008 ? 'unknown thresholds preset' : 'live mode is unavailable');
      stop(['Live mode could not start', `The server said: ${why}. Press Start camera to try again.`, true]);
      return;
    }
    scheduleReconnect();
  };
  ws.onerror = () => { /* a close event always follows */ };
}

function scheduleReconnect() {
  if (!S.running) return;
  clearTimeout(S.reconnectTimer);
  const delay = BACKOFF_MS[Math.min(S.attempt, BACKOFF_MS.length - 1)];
  S.attempt += 1;
  setConn('reconnecting', `reconnecting… (attempt ${S.attempt})`);
  view.classList.add('is-stale');
  view.classList.remove('is-over-person');
  S.overlay = null;                                     // the boxes on the dimmed picture are history: not clickable
  S.reconnectTimer = setTimeout(connect, delay);
}

/** Give up on a socket that has gone quiet (no reply for WATCHDOG_MS) and start the reconnect cycle. */
function dropDeadSocket() {
  const ws = S.ws;
  S.wsToken++;
  S.ws = null;
  S.inflight = false;
  if (ws) { try { ws.close(); } catch { /* ignore */ } }
  scheduleReconnect();
}

function sendControl(msg) {
  const ws = S.ws;
  if (!ws || ws.readyState !== WebSocket.OPEN) { toast('Not connected right now. Try again in a moment.'); return false; }
  ws.send(JSON.stringify(msg));
  return true;
}

function askSummary() {
  const ws = S.ws;
  if (ws && ws.readyState === WebSocket.OPEN && S.gotOverlay) ws.send(JSON.stringify({ type: 'summary' }));
}

function handleMessage(msg) {
  if (!msg || typeof msg !== 'object') return;
  if (msg.type === 'overlay') {
    S.inflight = false;
    clearTimeout(S.watchdog);
    try {
      showOverlay(msg);
    } finally {
      sendNextFrame();                                  // whatever happened while drawing, ask for the next frame
    }
  } else if (msg.type === 'summary') {
    renderSummary(msg);
  } else if (msg.type === 'error') {
    if (FRAME_ERRORS.has(msg.error)) {                  // this frame's reply: skip it and carry on
      S.inflight = false;
      clearTimeout(S.watchdog);
      sendNextFrame();
    } else {
      if (S.gotOverlay) toast(`The server said: ${msg.error}`);
      else S.serverError = String(msg.error);            // only a reason sent while the session was starting is quoted
    }
  }
}

function showOverlay(msg) {
  if (!S.gotOverlay) {
    S.gotOverlay = true;
    S.attempt = 0;
    if (S.sessions++ > 0) {                               // a reconnect: the server started a new session
      S.who = {}; S.peopleKey = ''; S.overlay = null;
      renderAlerts([]);
      renderSummary({ scorecard: { store: {}, staff: [] } });
      addFeed('Connection restarted: people are numbered again and scores start from zero', 'is-note', null);
    }
    hideStage();
    $('hud').hidden = false;
    setConn('live', 'Live');
    askSummary();
  }
  view.classList.remove('is-stale');
  onOverlay(msg);
}

/* ------------------------------------------------------------------ frame loop (stop-and-wait) */

function sendNextFrame() {
  const ws = S.ws;
  if (!S.running || !ws || ws.readyState !== WebSocket.OPEN || S.inflight) return;
  clearTimeout(S.loopTimer);
  const wait = S.lastSend + MIN_GAP_MS - performance.now();
  if (wait > 0) { S.loopTimer = setTimeout(sendNextFrame, wait); return; }
  if (cam.readyState < 2 || !cam.videoWidth) { S.loopTimer = setTimeout(sendNextFrame, 100); return; }
  const w = Math.min(MAX_W, cam.videoWidth);
  const h = Math.max(2, Math.round(cam.videoHeight * w / cam.videoWidth));
  if (frameCv.width !== w || frameCv.height !== h) { frameCv.width = w; frameCv.height = h; }
  frameCtx.drawImage(cam, 0, 0, w, h);
  S.inflight = true;
  S.lastSend = performance.now();
  const token = S.wsToken;
  frameCv.toBlob((blob) => {
    if (token !== S.wsToken || !S.running || ws.readyState !== WebSocket.OPEN) return;
    if (!blob) { S.inflight = false; S.loopTimer = setTimeout(sendNextFrame, MIN_GAP_MS); return; }
    ws.send(blob);
    clearTimeout(S.watchdog);
    S.watchdog = setTimeout(() => { if (token === S.wsToken) dropDeadSocket(); }, S.gotOverlay ? WATCHDOG_MS : FIRST_REPLY_MS);
  }, 'image/jpeg', JPEG_QUALITY);
}

/* ------------------------------------------------------------------ drawing */

/** A face rectangle in canvas pixels, or null. `face` is [x1, y1, x2, y2] in the overlay's frame. When it is missing or
 *  empty and the person's `bbox` is known, the top quarter of the box is covered instead. */
function faceRect(face, bbox, sx, sy, cw, ch) {
  let [x1, y1, x2, y2] = Array.isArray(face) && face.length === 4 ? face.map(Number) : [0, 0, 0, 0];
  if (!(x2 - x1 >= 2 && y2 - y1 >= 2)) {
    if (!Array.isArray(bbox) || bbox.length !== 4) return null;
    const b = bbox.map(Number);
    x1 = b[0]; x2 = b[2]; y1 = b[1]; y2 = b[1] + (b[3] - b[1]) * 0.25;
  }
  x1 = clamp(Math.floor(x1 * sx), 0, cw); x2 = clamp(Math.ceil(x2 * sx), 0, cw);
  y1 = clamp(Math.floor(y1 * sy), 0, ch); y2 = clamp(Math.ceil(y2 * sy), 0, ch);
  return x2 - x1 >= 1 && y2 - y1 >= 1 ? { x: x1, y: y1, w: x2 - x1, h: y2 - y1 } : null;
}

/** Pixelate the region (detail is gone for good), then soften it with a canvas blur where the browser has one. */
function obscure(r) {
  const bw = Math.max(2, Math.round(r.w / 7)), bh = Math.max(2, Math.round(r.h / 7));
  tinyCv.width = bw; tinyCv.height = bh;
  tinyCtx.imageSmoothingEnabled = true;
  tinyCtx.drawImage(frameCv, r.x, r.y, r.w, r.h, 0, 0, bw, bh);
  ctx.save();
  ctx.beginPath();
  ctx.rect(r.x, r.y, r.w, r.h);
  ctx.clip();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tinyCv, 0, 0, bw, bh, r.x, r.y, r.w, r.h);         // opaque blocks cover the whole region
  if (CAN_FILTER) {
    ctx.imageSmoothingEnabled = true;
    ctx.filter = `blur(${Math.max(14, Math.round(Math.max(r.w, r.h) * 0.12))}px)`;
    ctx.drawImage(tinyCv, 0, 0, bw, bh, r.x, r.y, r.w, r.h);
  }
  ctx.restore();
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** A small filled label. `anchor` is the top-left corner; it is nudged to stay inside the picture. */
function tag(text, x, y, bg, fg, size) {
  ctx.font = `600 ${size}px ${FONT}`;
  const padX = size * 0.45, h = Math.round(size * 1.5), w = Math.ceil(ctx.measureText(text).width + padX * 2);
  x = clamp(x, 0, Math.max(0, view.width - w));
  y = clamp(y, 0, Math.max(0, view.height - h));
  ctx.fillStyle = bg;
  roundRect(x, y, w, h, h / 3);
  ctx.fill();
  ctx.fillStyle = fg;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + padX, y + h / 2 + 1);
  return { x, y, w, h };
}

const plainLabel = (label) => String(label || '').replace(/\s*⏱\s*/, ' · ');   // the clock glyph is missing from some fonts

function draw(o) {
  const cw = frameCv.width, ch = frameCv.height;
  if (o.w && o.h && Math.abs(o.w / o.h - cw / ch) > 0.02) throw new Error('the overlay does not describe this frame');
  if (view.width !== cw || view.height !== ch) { view.width = cw; view.height = ch; }
  const sx = cw / (o.w || cw), sy = ch / (o.h || ch);
  S.scale = { sx, sy };
  ctx.globalAlpha = 1;
  ctx.filter = 'none';
  ctx.drawImage(frameCv, 0, 0);
  try {                                                   // faces first, before anything else is drawn
    for (const p of o.people) {
      const r = faceRect(p.face, p.bbox, sx, sy, cw, ch);
      if (r) obscure(r);
    }
    for (const f of o.faces || []) {                      // the top 20% of every detected box, listed people included
      const r = faceRect(f, null, sx, sy, cw, ch);
      if (r) obscure(r);
    }
  } catch (err) {
    console.error('face cover failed', err);
    ctx.fillStyle = '#11161D';                            // never show a frame whose faces are not covered
    ctx.fillRect(0, 0, cw, ch);
    return;
  }

  const lw = Math.max(2, Math.round(cw / 320));
  const fs = Math.max(12, Math.round(cw / 64));
  const box = (p) => [p.bbox[0] * sx, p.bbox[1] * sy, p.bbox[2] * sx, p.bbox[3] * sy];
  const byPid = new Map(o.people.map((p) => [p.pid, p]));

  for (const p of o.people) {                             // boxes (state null = tentative: thin and dashed)
    const [x1, y1, x2, y2] = box(p);
    const tentative = p.state == null;
    ctx.lineWidth = tentative ? Math.max(1, lw - 1) : lw;
    ctx.setLineDash(tentative ? [lw * 3, lw * 2] : []);
    ctx.strokeStyle = p.role === 'staff' ? STAFF : CUSTOMER;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  ctx.setLineDash([]);

  for (const [a, b, kind] of o.pairs || []) {             // lines between people who are talking
    const pa = byPid.get(a), pb = byPid.get(b);
    if (!pa || !pb) continue;
    const [ax1, ay1, ax2, ay2] = box(pa), [bx1, by1, bx2, by2] = box(pb);
    const A = [(ax1 + ax2) / 2, (ay1 + ay2) / 2], B = [(bx1 + bx2) / 2, (by1 + by2) / 2];
    ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(...A); ctx.lineTo(...B);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = lw + 3; ctx.stroke();
    ctx.beginPath(); ctx.moveTo(...A); ctx.lineTo(...B);
    ctx.strokeStyle = INK; ctx.lineWidth = lw; ctx.stroke();
    tag(String(kind || 'engaged'), (A[0] + B[0]) / 2 - fs, (A[1] + B[1]) / 2 - fs * 0.75, INK, '#fff', Math.round(fs * 0.85));
  }

  for (const p of o.people) {                             // labels and states, on top
    const [x1, y1, , y2] = box(p);
    const staff = p.role === 'staff';
    const late = !staff && p.wait_s != null && S.waitAlert != null && p.wait_s >= S.waitAlert;
    const bg = late ? ALERT : staff ? STAFF : CUSTOMER;
    const fg = late || staff ? '#fff' : INK;
    const size = fs;
    const h = Math.round(size * 1.5);
    const top = y1 - h - 1 >= 0 ? y1 - h - 1 : y1 + lw;
    ctx.globalAlpha = p.state == null ? 0.7 : 1;
    tag(plainLabel(p.label) || `${staff ? 'S' : 'C'}${p.pid}`, x1, top, bg, fg, size);
    ctx.globalAlpha = 1;
    if (p.state) tag(String(p.state), x1 + lw, y2 - Math.round(fs * 0.85 * 1.5) - lw, 'rgba(17,22,29,0.82)', '#fff', Math.round(fs * 0.85));
  }
}

/* ------------------------------------------------------------------ overlay -> page */

function onOverlay(o) {
  S.overlay = o;
  try { draw(o); } catch (err) {
    console.error('draw failed', err);
    clearCanvas();                                        // fail closed: no half-covered picture stays on screen
  }
  $('statFps').textContent = Number.isFinite(o.fps) ? o.fps.toFixed(1) : '0.0';   // server: frames over wall time, last 10 frames
  const now = performance.now();
  S.arrivals.push(now);
  while (S.arrivals.length > 1 && now - S.arrivals[0] > 3000) S.arrivals.shift();
  $('statShown').textContent = S.arrivals.length > 1 ? (1000 * (S.arrivals.length - 1) / (now - S.arrivals[0])).toFixed(1) : '0.0';
  $('statPeople').textContent = String(o.people.length);
  $('statPeopleWord').textContent = o.people.length === 1 ? 'person' : 'people';
  for (const p of o.people) S.who[p.pid] = p.role === 'staff' ? String(p.label || `S${p.pid}`) : `C${p.pid}`;
  renderPeople(o.people);
  renderAlerts(o.alerts || []);
  pushEvents(o.events || []);
}

function renderPeople(people) {
  const key = people.map((p) => `${p.pid}|${p.role}|${p.override}|${p.state == null}|${p.role === 'staff' ? p.label : ''}`).join(';');
  $('peopleCount').textContent = people.length ? String(people.length) : '';
  if (key === S.peopleKey) return;
  S.peopleKey = key;
  const host = $('peopleList');
  if (!people.length) { host.replaceChildren(el('p', { class: 'live__empty', text: 'Nobody in view.' })); return; }
  host.replaceChildren(...people.map((p) => {
    const staff = p.role === 'staff';
    const name = staff ? plainLabel(p.label) : `C${p.pid}`;
    return el('button', {
      type: 'button', class: `pchip pchip--${staff ? 'staff' : 'customer'}`,
      'aria-label': `${name} is ${staff ? 'staff' : 'a customer'}${p.override ? ' (set by you)' : ''}. Switch to ${staff ? 'customer' : 'staff'}.`,
      title: `Switch ${name} to ${staff ? 'customer' : 'staff'}`,
      onclick: () => toggle(p),
    }, name, el('small', { text: `${staff ? 'staff' : 'customer'}${p.override ? ' · set by you' : ''}${p.state == null ? ' · just seen' : ''}` }));
  }));
}

function renderAlerts(alerts) {
  const key = alerts.join('\n');
  if (key === S.alertsKey) return;
  S.alertsKey = key;
  $('alertCount').textContent = alerts.length ? String(alerts.length) : '';
  $('liveAlerts').replaceChildren(...(alerts.length ? alerts.map((a) => el('li', { text: String(a) })) : [el('li', { class: 'alerts__empty', text: 'No alerts yet.' })]));
}

const who = (pid) => (pid == null ? 'someone' : S.who[pid] || `S${pid}`);
const cust = (pid) => (pid == null ? 'someone' : `C${pid}`);

function describeEvent(e) {
  const d = e.data || {};
  switch (e.type) {
    case 'enter': return [`${cust(e.customer)} came in`, false];
    case 'leave': return [d.greeted ? `${cust(e.customer)} left after being greeted` : `${cust(e.customer)} left`, false];
    case 'greet': return [d.time_to_greet != null
      ? `${who(e.staff)} greeted ${cust(e.customer)} after ${Math.round(d.time_to_greet)} s`
      : `${who(e.staff)} greeted ${cust(e.customer)}`, false];
    case 'engagement_start': return [`${who(e.staff)} started talking with ${cust(e.customer)}`, false];
    case 'engagement': return [`${who(e.staff)} finished with ${cust(e.customer)}${e.t_end != null ? ` after ${mmss(e.t_end - e.t)}` : ''}`, false];
    case 'wait_alert': {
      const free = (d.free_staff || []).map(who).join(', ');
      return [`${cust(e.customer)} has waited ${mmss(d.wait_s)}${free ? `, free: ${free}` : ', all staff busy'}`, true];
    }
    case 'walkout': return [`${cust(e.customer)} walked out without being greeted`, true];
    case 'party': return [`${(d.members || []).map(cust).join(', ')} are one group`, false];
    case 'handoff': return [`${cust(e.customer)} handed from ${who(d.from_staff)} to ${who(d.to_staff)}`, false];
    default: return [String(e.type), false];
  }
}

function addFeed(text, kind, t) {
  const feed = $('feed');
  if (feed.firstElementChild && feed.firstElementChild.classList.contains('alerts__empty')) feed.replaceChildren();
  feed.prepend(el('li', { class: kind || null }, el('span', { class: 'feed__t', text: t == null ? '' : mmss(t) }), el('span', { class: 'feed__text', text })));
  while (feed.children.length > FEED_MAX) feed.lastElementChild.remove();
  S.feedN += 1;
  $('feedCount').textContent = String(S.feedN);
}

function pushEvents(events) {
  for (const e of events) {
    // Everyone already in view when the session starts "enters" at once with an unknown entry time: that is not news.
    if (e.type === 'enter' && !(e.data && e.data.entry_known)) continue;
    const [text, bad] = describeEvent(e);
    addFeed(text, bad ? 'is-alert' : null, e.t);
  }
}

/* ------------------------------------------------------------------ summary (mini scorecard) */

function renderSummary(msg) {
  const sc = msg.scorecard || {};
  const s = sc.store || {};
  const waits = s.wait_alerts ?? 0;
  const tiles = [
    ['Greet rate', s.greet_rate == null ? '—' : `${Math.round(s.greet_rate * 100)}%`,
      s.customers ? `${s.greeted ?? 0} of ${plural(s.customers, 'customer')} greeted` : 'No customers yet'],
    ['Median time to greet', mmss(s.median_time_to_greet),
      s.customers_entry_known ? `Measured on ${s.customers_entry_known} of ${s.customers} customers seen entering` : 'No customer seen entering yet'],
    ['Walk-outs', String(s.walkouts ?? 0), 'Left without being greeted', (s.walkouts ?? 0) > 0],
    ['Unattended waiting', mmss(s.unattended_wait_s), `Waited while staff were free, ${plural(waits, 'long-wait alert')}`],
  ];
  $('liveTiles').replaceChildren(...tiles.map(([name, value, note, bad]) =>
    el('div', { class: 'tile' }, el('p', { class: 'tile__name', text: name }),
      el('p', { class: `tile__value${bad ? ' is-bad' : ''}`, text: value }), el('p', { class: 'tile__note', text: note }))));

  const rows = sc.staff || [];
  const host = $('liveScorecard');
  if (!rows.length) {
    host.replaceChildren(el('p', { class: 'alerts__empty', text: 'Nobody is marked as staff yet. Click a person to mark them.' }));
    return;
  }
  const cols = [
    ['Greets', (r) => r.greets],
    ['Median greet', (r) => mmss(r.median_time_to_greet)],
    ['Engaged', (r) => mmss(r.engaged_total_s)],
    ['Missed while free', (r) => r.missed_while_free, (r) => r.missed_while_free > 0],
  ];
  host.replaceChildren(el('div', { class: 'table-wrap' }, el('table', {},
    el('caption', { class: 'sr-only', text: 'Scorecard for each salesperson, so far' }),
    el('thead', {}, el('tr', {}, el('th', { scope: 'col', text: 'Salesperson' }), cols.map((c) => el('th', { scope: 'col', text: c[0] })))),
    el('tbody', {}, rows.map((r) => el('tr', {},
      el('th', { scope: 'row' }, el('span', { class: 'who' }, el('i', { class: 'dot' }), r.name ? [r.name, el('small', { text: `S${r.staff}` })] : `S${r.staff}`)),
      cols.map(([, get, bad]) => el('td', { class: `num${bad && bad(r) ? ' is-bad' : ''}`, text: String(get(r) ?? '—') }))))))));
}

function resetResults() {
  S.who = {}; S.peopleKey = ''; S.alertsKey = ''; S.feedN = 0; S.overlay = null; S.sessions = 0; S.arrivals = [];
  $('feed').replaceChildren(el('li', { class: 'alerts__empty', text: 'Events appear here as they happen, newest first.' }));
  $('feedCount').textContent = '';
  $('peopleList').replaceChildren(el('p', { class: 'live__empty', text: 'Nobody in view yet.' }));
  $('peopleCount').textContent = '';
  renderAlerts([]);
  renderSummary({ scorecard: { store: {}, staff: [] } });
}

/* ------------------------------------------------------------------ clicks */

/** The pointer position in canvas pixels, or null over the letterbox bars. The canvas is shown with `object-fit: contain`,
 *  so the picture can be smaller than the element's box (a 4:3 camera in a wide box): scale by the contain factor and
 *  subtract the centring offsets instead of stretching the whole box over the picture. */
function canvasPoint(ev) {
  const r = view.getBoundingClientRect();
  if (!r.width || !r.height || !view.width || !view.height) return null;
  const k = Math.min(r.width / view.width, r.height / view.height);
  const x = (ev.clientX - r.left - (r.width - view.width * k) / 2) / k;
  const y = (ev.clientY - r.top - (r.height - view.height * k) / 2) / k;
  return x >= 0 && y >= 0 && x <= view.width && y <= view.height ? [x, y] : null;
}

function personAt(ev) {
  const o = S.overlay;
  if (!o) return null;
  const pt = canvasPoint(ev);
  if (!pt) return null;
  const x = pt[0] / S.scale.sx, y = pt[1] / S.scale.sy;
  let best = null, bestArea = Infinity;
  for (const p of o.people) {                             // the smallest box under the pointer wins (someone in front)
    const [x1, y1, x2, y2] = p.bbox;
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
      const area = (x2 - x1) * (y2 - y1);
      if (area < bestArea) { best = p; bestArea = area; }
    }
  }
  return best;
}

function toggle(p) {
  const staff = p.role === 'staff';
  if (sendControl({ type: 'toggle', pid: p.pid })) toast(`${staff ? plainLabel(p.label) : `C${p.pid}`} is now ${staff ? 'a customer' : 'staff'}`);
}

view.addEventListener('click', (ev) => { const p = personAt(ev); if (p) toggle(p); });
view.addEventListener('mousemove', (ev) => view.classList.toggle('is-over-person', !!personAt(ev)));
view.addEventListener('mouseleave', () => view.classList.remove('is-over-person'));

/* ------------------------------------------------------------------ wiring */

$('btnStart').addEventListener('click', () => (S.running ? stop() : start()));
$('preset').addEventListener('change', (ev) => {
  S.preset = ev.target.value;
  renderFacts();
  if (S.running && S.ws && S.ws.readyState === WebSocket.OPEN && sendControl({ type: 'preset', name: S.preset })) {
    addFeed(`Thresholds changed to “${(S.presets[S.preset] || {}).label || S.preset}”: scores are recalculated`, 'is-note', S.overlay ? S.overlay.t : null);
  }
});
$('skip').addEventListener('click', (ev) => { ev.preventDefault(); $('main').focus(); });
addEventListener('pagehide', () => { if (S.running || S.stream) stop(); });

if (Q.get('demo')) $('demoFlag').hidden = false;
clearCanvas();
resetResults();
loadPresets();

})();
