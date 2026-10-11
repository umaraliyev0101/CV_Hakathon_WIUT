/* Showroom Coach: static frontend. Vanilla JS (ES2020), no build step, no network except the backend API.
 *
 * Views: upload -> progress -> confirm -> results.  State lives in `state`; the DOM is built with el()
 * (textContent only, so file names and staff names can never inject markup).
 *
 * Mock mode: open the page with ?mock=1 (add &fail=1 to see the failure view). Every API call is then
 * answered from static/mock/*.json by mockApi() below, which is the only mock-aware code besides the
 * "Demo data" flag. Asset and mock URLs are relative; index.html adds <base href="/static/"> when the
 * backend serves it at "/".
 */
'use strict';
(() => {

const Q = new URLSearchParams(location.search);
const MOCK = Q.has('mock');
const POLL_MS = 700;
const ALLOWED_EXT = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v'];
const MAX_BYTES = 1024 ** 3;
const MAX_SALES_BYTES = 1024 ** 2;   // sales CSV, optional
const STAGES = ['Upload', 'Detect people', 'Work out roles', 'Ready to confirm'];

const $ = (sel, root = document) => root.querySelector(sel);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const plural = (n, one, many = one + 's') => `${n} ${n === 1 ? one : many}`;

/** m:ss (h:mm:ss from an hour up); "—" when missing. */
function mmss(sec) {
  if (sec == null || !Number.isFinite(sec)) return '—';
  const s = Math.max(0, Math.round(sec));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}` : `${m}:${String(r).padStart(2, '0')}`;
}

function fmtBytes(n) {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
}

function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k === 'style') n.style.cssText = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid != null && kid !== false) n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
}

const state = {
  view: 'upload', presets: {}, presetTouched: false,
  file: null, jobId: null, job: null,
  people: [], thumbs: {}, draft: { roles: {}, names: {} },
  results: null, videoEl: null, videoVersion: 0,
  tl: null, pollToken: 0, videoToken: 0,
  epoch: 0,            // bumped by resetToUpload: async work started before a reset must not touch the UI
  upload: null,        // the in-flight upload attempt {aborted, xhr}
  loadedNames: {},     // names as the server last reported them (to detect cleared names)
  reconnect: null,     // what the "Retry" button does after a connection loss
};

/* ------------------------------------------------------------------ API */

class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

function detailOf(data, fallback) {
  const d = data && data.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d) && d.length) return d.map((x) => x.msg || String(x)).join('; ');
  return fallback;
}

async function api(method, path, body) {
  if (MOCK) return mockApi(method, path, body);
  const form = body instanceof FormData;      // a file: the browser sets the multipart Content-Type itself
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: body && !form ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? (form ? body : JSON.stringify(body)) : undefined,
    });
  } catch {
    throw new ApiError(0, 'Could not reach the Showroom Coach server. Is it still running?');
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, detailOf(data, `The server answered ${res.status} ${res.statusText}.`));
  return data;
}

/** Multipart upload with progress (fetch cannot report upload progress). */
function uploadJob(file, preset, onProgress, attempt) {
  if (MOCK) return mockApi('POST', '/api/jobs', { onProgress, attempt });
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append('preset', preset);
    fd.append('file', file);
    const x = new XMLHttpRequest();
    attempt.xhr = x;
    x.open('POST', '/api/jobs');
    x.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    x.onload = () => {
      let data = null;
      try { data = JSON.parse(x.responseText); } catch { /* not JSON */ }
      if (x.status >= 200 && x.status < 300 && data && data.job_id) resolve(data);
      else reject(new ApiError(x.status, uploadMessage(x.status, data)));
    };
    x.onerror = () => reject(new ApiError(0, 'Could not reach the Showroom Coach server. Is it still running?'));
    x.onabort = () => reject(new ApiError(-1, 'Upload cancelled.'));
    x.send(fd);
  });
}

function uploadMessage(status, data) {
  if (status === 415) return 'That file type is not supported. Use MP4, MOV, AVI, MKV, WEBM or M4V.';
  if (status === 413) return 'That file is larger than 1 GB. Trim the clip and try again.';
  return detailOf(data, `The upload failed (${status}).`);
}

/* ------------------------------------------------------- mock mode only */

const mock = { t0: 0, rescored: 0, roles: null, names: null, sales: null };

/** Maps an API call to static/mock/*.json. Progress is simulated over ~5 s; a re-score "renders" for 2.5 s. */
async function mockApi(method, path, body) {
  const read = async (file) => {
    const res = await fetch(`mock/${file}`);
    if (!res.ok) throw new ApiError(res.status, `mock/${file} is missing`);
    return res.json();
  };
  const withEdits = (r) => {                       // keep roles/names the user posted
    if (mock.roles) r.roles = { ...r.roles, ...mock.roles };
    if (mock.names) r.names = { ...r.names, ...mock.names };
    if (mock.sales) {                              // pretend every salesperson sold twice
      r.sales = mock.sales;
      for (const s of r.scorecard.staff) { s.sold = 2; s.conversion = s.customers_engaged ? 2 / s.customers_engaged : null; }
    }
    return r;
  };
  if (path === '/api/presets') return read('presets.json');
  if (method === 'POST' && path === '/api/jobs') {
    for (let i = 1; i <= 5; i++) {
      if (body.attempt.aborted) throw new ApiError(-1, 'Upload cancelled.');
      body.onProgress(i / 5);
      await sleep(MOCK_SLOW ? 700 : 140);
    }
    if (body.attempt.aborted) throw new ApiError(-1, 'Upload cancelled.');
    Object.assign(mock, { t0: Date.now(), rescored: 0, roles: null, names: null, sales: null });
    return { job_id: 'mock0001' };
  }
  if (path.endsWith('/people')) {
    if (MOCK_EMPTY) return { people: [] };
    const d = await read('people.json');
    for (const p of d.people) {
      if (mock.roles?.[p.pid]) p.role = mock.roles[p.pid];
      if (mock.names?.[p.pid] !== undefined) p.name = mock.names[p.pid] || null;
    }
    return d;
  }
  if (path.endsWith('/sales')) {
    if (method === 'DELETE') mock.sales = null;
    else mock.sales = { rows: Math.max(0, (await body.get('file').text()).split(/\r?\n/).filter(Boolean).length - 1), unmatched: ['Karim'], skipped: 0 };
    return withEdits(await read('results.json'));
  }
  if (path.endsWith('/roles')) {
    mock.rescored = Date.now();
    mock.roles = { ...mock.roles, ...body.roles };
    mock.names = { ...mock.names, ...body.names };
    const r = withEdits(await read('results.json'));
    if (body.preset) r.preset = body.preset;
    return { ...r, video_status: 'rendering' };
  }
  if (path.endsWith('/results')) {
    const r = withEdits(await read('results.json'));
    if (MOCK_EMPTY) return emptyResults(r);
    r.video_status = mock.rescored && Date.now() - mock.rescored < 2500 ? 'rendering' : 'ready';
    return r;
  }
  if (/^\/api\/jobs\/[^/]+$/.test(path)) {
    const e = (Date.now() - mock.t0) / 1000;
    let stage = e < 0.8 ? 'queued' : e < 2.8 ? 'perceiving' : e < 4.4 ? 'analyzing' : 'needs_confirm';
    if (MOCK_FAIL && e > 2) stage = 'failed';
    if (mock.rescored) stage = Date.now() - mock.rescored < 2500 ? 'rendering' : 'done';
    const job = await read(`job_${stage}.json`);
    if (!mock.rescored && stage !== 'failed') job.progress = clamp(e / 4.4, 0, 1);
    return job;
  }
  throw new ApiError(404, `no mock for ${method} ${path}`);
}
const MOCK_FAIL = Q.has('fail');     // ?mock=1&fail=1  the job fails part-way
const MOCK_EMPTY = Q.has('empty');   // ?mock=1&empty=1 nobody detected
const MOCK_SLOW = Q.has('slow');     // ?mock=1&slow=1  ~3.5 s upload (to try "New clip" mid-upload)
const emptyResults = (r) => ({
  ...r, persons: [], suggestions: {}, roles: {}, names: {}, events: [], timelines: {}, video_status: 'none', video_url: null,
  scorecard: { staff: [], store: { customers: 0, customers_entry_known: 0, greeted: 0, greet_rate: null, median_time_to_greet: null, walkouts: 0, unattended_wait_s: 0, wait_alerts: 0, engagements: 0 } },
  warnings: ['nobody detected — the camera may be too far or too dark'],
});

/* ---------------------------------------------------------------- views */

const VIEW_STEP = { upload: 0, progress: 1, confirm: 2, results: 3 };

function setView(name) {
  state.view = name;
  for (const s of document.querySelectorAll('.view')) s.hidden = s.dataset.view !== name;
  const idx = VIEW_STEP[name];
  document.querySelectorAll('#steps li').forEach((li, i) => {
    li.classList.toggle('is-current', i === idx);
    li.classList.toggle('is-done', i < idx);
    if (i === idx) li.setAttribute('aria-current', 'step'); else li.removeAttribute('aria-current');
  });
  $('#btnNew').hidden = name === 'upload';
  showNotice('');
  window.scrollTo(0, 0);
  const h = $(`#view-${name} [tabindex="-1"]`);
  if (h) h.focus({ preventScroll: true });
}

function showNotice(msg) {
  const n = $('#notice');
  n.textContent = msg || '';
  n.hidden = !msg;
}

let toastTimer = 0;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 3500);
}

function renderBadges(ul, list) {
  ul.replaceChildren(...(list || []).map((w) => el('li', { class: 'badge', text: w })));
}

/* --------------------------------------------------------------- upload */

async function loadPresets() {
  try {
    state.presets = await api('GET', '/api/presets');
  } catch (e) {
    state.presets = {};
    showNotice(`Could not load the threshold presets. ${e.message}`);
  }
  const keys = Object.keys(state.presets);
  for (const sel of [$('#preset'), $('#resPreset')]) {
    sel.replaceChildren(...keys.map((k) => el('option', { value: k, text: state.presets[k].label || k })));
  }
  $('#preset').value = keys.includes('short') ? 'short' : keys[0] || '';
  renderPresetFacts();
}

function renderPresetFacts() {
  const t = state.presets[$('#preset').value]?.thresholds;
  const m = (x) => `${+Number(x).toFixed(1)} m`;
  const facts = t ? [
    ['Greeting', `within ${m(t.greet_dist)} for ${mmss(t.greet_time)}`],
    ['Engagement', `within ${m(t.engage_dist)} for ${mmss(t.engage_time)}`],
    ['Long-wait alert', `after ${mmss(t.wait_alert)}`],
  ] : [];
  $('#presetFacts').replaceChildren(...facts.map(([k, v]) => el('div', {}, el('dt', { text: k }), el('dd', { text: v }))));
}

function showUploadError(msg) {
  const e = $('#uploadError');
  e.textContent = msg || '';
  e.hidden = !msg;
}

function setFile(file) {
  showUploadError('');
  const ext = file.name.split('.').pop().toLowerCase();
  if (!ALLOWED_EXT.includes(ext)) {
    return showUploadError(`${file.name} is not a supported video. Use MP4, MOV, AVI, MKV, WEBM or M4V.`);
  }
  if (file.size > MAX_BYTES) return showUploadError(`${file.name} is larger than 1 GB. Trim the clip and try again.`);
  state.file = file;
  $('#drop').classList.add('has-file');
  $('#dropTitle').textContent = file.name;
  $('#dropHint').textContent = `${fmtBytes(file.size)}. Choose another file to replace it.`;
  $('#btnAnalyse').disabled = false;
  $('#analyseHint').textContent = '';
  suggestPreset(file);
}

/** Read the clip length in the browser and pick the matching preset (unless the user already chose one). */
function suggestPreset(file) {
  const url = URL.createObjectURL(file);
  const v = document.createElement('video');
  v.preload = 'metadata';
  const done = () => URL.revokeObjectURL(url);
  v.onloadedmetadata = () => {
    if (Number.isFinite(v.duration)) {
      $('#dropHint').textContent = `${fmtBytes(file.size)}, ${mmss(v.duration)} long. Choose another file to replace it.`;
      const want = v.duration > 120 ? 'dealership' : 'short';
      if (!state.presetTouched && state.presets[want]) { $('#preset').value = want; renderPresetFacts(); }
    }
    done();
  };
  v.onerror = done;
  v.src = url;
}

function bindUpload() {
  const input = $('#file'), drop = $('#drop');
  input.addEventListener('change', () => { if (input.files[0]) setFile(input.files[0]); });
  $('#preset').addEventListener('change', () => { state.presetTouched = true; renderPresetFacts(); });
  ['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, () => drop.classList.remove('is-over')));
  drop.addEventListener('drop', (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
  // a missed drop must not navigate the tab away from the page
  for (const t of ['dragover', 'drop']) window.addEventListener(t, (e) => e.preventDefault());
  $('#uploadForm').addEventListener('submit', (e) => { e.preventDefault(); if (state.file) startJob(); });
  $('#btnRetry').addEventListener('click', resetToUpload);
  $('#btnNew').addEventListener('click', resetToUpload);
}

function resetToUpload() {
  state.epoch++; state.pollToken++; state.videoToken++;
  if (state.upload) { state.upload.aborted = true; if (state.upload.xhr) state.upload.xhr.abort(); }
  Object.assign(state, { file: null, jobId: null, job: null, people: [], thumbs: {}, loadedNames: {}, results: null, videoEl: null, tl: null, upload: null, reconnect: null });
  hideFail();
  $('#file').value = '';
  $('#drop').classList.remove('has-file');
  $('#dropTitle').textContent = 'Drop a video here, or choose a file';
  $('#dropHint').textContent = 'MP4, MOV, AVI, MKV, WEBM or M4V, up to 1 GB';
  $('#btnAnalyse').disabled = true;
  $('#analyseHint').textContent = 'Choose a video to start.';
  showUploadError('');
  history.replaceState(null, '', location.pathname + location.search);
  setView('upload');
}

/* ------------------------------------------------------------- progress */

function renderStages(current) {
  $('#stages').replaceChildren(...STAGES.map((s, i) =>
    el('li', { class: i < current ? 'is-done' : i === current ? 'is-current' : '', text: s, 'aria-current': i === current ? 'step' : null })));
}

function setMeter(frac, msg, active = true) {
  const pct = Math.round(clamp(frac, 0, 1) * 100);
  $('#meterFill').style.width = `${pct}%`;
  $('#meter').setAttribute('aria-valuenow', String(pct));
  $('#meter').classList.toggle('is-active', active);
  $('#progPct').textContent = `${pct}%`;
  if (msg != null) $('#progMsg').textContent = msg;
}

function renderJob(job) {
  const stage = { queued: 1, perceiving: 1, analyzing: 2 }[job.status] ?? 3;
  renderStages(stage);
  setMeter(job.progress ?? 0, job.message || job.status);
  renderBadges($('#progWarn'), job.warnings);
}

const isTransient = (err) => err.status === 0 || [502, 503, 504].includes(err.status);

/** Failure panel. With `retry` it becomes a "Connection lost" panel whose Retry button keeps the job (and #job= in the URL). */
function showFail(message, retry = null) {
  state.pollToken++;
  state.reconnect = retry;
  setMeter(state.job?.progress ?? 0, retry ? 'Connection lost' : 'Stopped', false);
  $('#progVerb').textContent = retry ? 'Lost contact while analysing' : 'Stopped analysing';
  $('#progFailTitle').textContent = retry ? 'Connection lost' : 'This clip could not be analysed';
  $('#progError').textContent = message;
  $('#btnReconnect').hidden = !retry;
  $('#btnRetry').hidden = !!retry;
  $('#progFail').hidden = false;
  $('#progFoot').hidden = true;
}

function hideFail() {
  state.reconnect = null;
  $('#progVerb').textContent = 'Analysing';
  $('#progFail').hidden = true;
  $('#progFoot').hidden = false;
}

async function startJob() {
  if (state.upload) return;                       // an upload is already running
  const file = state.file, preset = $('#preset').value;
  const attempt = state.upload = { aborted: false, xhr: null };
  const current = () => !attempt.aborted && state.upload === attempt;   // false once "New clip" was pressed
  $('#progFile').textContent = file.name;
  hideFail();
  renderBadges($('#progWarn'), []);
  renderStages(0);
  setMeter(0, 'Uploading…');
  setView('progress');
  try {
    const { job_id } = await uploadJob(file, preset, (p) => { if (current()) setMeter(p, `Uploading ${fmtBytes(file.size * p)} of ${fmtBytes(file.size)}`); }, attempt);
    if (!current()) return;
    state.upload = null;
    state.jobId = job_id;
    if (!MOCK) history.replaceState(null, '', `${location.pathname}${location.search}#job=${job_id}`);
    setMeter(0, 'Waiting for the analysis to start…');
    pollJob(job_id);
  } catch (err) {
    if (!current()) return;
    state.upload = null;
    setView('upload');
    showUploadError(err.message);
  }
}

async function pollJob(id) {
  const token = ++state.pollToken;
  let errors = 0;
  while (token === state.pollToken) {
    try {
      const job = await api('GET', `/api/jobs/${id}`);
      if (token !== state.pollToken) return;
      errors = 0;
      state.job = job;
      if (job.status === 'failed') return showFail(job.error || job.message || 'The analysis failed without an error message.');
      if (['needs_confirm', 'rendering', 'done'].includes(job.status)) return enterConfirm(false);
      renderJob(job);
    } catch (err) {
      if (token !== state.pollToken) return;
      if (err.status === 404) return showFail('The server no longer knows this job. It may have been restarted. Upload the clip again.');
      if (!isTransient(err)) { if (++errors >= 5) return showFail(err.message); }
      else if (++errors >= 5) return showFail('Showroom Coach cannot reach the server. Check that it is still running, then retry. Your job is kept.', () => { hideFail(); setMeter(state.job?.progress ?? 0, 'Reconnecting…'); pollJob(id); });
      else setMeter(state.job?.progress ?? 0, 'Connection lost — retrying…');
    }
    await sleep(POLL_MS);
  }
}

/* -------------------------------------------------------------- confirm */

async function enterConfirm(fromResults) {
  const epoch = state.epoch, id = state.jobId;
  try {
    const data = await api('GET', `/api/jobs/${id}/people`);
    if (epoch !== state.epoch) return;
    state.people = data.people || [];
    // Nobody detected: there is nothing to confirm, so go straight to one empty-state results page.
    if (!state.people.length) return showResults(await api('GET', `/api/jobs/${id}/results`), epoch);
  } catch (err) {
    if (epoch !== state.epoch) return;
    if (state.view === 'progress') {
      return showFail(err.message, isTransient(err) ? () => { hideFail(); pollJob(id); } : null);
    }
    return showNotice(isTransient(err) ? 'Connection lost. Check that Showroom Coach is still running, then try again.' : err.message);
  }
  state.draft = { roles: {}, names: {} };
  state.loadedNames = {};
  for (const p of state.people) {
    state.thumbs[p.pid] = p.thumb_url;
    state.draft.roles[p.pid] = p.role === 'staff' ? 'staff' : 'customer';
    state.draft.names[p.pid] = p.name || '';
    state.loadedNames[p.pid] = (p.name || '').trim();
  }
  renderBadges($('#confirmWarn'), state.job?.warnings);
  $('#people').replaceChildren(...state.people.map(personCard));
  $('#btnBackToResults').hidden = !fromResults;
  const btn = $('#btnConfirm');
  btn.disabled = false; btn.removeAttribute('aria-busy'); btn.textContent = 'Looks right →';
  updateCount();
  setView('confirm');
}

function suggestionText(s) {
  const pct = `${Math.round((s.confidence ?? 0) * 100)}%`;
  if (s.role === 'staff') return `Likely staff, ${pct}`;
  if (s.role === 'customer') return `Likely customer, ${pct}`;
  return `Unsure, ${pct}`;
}

function personCard(p) {
  const pid = p.pid, s = p.suggestion || {};
  const isStaff = () => state.draft.roles[pid] === 'staff';
  const sw = el('button', { type: 'button', class: 'switch', role: 'switch', 'aria-label': `Person ${pid} is staff`, 'aria-checked': String(isStaff()) },
    el('span', { text: 'Staff' }), el('span', { text: 'Customer' }));
  const input = el('input', {
    type: 'text', id: `name-${pid}`, maxlength: '40', autocomplete: 'off', placeholder: 'e.g. Aziz', value: state.draft.names[pid],
    oninput: (e) => { state.draft.names[pid] = e.target.value; },
  });
  const nameField = el('label', { class: 'name-field', for: `name-${pid}`, hidden: !isStaff() }, 'Name', input);
  const card = el('article', { class: `person${isStaff() ? ' is-staff' : ''}`, 'data-pid': pid },
    el('div', { class: 'person__photo' },
      el('img', { src: p.thumb_url, alt: `Person ${pid}, face blurred`, width: '112', height: '168' }),
      el('span', { class: 'person__tag', text: `Person ${pid}` })),
    el('div', { class: 'person__body' },
      el('p', { class: `chip chip--${s.role === 'staff' || s.role === 'customer' ? s.role : 'unsure'}`, text: suggestionText(s) }),
      el('ul', { class: 'reasons' }, (s.reasons || []).slice(0, 3).map((r) => el('li', { text: r }))),
      sw, nameField,
      el('p', { class: 'muted', text: `On screen ${mmss(p.t_first)} to ${mmss(p.t_last)}` })));
  sw.addEventListener('click', () => {
    state.draft.roles[pid] = isStaff() ? 'customer' : 'staff';     // one click flips the role
    sw.setAttribute('aria-checked', String(isStaff()));
    card.classList.toggle('is-staff', isStaff());
    nameField.hidden = !isStaff();
    updateCount();
  });
  return card;
}

function updateCount() {
  const roles = Object.values(state.draft.roles);
  const staff = roles.filter((r) => r === 'staff').length, customers = roles.length - staff;
  const box = $('#roleCount');
  box.replaceChildren(`${staff} staff, ${plural(customers, 'customer')}`);
  if (!staff) box.append(el('span', { class: 'warn', text: '. Mark at least one person as staff to get scorecards.' }));
}

async function submitRoles() {
  const btn = $('#btnConfirm');
  btn.disabled = true; btn.setAttribute('aria-busy', 'true'); btn.textContent = 'Scoring…';
  const body = { roles: {}, names: {} };
  for (const p of state.people) {
    body.roles[p.pid] = state.draft.roles[p.pid];
    // The server keeps ids that are left out, so a cleared name (or a staff member switched back to customer) must be sent as "".
    const name = state.draft.roles[p.pid] === 'staff' ? (state.draft.names[p.pid] || '').trim() : '';
    if (name !== (state.loadedNames[p.pid] || '')) body.names[p.pid] = name;
  }
  const epoch = state.epoch;
  try {
    const r = await api('POST', `/api/jobs/${state.jobId}/roles`, body);
    if (epoch !== state.epoch) return;
    await showResults(r, epoch);
  } catch (err) {
    if (epoch !== state.epoch) return;
    showNotice(err.message);
    btn.disabled = false; btn.removeAttribute('aria-busy'); btn.textContent = 'Looks right →';
  }
}

/* -------------------------------------------------------------- results */

async function showResults(r, epoch = state.epoch) {
  const nobody = !(r.persons && r.persons.length);
  if (!nobody && !Object.keys(state.thumbs).length) {
    try { for (const p of (await api('GET', `/api/jobs/${state.jobId}/people`)).people) state.thumbs[p.pid] = p.thumb_url; } catch { /* initials fall back */ }
  }
  if (epoch !== state.epoch) return;
  state.results = r;
  state.videoVersion++;
  setView('results');            // show the view before rendering: the timeline sizes its ticks from a visible box
  renderResults(r, nobody);
  if (!nobody && r.video_status !== 'ready' && r.video_status !== 'failed') watchVideo();
}

function renderResults(r, nobody) {
  $('#nobody').hidden = !nobody;
  $('#resultsBody').hidden = nobody;
  $('#btnChange').hidden = nobody;
  $('#resPreset').closest('.field').hidden = nobody;
  $('#salesTools').hidden = nobody;
  if (nobody) {
    $('#salesNote').hidden = true;
    state.tl = null; state.videoEl = null; state.videoToken++;
    $('#resMeta').textContent = `${state.job?.filename || 'Clip'}, ${mmss(r.duration)} long, no people detected`;
    renderBadges($('#resWarn'), (r.warnings || []).filter((w) => !w.startsWith('nobody detected')));   // the empty state already says it
    const ex = r.export || {};
    $('#exports').replaceChildren(el('span', { class: 'muted', text: 'Download' }),
      ...[['JSON', ex.json]].filter(([, href]) => href).map(([label, href]) => el('a', { href, download: '', text: label })));
    return;
  }
  const staffN = Object.values(r.roles || {}).filter((x) => x === 'staff').length;
  const custN = Object.keys(r.roles || {}).length - staffN;
  $('#resMeta').textContent = `${state.job?.filename || 'Clip'}, ${mmss(r.duration)} long, ${staffN} staff and ${plural(custN, 'customer')}`;
  $('#resPreset').value = r.preset;
  renderBadges($('#resWarn'), r.warnings);
  const ex = r.export || {};
  $('#exports').replaceChildren(el('span', { class: 'muted', text: 'Download' }),
    ...[['JSON', ex.json], ['Events CSV', ex.events_csv], ['Scorecard CSV', ex.scorecard_csv]]
      .filter(([, href]) => href).map(([label, href]) => el('a', { href, download: '', text: label })));
  renderVideo(r);
  renderTiles(r);
  renderAlerts(r);
  renderSheets(r);
  renderTimeline(r);
  renderScorecard(r);
  renderSalesNote(r);
}

function renderVideo(r) {
  const stage = $('#videoStage');
  stage.replaceChildren();
  state.videoEl = null;
  if (r.video_status === 'ready' && r.video_url) {
    const sep = r.video_url.includes('?') ? '&' : '?';
    const v = el('video', { controls: '', preload: 'metadata', playsinline: '', 'aria-label': 'Annotated video, faces blurred' });
    const sync = () => syncPlayhead();
    for (const ev of ['timeupdate', 'seeked', 'pause', 'loadedmetadata', 'ended']) v.addEventListener(ev, sync);
    v.addEventListener('play', () => { const tick = () => { syncPlayhead(); if (!v.paused && !v.ended) requestAnimationFrame(tick); }; tick(); });
    v.addEventListener('error', () => {
      if (state.videoEl !== v) return;
      state.videoEl = null;
      stage.replaceChildren(stageMessage('The video could not be played', 'Your browser could not open the annotated video. The timeline, alerts and scores below are still correct.'));
      syncPlayhead();
    });
    v.src = `${r.video_url}${sep}v=${state.videoVersion}`;
    stage.append(v);
    state.videoEl = v;
  } else if (r.video_status === 'failed') {
    stage.append(stageMessage('The annotated video could not be drawn', 'The timeline, alerts and scores below are still correct. Changing roles will try to draw it again.'));
  } else {
    stage.append(stageMessage('Drawing the annotated video…', 'Scores and the timeline are already up to date. The video appears here when it is ready.', true));
  }
  syncPlayhead();
}

function stageMessage(title, text, busy = false) {
  return el('div', { class: 'stage__msg', role: busy ? 'status' : null }, el('strong', { text: title }), el('p', { text }), busy ? el('div', { class: 'stage__bar' }) : null);
}

async function watchVideo() {
  const token = ++state.videoToken;
  let errors = 0;
  while (token === state.videoToken) {
    await sleep(POLL_MS);
    if (token !== state.videoToken) return;
    try {
      const job = await api('GET', `/api/jobs/${state.jobId}`);
      if (token !== state.videoToken) return;
      errors = 0;
      state.job = job;
      if (job.video_status === 'ready' || job.video_status === 'failed' || job.status === 'failed') {
        const r = await api('GET', `/api/jobs/${state.jobId}/results`);
        if (token !== state.videoToken) return;
        state.results = r;
        if (job.status === 'failed' && r.video_status !== 'ready') r.video_status = 'failed';
        renderVideo(r);
        renderSheets(r);
        return;
      }
    } catch (err) {
      if (++errors >= 5) { renderVideo({ ...state.results, video_status: 'failed' }); return; }
    }
  }
}

function seekTo(t) {
  const v = state.videoEl;
  if (!v) return toast('The annotated video is not ready yet. Try again in a moment.');
  v.currentTime = clamp(t, 0, Number.isFinite(v.duration) ? v.duration : t);
  v.play().catch(() => { /* autoplay refused: the seek still happened */ });
  const box = $('#videoStage').getBoundingClientRect();
  if (box.top < 0 || box.bottom > innerHeight) $('#videoStage').scrollIntoView({ block: 'nearest', behavior: reducedMotion() ? 'auto' : 'smooth' });
}

/* labels */
const custLabel = (pid) => `C${pid}`;
const staffName = (pid, names) => names?.[pid] || `S${pid}`;
const staffLabel = (pid, names) => (names?.[pid] ? `S${pid} ${names[pid]}` : `S${pid}`);
const list = (pids, names) => (pids || []).map((p) => staffName(p, names)).join(', ');

function renderTiles(r) {
  const s = r.scorecard?.store || {};
  const waits = s.wait_alerts ?? 0;
  const tiles = [
    ['Greet rate', s.greet_rate == null ? '—' : `${Math.round(s.greet_rate * 100)}%`,
      s.customers ? `${s.greeted ?? 0} of ${plural(s.customers, 'customer')} greeted` : 'No customers detected'],
    ['Median time to greet', mmss(s.median_time_to_greet),
      s.customers_entry_known ? `Measured on ${s.customers_entry_known} of ${s.customers} customers seen entering` : 'No customer was seen entering'],
    ['Walk-outs', String(s.walkouts ?? 0), 'Left without being greeted', (s.walkouts ?? 0) > 0],
    ['Unattended waiting', mmss(s.unattended_wait_s), `Customers waited while staff were free, ${plural(waits, 'long-wait alert')}`],
  ];
  $('#tiles').replaceChildren(...tiles.map(([name, value, note, bad]) =>
    el('div', { class: 'tile' }, el('p', { class: 'tile__name', text: name }),
      el('p', { class: `tile__value${bad ? ' is-bad' : ''}`, text: value }), el('p', { class: 'tile__note', text: note }))));
}

function waitedFor(r, pid) {
  const segs = r.timelines?.[pid]?.waiting || [];
  return segs.reduce((sum, [a, b]) => sum + (b - a), 0);
}

function buildAlerts(r) {
  const names = r.names, out = [];
  for (const e of r.events || []) {
    const d = e.data || {};
    if (e.type === 'wait_alert') {
      const free = list(d.free_staff, names);
      out.push({
        kind: 'wait', t: e.t,
        title: free ? `Missed greet — staff were free: ${free}` : 'Long wait — all staff were busy',
        detail: `Long wait: ${custLabel(e.customer)} had waited ${mmss(d.wait_s)} without a greeting`,
      });
    } else if (e.type === 'walkout') {
      const waited = waitedFor(r, e.customer);
      const free = list(d.free_staff, names);
      out.push({
        kind: 'walk', t: e.t, title: 'Walk-out — nobody greeted',
        detail: `${custLabel(e.customer)} left${waited ? ` after waiting ${mmss(waited)}` : ''}${free ? `. Free: ${free}` : ''}`,
      });
    } else if (e.type === 'missed_greet') {
      out.push({
        kind: 'wait', t: e.t, title: 'Missed greet',
        detail: `${custLabel(e.customer)} was not greeted${e.staff != null ? ` by ${staffName(e.staff, names)}` : ''}`,
      });
    }
  }
  return out.sort((a, b) => a.t - b.t);
}

function renderAlerts(r) {
  const alerts = buildAlerts(r);
  $('#alertCount').textContent = alerts.length ? String(alerts.length) : '';
  const box = $('#alerts');
  if (!alerts.length) {
    box.replaceChildren(el('li', { class: 'alerts__empty', text: 'No alerts in this clip.' }));
    return;
  }
  box.replaceChildren(...alerts.map((a) => el('li', {},
    el('button', { type: 'button', class: 'alert', title: `Jump to ${mmss(a.t)} in the video`, onclick: () => seekTo(a.t) },
      el('span', { class: 'alert__t', text: mmss(a.t) }),
      el('span', {}, el('span', { class: 'alert__title', text: a.title }), el('span', { class: 'alert__detail', text: a.detail }),
        el('span', { class: 'sr-only', text: `. Jump to ${mmss(a.t)} in the video` }))))));
}

/** Event contact sheets (three face-blurred frames around a greet, engagement, long wait or walk-out). The server lists
 *  them once the annotated video is ready; each opens full size in a new tab. Captions are set as text only. */
function renderSheets(r) {
  const sheets = Array.isArray(r.contact_sheets) ? r.contact_sheets : [];
  $('#sheetsPanel').hidden = !sheets.length;
  $('#sheetCount').textContent = sheets.length ? String(sheets.length) : '';
  $('#sheets').replaceChildren(...sheets.map((s) => el('li', {},
    el('a', { class: 'alert alert--info', href: s.url, target: '_blank', rel: 'noopener', title: `Open the ${s.caption} snapshot full size in a new tab` },
      el('span', { class: 'alert__t', text: mmss(s.t) }),
      el('span', {}, el('span', { class: 'alert__title', text: s.caption }),
        el('img', { class: 'sheet__img', src: s.url, alt: `${s.caption}: three frames around the event, faces blurred`, loading: 'lazy' }))))));
}

/* timeline ------------------------------------------------------------ */

const merge = (segs) => {
  const out = [];
  for (const s of [...segs].sort((a, b) => a[0] - b[0])) {
    if (out.length && s[0] <= out[out.length - 1][1] + 0.05) out[out.length - 1][1] = Math.max(out[out.length - 1][1], s[1]);
    else out.push([...s]);
  }
  return out;
};

/** One lane per person: customers come from `timelines`; anyone without an entry (staff) is derived. */
function buildLanes(r) {
  const eng = (r.events || []).filter((e) => e.type === 'engagement' && e.t_end != null);
  const greets = (r.events || []).filter((e) => e.type === 'greet');
  const lanes = (r.persons || []).map((p) => {
    const role = r.roles?.[p.pid] === 'staff' ? 'staff' : 'customer';
    const tl = r.timelines?.[p.pid];
    const mine = eng.filter((e) => e.staff === p.pid || e.customer === p.pid);
    return {
      pid: p.pid, role, t_first: p.t_first,
      present: tl?.present || [[p.t_first, p.t_last]],
      waiting: tl?.waiting || [],
      engaged: tl?.engaged || merge(mine.map((e) => [e.t, e.t_end])),
      greeted: tl ? tl.greeted : null,
      greetsGiven: role === 'staff' ? greets.filter((g) => g.staff === p.pid) : [],
    };
  });
  const rank = (l) => (l.role === 'staff' ? 0 : 1);
  return lanes.sort((a, b) => rank(a) - rank(b) || a.t_first - b.t_first || a.pid - b.pid);
}

const NICE_STEPS = [1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1800, 3600];

function renderTimeline(r) {
  const host = $('#timeline');
  const lanes = buildLanes(r);
  const dur = Math.max(r.duration || 0, ...lanes.flatMap((l) => l.present.map((s) => s[1]))) || 1;
  const pct = (t) => `${clamp(t / dur, 0, 1) * 100}%`;
  const maxTicks = clamp(Math.floor(((host.clientWidth || 900) * 0.8) / 58), 3, 12);   // keep labels from colliding on narrow screens
  const step = NICE_STEPS.find((s) => dur / s <= maxTicks) || 3600;
  const ticks = [];
  for (let t = 0; t <= dur + 1e-6; t += step) ticks.push(t);
  const names = r.names, events = r.events || [];

  const bars = (segs, cls) => segs.map(([a, b]) => el('i', { class: `bar bar--${cls}`, style: `left:${pct(a)};width:${((clamp(b, 0, dur) - clamp(a, 0, dur)) / dur) * 100}%` }));
  const mark = (kind, t, label) => el('button', {
    type: 'button', class: `mk mk--${kind}`, style: `left:${pct(t)}`, title: `${label} at ${mmss(t)}`,
    'aria-label': `${label} at ${mmss(t)}. Jump to this moment in the video`, onclick: () => seekTo(t),
  });

  const laneEl = (l) => {
    const who = l.role === 'staff' ? staffLabel(l.pid, names) : custLabel(l.pid);
    const thumb = state.thumbs[l.pid];
    const avatar = thumb ? el('img', { class: 'avatar', src: thumb, alt: '' }) : el('span', { class: 'avatar', text: who.slice(0, 1) });
    const marks = [];
    if (l.role === 'customer' && l.greeted != null) marks.push(mark('greet', l.greeted, `${who} greeted`));
    for (const g of l.greetsGiven) marks.push(mark('greet', g.t, `${who} greeted ${custLabel(g.customer)}`));
    for (const e of events) {
      if (e.customer !== l.pid || l.role !== 'customer') continue;
      if (e.type === 'wait_alert') marks.push(mark('wait', e.t, `${who} long wait while staff free`));
      if (e.type === 'walkout') marks.push(mark('walk', e.t, `${who} walked out`));
    }
    return el('div', { class: `tl__row tl__row--${l.role}` },
      el('div', { class: 'tl__who', title: who }, avatar, el('span', { text: who })),
      el('div', { class: 'tl__track' }, bars(l.present, 'present'), bars(l.waiting, 'waiting'), bars(l.engaged, 'engaged'), marks));
  };

  const scrub = el('div', {
    class: 'tl__scrub', role: 'slider', tabindex: '0', 'aria-label': 'Video position on the timeline. Arrow keys move 5 seconds',
    'aria-valuemin': '0', 'aria-valuemax': String(Math.round(dur)), 'aria-valuenow': '0', 'aria-valuetext': '0:00',
  });
  const head = el('div', { class: 'tl__head', 'data-label': '0:00', hidden: true });
  const hoverLabel = el('b');
  const hover = el('div', { class: 'tl__hover', hidden: true }, hoverLabel);
  const at = (e) => { const b = scrub.getBoundingClientRect(); return clamp((e.clientX - b.left) / b.width, 0, 1) * dur; };
  scrub.addEventListener('pointermove', (e) => { const t = at(e); hover.hidden = false; hover.style.left = pct(t); hoverLabel.textContent = mmss(t); });
  scrub.addEventListener('pointerleave', () => { hover.hidden = true; });
  scrub.addEventListener('click', (e) => seekTo(at(e)));
  scrub.addEventListener('keydown', (e) => {
    const v = state.videoEl;
    const now = v ? v.currentTime : 0;
    const to = { ArrowLeft: now - 5, ArrowRight: now + 5, Home: 0, End: dur }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    seekTo(to);
  });

  const grid = el('div', { class: 'tl__grid', 'aria-hidden': 'true' }, ticks.map((t) => el('i', { class: 'tl__line', style: `left:${pct(t)}` })));
  host.replaceChildren(el('div', { class: 'tl' },
    el('div', { class: 'tl__axis' }, el('div'), el('div', { class: 'tl__ticks' }, ticks.map((t) => el('span', { class: 'tl__tick', style: `left:${pct(t)}`, text: mmss(t) })))),
    el('div', { class: 'tl__body' }, el('div', { class: 'tl__plot' }, grid, scrub, hover, head), lanes.map(laneEl))));
  state.tl = { dur, head, scrub, pct };
  syncPlayhead();
}

function syncPlayhead() {
  const tl = state.tl;
  if (!tl) return;
  const v = state.videoEl;
  tl.head.hidden = !v;
  tl.scrub.toggleAttribute('data-off', !v);
  if (!v) return;
  const t = v.currentTime || 0;
  tl.head.style.left = tl.pct(t);
  tl.head.dataset.label = mmss(t);
  tl.scrub.setAttribute('aria-valuenow', String(Math.round(t)));
  tl.scrub.setAttribute('aria-valuetext', mmss(t));
}

/* scorecard ----------------------------------------------------------- */

function renderScorecard(r) {
  const host = $('#scorecard');
  const rows = r.scorecard?.staff || [];
  if (!rows.length) {
    host.replaceChildren(el('div', { class: 'empty' },
      el('p', { text: 'Nobody is marked as staff, so there are no scorecards yet. Mark your salespeople to see greeting and engagement numbers for each of them.' }),
      el('button', { type: 'button', class: 'btn btn--primary', onclick: () => enterConfirm(true), text: 'Change roles' })));
    return;
  }
  const sales = rows.some((s) => s.sold != null || s.conversion != null);
  const cols = [
    ['Greets', (s) => s.greets],
    ['Median time to greet', (s) => mmss(s.median_time_to_greet)],
    ['Customers engaged', (s) => s.customers_engaged],
    ['Time engaged', (s) => mmss(s.engaged_total_s)],
    ['Median engagement', (s) => mmss(s.engaged_median_s)],
    ['Missed while free', (s) => s.missed_while_free, (s) => s.missed_while_free > 0],
    ['Walk-outs while free', (s) => s.walkouts_while_free, (s) => s.walkouts_while_free > 0],
    ...(sales ? [['Sold', (s) => s.sold ?? '—'], ['Conversion', (s) => (s.conversion == null ? '—' : `${Math.round(s.conversion * 100)}%`)]] : []),
  ];
  host.replaceChildren(el('div', { class: 'table-wrap' }, el('table', {},
    el('caption', { class: 'sr-only', text: 'Scorecard for each salesperson' }),
    el('thead', {}, el('tr', {}, el('th', { scope: 'col', text: 'Salesperson' }), cols.map((c) => el('th', { scope: 'col', text: c[0] })))),
    el('tbody', {}, rows.map((s) => el('tr', {},
      el('th', { scope: 'row' }, el('span', { class: 'who' }, el('i', { class: 'dot' }),
        s.name ? [s.name, el('small', { text: `S${s.staff}` })] : `S${s.staff}`)),
      cols.map(([, get, bad]) => el('td', { class: `num${bad && bad(s) ? ' is-bad' : ''}`, text: String(get(s) ?? '—') }))))))));
}

/* sales CSV (optional) ------------------------------------------------ */

/** The button label, the Remove button and the one-line summary of what the server read from the sales CSV. */
function renderSalesNote(r) {
  const s = r.sales, note = $('#salesNote');
  $('#btnSales').textContent = s ? 'Replace sales CSV' : 'Add sales CSV (optional)';
  $('#btnSalesClear').hidden = !s;
  note.hidden = !s;
  if (!s) { note.textContent = ''; return; }
  const parts = [`Sales CSV: ${plural(s.rows, 'row')} read`];
  if (s.skipped) parts.push(`${plural(s.skipped, 'row')} skipped as invalid`);
  const lost = s.unmatched || [];
  if (lost.length) {
    const more = lost.length > 5 ? ` and ${lost.length - 5} more` : '';
    parts.push(`Not matched to a salesperson: ${lost.slice(0, 5).join(', ')}${more} (names must match the ones typed on the confirm page, or use S<number>)`);
  }
  note.textContent = `${parts.join('. ')}.`;
}

/** Send a sales CSV (or `null` to remove it) and redraw the scorecard from the answer. */
async function sendSales(file) {
  const epoch = state.epoch, id = state.jobId;
  if (file && file.size > MAX_SALES_BYTES) return showNotice(`${file.name} is larger than 1 MB. A sales CSV should be much smaller than that.`);
  const buttons = [$('#btnSales'), $('#btnSalesClear')];
  for (const b of buttons) b.disabled = true;
  try {
    let r;
    if (file) {
      const fd = new FormData();
      fd.append('file', file);
      r = await api('POST', `/api/jobs/${id}/sales`, fd);
    } else {
      r = await api('DELETE', `/api/jobs/${id}/sales`);
    }
    if (epoch !== state.epoch) return;
    showNotice('');
    state.results = r;
    renderScorecard(r);
    renderSalesNote(r);
    toast(file ? 'Sales CSV added' : 'Sales CSV removed');
  } catch (err) {
    if (epoch === state.epoch) showNotice(err.message);
  } finally {
    for (const b of buttons) b.disabled = false;
  }
}

function bindSales() {
  const input = $('#salesFile');
  $('#btnSales').addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    const file = input.files[0];
    input.value = '';                               // the same file can be chosen again after a fix
    if (file) sendSales(file);
  });
  $('#btnSalesClear').addEventListener('click', () => sendSales(null));
}

/* ----------------------------------------------------------------- init */

function bindResults() {
  let resizeTimer = 0;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.view === 'results' && state.results?.persons?.length) renderTimeline(state.results); }, 150);
  });
  $('#btnConfirm').addEventListener('click', submitRoles);
  $('#btnChange').addEventListener('click', () => enterConfirm(true));
  $('#btnBackToResults').addEventListener('click', () => setView('results'));
  $('#resPreset').addEventListener('change', async (e) => {
    const sel = e.target;
    sel.disabled = true;
    const epoch = state.epoch;
    try {
      const r = await api('POST', `/api/jobs/${state.jobId}/roles`, { preset: sel.value });
      if (epoch === state.epoch) await showResults(r, epoch);
    } catch (err) {
      if (epoch === state.epoch) { showNotice(err.message); sel.value = state.results?.preset || sel.value; }
    } finally { sel.disabled = false; }
  });
  $('#btnNobodyRetry').addEventListener('click', resetToUpload);
  $('#btnReconnect').addEventListener('click', () => { if (state.reconnect) state.reconnect(); });
  // Under <base href="/static/"> a "#main" link would navigate to /static/#main and reset the app, so handle it here.
  $('.skip').addEventListener('click', (e) => { e.preventDefault(); $('#main').focus(); });
}

/** Connection loss while resuming: keep the job (and #job= in the URL) and offer a Retry. */
function lostContact(id) {
  $('#progFile').textContent = 'this clip';
  renderStages(1);
  setView('progress');
  showFail('Showroom Coach cannot reach the server. Check that it is still running, then retry. Your job is kept.', () => { hideFail(); resume(id); });
}

/** Reload-proof demo: the job id lives in the URL hash, so F5 returns to wherever the job is. */
async function resume(id) {
  const epoch = state.epoch;
  state.jobId = id;
  const forget = () => { history.replaceState(null, '', location.pathname + location.search); state.jobId = null; };
  try {
    const job = await api('GET', `/api/jobs/${id}`);
    if (epoch !== state.epoch) return;
    state.job = job;
    $('#progFile').textContent = job.filename || 'clip';
    hideFail();
    if (job.status === 'failed') { setView('progress'); renderStages(1); return showFail(job.error || job.message || 'The analysis failed.'); }
    if (job.status === 'needs_confirm') return enterConfirm(false);
    if (job.status === 'rendering' || job.status === 'done') {
      const r = await api('GET', `/api/jobs/${id}/results`);
      return showResults(r, epoch);
    }
    renderStages(1); setView('progress'); pollJob(id);
  } catch (err) {
    if (epoch !== state.epoch) return;
    if (isTransient(err)) lostContact(id); else forget();    // only an unknown/unusable job clears the URL
  }
}

async function init() {
  if (MOCK) $('.brand').append(el('span', { class: 'mockflag', text: 'Demo data' }));
  bindUpload();
  bindResults();
  bindSales();
  $('#analyseHint').textContent = 'Choose a video to start.';
  setView('upload');
  await loadPresets();
  const m = /#job=([\w-]+)/.exec(location.hash);
  if (m && !MOCK) resume(m[1]);
}

init();
})();
