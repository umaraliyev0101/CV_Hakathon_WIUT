// Live mode: webcam -> WebSocket -> server overlays. Plain ES module, no dependencies.
import { t, onLangChange } from "/static/i18n.js";
const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), grab = $("grab");
const octx = overlay.getContext("2d");
// Privacy: the raw <video> is never visible (CSS keeps it hidden but playing, for frame grabs). The
// preview is the exact frame the server analysed (shownFrame) with detected heads pixelated from
// that frame's own overlay. pendingFrame holds the copy of the frame currently in flight.
const pendingFrame = document.createElement("canvas");
const shownFrame = document.createElement("canvas");
let haveShown = false;
function copyFrame(from, to) {
  to.width = from.width; to.height = from.height;
  to.getContext("2d").drawImage(from, 0, 0);
}
const SEND_MS = 200;            // ~5 fps, only when no frame is in flight
const STUCK_MS = 6000;          // give up waiting for a reply and send again
const COLORS = { customer: "#ff8c00", staff: "#3c78b4", alert: "#e5182b" };

const S = {
  ws: null, open: false, attempt: 0, retryTimer: null,
  inFlight: false, sentAt: 0,
  stream: null, running: false,
  zones: [], last: null,
  drawing: false, drag: null,
};

function notice(msg, ms = 6000) {
  const n = $("notice");
  n.textContent = msg; n.hidden = !msg;
  if (msg && ms) setTimeout(() => { if (n.textContent === msg) n.hidden = true; }, ms);
}

// ------------------------------------------------------------------ socket
function wsUrl() {
  return (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/live";
}

let connText = () => t("conn.not_connected");      // re-run on a language switch
function setConn(fn) { connText = fn; $("conn").textContent = fn(); }

function connect() {
  clearTimeout(S.retryTimer);
  setConn(S.attempt ? () => t("conn.reconnecting") : () => t("conn.connecting"));
  const ws = new WebSocket(wsUrl());
  S.ws = ws;
  ws.onopen = () => {
    S.open = true; S.attempt = 0; S.inFlight = false;
    setConn(() => t("conn.connected"));
    ws.send(JSON.stringify({ type: "preset", name: $("preset").value }));
    if (S.zones.length) ws.send(JSON.stringify({ type: "zones", zones: S.zones }));
  };
  ws.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    if (m.type === "overlay") { S.inFlight = false; onOverlay(m); }
    else if (m.type === "zones") { S.zones = m.zones; drawAll(); renderZoneStats(); updateButtons(); }
    else if (m.type === "error") { S.inFlight = false; notice(m.message); }
  };
  ws.onclose = () => {
    if (S.ws !== ws) return;
    S.open = false; S.inFlight = false;
    const wait = Math.min(8000, 500 * 2 ** S.attempt);
    S.attempt += 1;
    setConn(() => t("conn.disconnected", { s: Math.round(wait / 100) / 10 }));
    S.retryTimer = setTimeout(connect, wait);
  };
  ws.onerror = () => { try { ws.close(); } catch { /* already closing */ } };
}

function sendControl(obj) {
  if (S.open) S.ws.send(JSON.stringify(obj));
}

// ------------------------------------------------------------------ camera and frames
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    notice(t("live.err_no_camera_api"), 0);
    return;
  }
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960 }, audio: false });
  } catch (e) {
    notice(e && e.name === "NotAllowedError" ? t("live.err_denied") : t("live.err_busy"), 0);
    return;
  }
  video.srcObject = S.stream;
  await video.play();
  await new Promise((r) => (video.videoWidth ? r() : video.addEventListener("loadedmetadata", r, { once: true })));
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  $("start").disabled = true;
  S.running = true;
  updateButtons();
  setInterval(pump, SEND_MS);
  requestAnimationFrame(redrawLoop);
}

function pump() {
  if (!S.running || S.drawing || !S.open || !video.videoWidth) return;
  const now = performance.now();
  if (S.inFlight && now - S.sentAt < STUCK_MS) return;
  grab.width = video.videoWidth; grab.height = video.videoHeight;
  grab.getContext("2d").drawImage(video, 0, 0);
  copyFrame(grab, pendingFrame);
  S.inFlight = true; S.sentAt = now;
  grab.toBlob((blob) => {
    if (blob && S.open) S.ws.send(blob); else S.inFlight = false;
  }, "image/jpeg", 0.7);
}

// ------------------------------------------------------------------ overlay
function onOverlay(m) {
  S.last = m;
  if (pendingFrame.width) { copyFrame(pendingFrame, shownFrame); haveShown = true; $("stage-hint").hidden = true; }
  $("fps").textContent = m.fps.toFixed(1);
  $("dropped").textContent = String(m.dropped);
  $("occupancy").textContent = String(m.occupancy);
  renderZoneStats();
  const b = $("alert-banner");
  if (m.alerts.length) {
    b.textContent = m.alerts.map((a) => a.message).join(" | ");
    b.hidden = false;
  } else b.hidden = true;
}

function renderZoneStats() {
  const ul = $("zone-stats");
  ul.textContent = "";
  if (!S.zones.length) {
    const li = document.createElement("li");
    li.className = "muted"; li.textContent = t("live.no_zones");
    ul.append(li); return;
  }
  const byId = new Map(((S.last && S.last.zones) || []).map((z) => [z.zone_id, z]));
  for (const z of S.zones) {
    const c = byId.get(z.zone_id) || { now: 0, total: 0 };
    const li = document.createElement("li");
    li.textContent = t("live.zone_stat", { name: z.name, now: c.now, total: c.total });
    ul.append(li);
  }
}

function redrawLoop() { drawAll(); requestAnimationFrame(redrawLoop); }

function pixelateHead(x1, y1, x2, y2, sx, sy) {
  // privacy: the local preview never shows a legible face; the head region is pixelated
  const w = x2 - x1, h = y2 - y1;
  const hx = (x1 + w * 0.2) * sx, hy = y1 * sy, hw = w * 0.6 * sx, hh = h * 0.2 * sy;
  if (hw < 2 || hh < 2) return;
  const tmp = pixelateHead.tmp || (pixelateHead.tmp = document.createElement("canvas"));
  tmp.width = 6; tmp.height = 6;
  tmp.getContext("2d").drawImage(shownFrame, hx, hy, hw, hh, 0, 0, 6, 6);
  octx.imageSmoothingEnabled = false;
  octx.drawImage(tmp, 0, 0, 6, 6, hx, hy, hw, hh);
  octx.imageSmoothingEnabled = true;
}

function label(text, x, y, color) {
  octx.font = "600 14px system-ui, sans-serif";
  const w = octx.measureText(text).width + 8;
  octx.fillStyle = color; octx.fillRect(x, y - 18, w, 18);
  octx.fillStyle = "#fff"; octx.fillText(text, x + 4, y - 4);
}

function drawAll() {
  const W = overlay.width, H = overlay.height;
  if (!W) return;
  octx.clearRect(0, 0, W, H);
  octx.lineWidth = 2;
  if (haveShown) {
    octx.drawImage(shownFrame, 0, 0, W, H);
    const m0 = S.last;      // pixelate every detected head from the overlay computed on this very frame
    if (m0) for (const p of m0.persons) {
      const [x1, y1, x2, y2] = p.bbox;
      pixelateHead(x1, y1, x2, y2, W / m0.frame_size[0], H / m0.frame_size[1]);
    }
  } else { octx.fillStyle = "#000"; octx.fillRect(0, 0, W, H); }
  const byId = new Map(((S.last && S.last.zones) || []).map((z) => [z.zone_id, z]));
  for (const z of S.zones) {
    octx.strokeStyle = "#1b9e77"; octx.fillStyle = "rgba(27,158,119,.12)";
    octx.beginPath();
    z.polygon.forEach(([x, y], i) => (i ? octx.lineTo(x * W, y * H) : octx.moveTo(x * W, y * H)));
    octx.closePath(); octx.fill(); octx.stroke();
    const c = byId.get(z.zone_id) || { now: 0, total: 0 };
    label(t("live.zone_label", { name: z.name, now: c.now, total: c.total }), z.polygon[0][0] * W + 2, z.polygon[0][1] * H + 20, "#1b9e77");
  }
  const m = S.last;
  if (m && !S.drawing) {
    const sx = W / m.frame_size[0], sy = H / m.frame_size[1];
    for (const p of m.persons) {
      const [x1, y1, x2, y2] = p.bbox;
      const color = p.alert ? COLORS.alert : COLORS[p.role] || COLORS.customer;
      octx.strokeStyle = color; octx.lineWidth = p.alert ? 4 : 2;
      octx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      let text = `#${p.person_id} ${p.role === "staff" ? t("role.staff") : t("role.customer")}`;
      if (p.zone_name) text += ` | ${t("live.person_zone", { zone: p.zone_name, s: Math.round(p.dwell) })}`;
      label(text, x1 * sx, y1 * sy, color);
    }
  }
  if (S.drag) {
    const { x0, y0, x1, y1 } = S.drag;
    octx.strokeStyle = "#1b9e77"; octx.setLineDash([6, 4]);
    octx.strokeRect(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H);
    octx.setLineDash([]);
  }
}

// ------------------------------------------------------------------ interaction
function norm(ev) {
  const r = overlay.getBoundingClientRect();
  return [Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)),
          Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height))];
}

overlay.addEventListener("pointerdown", (ev) => {
  const [x, y] = norm(ev);
  if (S.drawing) { S.drag = { x0: x, y0: y, x1: x, y1: y }; overlay.setPointerCapture(ev.pointerId); }
  else sendControl({ type: "toggle", x, y });
});
overlay.addEventListener("pointermove", (ev) => {
  if (S.drag) { [S.drag.x1, S.drag.y1] = norm(ev); }
});
overlay.addEventListener("pointerup", () => {
  const d = S.drag; S.drag = null;
  if (!d) return;
  const [x0, x1] = [Math.min(d.x0, d.x1), Math.max(d.x0, d.x1)];
  const [y0, y1] = [Math.min(d.y0, d.y1), Math.max(d.y0, d.y1)];
  if ((x1 - x0) * (y1 - y0) < 0.002) { notice(t("live.err_small")); return; }
  const name = $("zone-name").value.trim() || `Zone ${S.zones.length + 1}`;
  let id = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "zone";
  while (S.zones.some((z) => z.zone_id === id)) id += "-2";
  S.zones.push({ zone_id: id, name, kind: $("zone-kind").value,
                 polygon: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], source: "user" });
  sendControl({ type: "zones", zones: S.zones });
  renderZoneStats(); updateButtons();
});

function setDrawing(on) {
  S.drawing = on;
  $("draw").setAttribute("aria-pressed", String(on));
  $("draw").textContent = on ? t("live.done_drawing") : t("live.draw");
  overlay.style.cursor = on ? "crosshair" : "pointer";
  if (on) video.pause(); else video.play();      // pump() stops while drawing; the shown frame stays pixelated
  if (on) notice(t("live.drawing_hint"), 8000);
}

function updateButtons() {
  $("draw").disabled = !S.running;
  $("clear-zones").disabled = !S.zones.length;
}

$("start").addEventListener("click", startCamera);
$("draw").addEventListener("click", () => setDrawing(!S.drawing));
$("clear-zones").addEventListener("click", () => {
  S.zones = []; sendControl({ type: "zones", zones: [] });
  renderZoneStats(); updateButtons();
});
$("load-zones").addEventListener("click", () => {
  const path = $("zone-path").value.trim();
  if (!path) { notice(t("live.err_path")); return; }
  if (!S.open) { notice(t("live.err_not_connected")); return; }
  sendControl({ type: "load_zones", path });
});
$("preset").addEventListener("change", () => sendControl({ type: "preset", name: $("preset").value }));

onLangChange(() => {
  setConn(connText);
  $("draw").textContent = S.drawing ? t("live.done_drawing") : t("live.draw");
  renderZoneStats();
});

connect();
updateButtons();
