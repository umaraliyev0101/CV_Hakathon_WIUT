// Zone editor: a canvas over the keyframe. All geometry is stored normalised to [0,1]; pixels only
// exist while drawing and hit-testing, so the canvas follows the displayed image size.
const KIND_COLOURS = { car: "#d9650b", desk: "#1f5f8b", entrance: "#1b6e3c", lounge: "#6b3fa0", other: "#6a645d" };
import { t } from "/static/i18n.js";
const kindLabel = (k) => ({ car: t("kind.car"), desk: t("kind.desk"), entrance: t("kind.entrance"), lounge: t("kind.lounge"), other: t("kind.other") }[k] || k);
const HANDLE = 6, HIT = 11, MIN_RECT = 0.01;
const clamp01 = (v) => Math.min(1, Math.max(0, v));

export function pointInPolygon(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export class ZoneEditor {
  constructor(els) {
    this.els = els;                  // {img, canvas, list, form, name, kind, hint, error, btn:{select,rect,poly,finish,del}}
    this.ctx = els.canvas.getContext("2d");
    this.zones = [];
    this.selected = null;            // index
    this.mode = "select";            // select | rect | poly
    this.draft = null;               // {type:'rect', a, b} | {type:'poly', pts, hover}
    this.drag = null;                // {zi, vi}
    this.frameSize = null;
    this.floor = { img: [], world: [] };   // optional floor calibration: 4 clicked points + their metres
    this._bind();
    this.updateHint();
  }

  // language switch: redraw the parts built from translated text
  refreshLang() { this.updateHint(); this.renderList(); this.renderFloor(); }

  async load(zonesDoc, imgUrl) {
    this.frameSize = zonesDoc.frame_size;
    this.zones = zonesDoc.zones.map((z) => ({ ...z, polygon: z.polygon.map((p) => [p[0], p[1]]) }));
    const f = zonesDoc.floor;
    this.floor = f ? { img: f.img.map((p) => [p[0], p[1]]), world: f.world.map((p) => [String(p[0]), String(p[1])]) }
                   : { img: [], world: [] };
    this.selected = null; this.draft = null; this.setMode("select"); this.showError(""); this.renderFloor();
    await new Promise((res) => {
      this.els.img.onload = this.els.img.onerror = res;
      this.els.img.src = imgUrl;
    });
    this.resize(); this.refreshUi();
  }

  toDoc() {
    const doc = { version: 1, frame_size: this.frameSize, zones: this.zones.map((z) => ({
      zone_id: z.zone_id, name: z.name, kind: z.kind, polygon: z.polygon, source: z.source || "user" })) };
    const n = this.floor.img.length;
    if (n) {
      const world = this.floor.world.map((p) => [parseFloat(p[0]), parseFloat(p[1])]);
      if (world.some((p) => p.some((v) => Math.abs(v) > 1000))) throw new Error(t("floor.err_range"));
      if (n !== 4 || world.slice(0, 4).some((p) => !p.every(Number.isFinite)))
        throw new Error(t("floor.err_incomplete"));
      doc.floor = { img: this.floor.img, world };
    }
    return doc;
  }

  // ---------------------------------------------------------------- floor calibration
  clearFloor() { this.floor = { img: [], world: [] }; this.renderFloor(); this.draw(); }
  renderFloor() {
    const rows = this.els.floorRows; if (!rows) return;
    rows.textContent = "";
    this.floor.img.forEach((p, i) => {
      const li = document.createElement("li");
      li.append(t("floor.point", { n: i + 1 }));
      [0, 1].forEach((k) => {
        const inp = document.createElement("input");
        inp.type = "number"; inp.step = "any"; inp.size = 5; inp.value = this.floor.world[i][k];
        inp.setAttribute("aria-label", k ? t("floor.point_y_aria", { n: i + 1 }) : t("floor.point_x_aria", { n: i + 1 }));
        inp.addEventListener("input", () => {
          this.floor.world[i][k] = inp.value;
          const v = parseFloat(inp.value);
          this.showError(Number.isFinite(v) && Math.abs(v) > 1000 ? t("floor.err_range") : "");
        });
        li.append(inp, k ? t("floor.m") : t("floor.m_y"));
      });
      rows.append(li);
    });
  }

  showError(msg) {
    this.els.error.textContent = msg; this.els.error.hidden = !msg;
  }

  // ---------------------------------------------------------------- geometry
  resize() {
    const { img, canvas } = this.els;
    const w = img.clientWidth || 640, h = img.clientHeight || 360, dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    this.w = w; this.h = h;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }
  toNorm(ev) {
    const r = this.els.canvas.getBoundingClientRect();
    return [clamp01((ev.clientX - r.left) / r.width), clamp01((ev.clientY - r.top) / r.height)];
  }
  px(p) { return [p[0] * this.w, p[1] * this.h]; }

  hitVertex(n) {
    // selected zone's vertices first, so overlapping zones stay editable
    const order = this.zones.map((_, i) => i);
    if (this.selected !== null) order.unshift(this.selected);
    for (const zi of order) {
      const poly = this.zones[zi].polygon;
      for (let vi = 0; vi < poly.length; vi++) {
        const a = this.px(poly[vi]), b = this.px(n);
        if (Math.hypot(a[0] - b[0], a[1] - b[1]) <= HIT && (zi === this.selected)) return { zi, vi };
      }
    }
    return null;
  }
  hitZone(n) {
    for (let i = this.zones.length - 1; i >= 0; i--) if (pointInPolygon(n, this.zones[i].polygon)) return i;
    return null;
  }

  // ---------------------------------------------------------------- interaction
  _bind() {
    const c = this.els.canvas, b = this.els.btn;
    c.addEventListener("pointerdown", (e) => this.onDown(e));
    c.addEventListener("pointermove", (e) => this.onMove(e));
    c.addEventListener("pointerup", (e) => this.onUp(e));
    c.addEventListener("dblclick", (e) => this.onDbl(e));
    window.addEventListener("resize", () => this.resize());
    if (window.ResizeObserver) new ResizeObserver(() => this.resize()).observe(this.els.img);
    b.select.addEventListener("click", () => this.setMode("select"));
    b.rect.addEventListener("click", () => this.setMode("rect"));
    b.poly.addEventListener("click", () => this.setMode("poly"));
    b.finish.addEventListener("click", () => this.finishPoly());
    b.del.addEventListener("click", () => this.deleteSelected());
    b.calib?.addEventListener("click", () => this.setMode("calib"));
    b.clearFloor?.addEventListener("click", () => this.clearFloor());
    this.els.name.addEventListener("input", () => {
      if (this.selected === null) return;
      this.zones[this.selected].name = this.els.name.value; this.renderList(); this.draw();
    });
    this.els.kind.addEventListener("change", () => {
      if (this.selected === null) return;
      this.zones[this.selected].kind = this.els.kind.value; this.renderList(); this.draw();
    });
    document.addEventListener("keydown", (e) => {
      if (this.els.panel.hidden) return;
      const tag = (e.target.tagName || "").toLowerCase();
      if (["input", "select", "textarea", "button"].includes(tag)) return;   // never act while a control has focus
      if (e.key === "Escape") { this.draft = null; this.setMode("select"); this.draw(); return; }
      if (e.key === "Delete" && this.selected !== null) { e.preventDefault(); this.deleteSelected(); }
    });
  }

  setMode(mode) {
    this.mode = mode; this.draft = null;
    const b = this.els.btn;
    b.select.setAttribute("aria-pressed", String(mode === "select"));
    b.rect.setAttribute("aria-pressed", String(mode === "rect"));
    b.poly.setAttribute("aria-pressed", String(mode === "poly"));
    b.calib?.setAttribute("aria-pressed", String(mode === "calib"));
    b.finish.hidden = mode !== "poly";
    this.updateHint();
    this.els.canvas.style.cursor = mode === "select" ? "default" : "crosshair";
    this.draw();
  }

  updateHint() {
    this.els.hint.textContent = { select: t("hint.select"), rect: t("hint.rect"), calib: t("hint.calib"), poly: t("hint.poly") }[this.mode];
  }

  onDown(e) {
    const n = this.toNorm(e);
    this.els.canvas.setPointerCapture?.(e.pointerId);
    if (this.mode === "calib") {
      if (this.floor.img.length < 4) {
        this.floor.img.push(n); this.floor.world.push(["", ""]); this.renderFloor(); this.draw();
      } else this.showError(t("floor.err_full"));
      return;
    }
    if (this.mode === "rect") { this.draft = { type: "rect", a: n, b: n }; return; }
    if (this.mode === "poly") {
      if (!this.draft) this.draft = { type: "poly", pts: [], hover: n };
      const last = this.draft.pts[this.draft.pts.length - 1];
      if (!last || Math.hypot((last[0] - n[0]) * this.w, (last[1] - n[1]) * this.h) > 3) this.draft.pts.push(n);
      this.draw(); return;
    }
    const v = this.hitVertex(n);
    if (v) { this.drag = v; return; }
    this.select(this.hitZone(n));
  }
  onMove(e) {
    const n = this.toNorm(e);
    if (this.drag) {
      this.zones[this.drag.zi].polygon[this.drag.vi] = n; this.draw(); return;
    }
    if (this.draft?.type === "rect") { this.draft.b = n; this.draw(); }
    else if (this.draft?.type === "poly") { this.draft.hover = n; this.draw(); }
  }
  onUp(e) {
    if (this.drag) { this.drag = null; return; }
    if (this.draft?.type === "rect") {
      const { a, b } = this.draft; this.draft = null;
      if (Math.abs(a[0] - b[0]) > MIN_RECT && Math.abs(a[1] - b[1]) > MIN_RECT) {
        const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
        this.addZone([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
      }
      this.setMode("select");
    }
  }
  onDbl() { if (this.mode === "poly") this.finishPoly(); }

  finishPoly() {
    const pts = this.draft?.pts || [];
    if (pts.length < 3) { this.showError(t("zone.err_polygon")); return; }
    this.showError("");
    this.draft = null; this.addZone(pts); this.setMode("select");
  }

  addZone(polygon) {
    let k = this.zones.length + 1;
    const ids = new Set(this.zones.map((z) => z.zone_id));
    while (ids.has(`zone_${k}`)) k++;
    this.zones.push({ zone_id: `zone_${k}`, name: `Zone ${k}`, kind: "car", polygon, source: "user" });
    this.select(this.zones.length - 1);
  }
  deleteSelected() {
    if (this.selected === null) return;
    this.zones.splice(this.selected, 1); this.select(null);
  }
  select(i) { this.selected = i; this.refreshUi(); this.draw(); }

  // ---------------------------------------------------------------- UI sync + drawing
  refreshUi() {
    const z = this.selected === null ? null : this.zones[this.selected];
    this.els.form.hidden = !z;
    this.els.btn.del.disabled = !z;
    if (z) { this.els.name.value = z.name; this.els.kind.value = z.kind; }
    this.renderList();
  }
  renderList() {
    const ul = this.els.list; ul.textContent = "";
    if (!this.zones.length) {
      const li = document.createElement("li"); li.className = "muted"; li.textContent = t("zones.none");
      ul.append(li);
    }
    this.zones.forEach((z, i) => {
      const li = document.createElement("li"), b = document.createElement("button");
      b.type = "button"; if (i === this.selected) b.setAttribute("aria-current", "true");
      const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = KIND_COLOURS[z.kind] || "#888";
      b.append(sw, `${z.name || t("zone.unnamed")} (${kindLabel(z.kind)})`);
      b.addEventListener("click", () => this.select(i));
      li.append(b); ul.append(li);
    });
  }
  draw() {
    const g = this.ctx; if (!g || !this.w) return;
    g.clearRect(0, 0, this.w, this.h);
    this.zones.forEach((z, i) => {
      const col = KIND_COLOURS[z.kind] || "#888", sel = i === this.selected;
      g.beginPath();
      z.polygon.forEach((p, k) => { const [x, y] = this.px(p); k ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.closePath();
      g.fillStyle = col + (sel ? "55" : "33"); g.fill();
      g.lineWidth = sel ? 3 : 2; g.strokeStyle = col; g.setLineDash(z.kind === "car" ? [] : [7, 4]); g.stroke(); g.setLineDash([]);
      const cx = z.polygon.reduce((s, p) => s + p[0], 0) / z.polygon.length, cy = z.polygon.reduce((s, p) => s + p[1], 0) / z.polygon.length;
      const [lx, ly] = this.px([cx, cy]), text = z.name || z.zone_id;
      g.font = "600 13px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
      const tw = g.measureText(text).width;
      g.fillStyle = "rgba(255,255,255,.88)"; g.fillRect(lx - tw / 2 - 4, ly - 10, tw + 8, 20);
      g.fillStyle = "#1d1b19"; g.fillText(text, lx, ly);
      if (sel) z.polygon.forEach((p) => {
        const [x, y] = this.px(p);
        g.beginPath(); g.arc(x, y, HANDLE, 0, Math.PI * 2); g.fillStyle = "#fff"; g.fill(); g.lineWidth = 2; g.strokeStyle = col; g.stroke();
      });
    });
    this.floor.img.forEach((p, i) => {
      const [x, y] = this.px(p);
      g.beginPath(); g.arc(x, y, 7, 0, Math.PI * 2); g.fillStyle = "#0a8f8f"; g.fill();
      g.lineWidth = 2; g.strokeStyle = "#fff"; g.stroke();
      g.fillStyle = "#fff"; g.font = "700 11px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText(String(i + 1), x, y);
    });
    const d = this.draft;
    if (d?.type === "rect") {
      const [x0, y0] = this.px(d.a), [x1, y1] = this.px(d.b);
      g.strokeStyle = "#fff"; g.lineWidth = 2; g.setLineDash([6, 4]); g.strokeRect(x0, y0, x1 - x0, y1 - y0); g.setLineDash([]);
    } else if (d?.type === "poly") {
      g.beginPath(); d.pts.forEach((p, k) => { const [x, y] = this.px(p); k ? g.lineTo(x, y) : g.moveTo(x, y); });
      if (d.hover) { const [x, y] = this.px(d.hover); g.lineTo(x, y); }
      g.strokeStyle = "#fff"; g.lineWidth = 2; g.stroke();
      d.pts.forEach((p) => { const [x, y] = this.px(p); g.beginPath(); g.arc(x, y, 4, 0, Math.PI * 2); g.fillStyle = "#fff"; g.fill(); });
    }
  }
}
