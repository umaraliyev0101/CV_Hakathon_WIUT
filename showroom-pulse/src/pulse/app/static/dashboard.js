// Results dashboard: renders report.json (see pulse/report.py) into the #dashboard element.
import { t } from "/static/i18n.js";
const SVG_NS = "http://www.w3.org/2000/svg";

export function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) e.append(kid);
  return e;
}
const s = (tag, attrs = {}, ...kids) => {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const kid of kids) if (kid !== null && kid !== undefined) e.append(kid);
  return e;
};

const isNum = (v) => typeof v === "number" && Number.isFinite(v);
export const pct = (v) => (isNum(v) ? `${Math.round(v * 100)}%` : t("common.na"));
export const mmss = (v) => {
  if (!isNum(v)) return t("common.na");
  const sec = Math.round(v);
  return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
};
const int = (v) => (isNum(v) ? String(Math.round(v)) : t("common.na"));

function kpi(label, value, unit = "") {
  return h("div", { class: "kpi" }, h("div", { class: "v" }, value, unit && h("span", { class: "u" }, unit)), h("div", { class: "l" }, label));
}

function evidenceValue(key, v) {
  if (!isNum(v)) return v === null || v === undefined ? t("common.na") : String(v);
  if (/share|rate/.test(key) && v >= 0 && v <= 1) return pct(v);
  if (/dwell|_s$|time/.test(key)) return mmss(v);
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}
const prettyKey = (k) => k.replace(/_/g, " ");

// ------------------------------------------------------------------ journey flow (SVG)
export function journeyFlowSvg(flow) {
  const nodes = flow.nodes || [], links = flow.links || [];
  const real = links.filter((l) => l.count > 0);
  if (!real.length) return null;
  // column = shortest distance from Start; End goes in the last column
  const depth = { Start: 0 }, queue = ["Start"];
  while (queue.length) {
    const n = queue.shift();
    for (const l of real) if (l.source === n && !(l.target in depth)) { depth[l.target] = depth[n] + 1; queue.push(l.target); }
  }
  const maxMid = Math.max(0, ...Object.entries(depth).filter(([k]) => k !== "End").map(([, d]) => d));
  depth.End = maxMid + 1;
  for (const n of nodes) if (!(n.id in depth)) depth[n.id] = 1;
  const fwd = real.filter((l) => depth[l.target] > depth[l.source]);
  const skipped = real.length - fwd.length;
  const cols = depth.End + 1, W = 640, colW = W / cols, boxW = Math.min(110, colW * 0.7), PAD = 14, GAP = 14;
  const flowOf = (id) => Math.max(
    fwd.filter((l) => l.source === id).reduce((a, l) => a + l.count, 0),
    fwd.filter((l) => l.target === id).reduce((a, l) => a + l.count, 0));
  const byCol = Array.from({ length: cols }, () => []);
  for (const n of nodes) byCol[Math.min(depth[n.id], cols - 1)].push(n);
  const colTotal = byCol.map((c) => c.reduce((a, n) => a + Math.max(flowOf(n.id), 1), 0));
  const maxCount = Math.max(...colTotal, 1);
  const maxNodes = Math.max(...byCol.map((c) => c.length), 1);
  const H = Math.max(220, maxNodes * 56);
  const scale = Math.min(((H - 2 * PAD - (maxNodes - 1) * GAP) / maxCount), 40);
  const pos = {};
  byCol.forEach((col, ci) => {
    let y = PAD;
    for (const n of col) {
      const hgt = Math.max(26, Math.max(flowOf(n.id), 1) * scale);
      pos[n.id] = { x: ci * colW + (colW - boxW) / 2, y, h: hgt, out: 0, inn: 0 };
      y += hgt + GAP;
    }
  });
  const svg = s("svg", { class: "flow", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("flow.aria") });
  const sorted = [...fwd].sort((a, b) => depth[a.source] - depth[b.source] || b.count - a.count);
  for (const l of sorted) {
    const a = pos[l.source], b = pos[l.target]; if (!a || !b) continue;
    const w = Math.max(2, l.count * scale);
    const y0 = a.y + a.out + w / 2, y1 = b.y + b.inn + w / 2; a.out += w; b.inn += w;
    const x0 = a.x + boxW, x1 = b.x, mx = (x0 + x1) / 2;
    const path = s("path", { class: "link", d: `M${x0},${y0} C${mx},${y0} ${mx},${y1} ${x1},${y1}`, "stroke-width": w });
    path.append(s("title", {}, document.createTextNode(t("flow.link", { source: l.source, target: l.target, count: l.count }))));
    svg.append(path);
  }
  for (const n of nodes) {
    const p = pos[n.id]; if (!p) continue;
    const label = n.name.length > 18 ? n.name.slice(0, 17) + "…" : n.name;
    svg.append(s("rect", { class: "node", x: p.x, y: p.y, width: boxW, height: p.h, rx: 5 }),
      s("text", { x: p.x + boxW / 2, y: p.y + p.h / 2 + 4, "text-anchor": "middle" }, document.createTextNode(label)));
  }
  return { svg, skipped };
}

// ------------------------------------------------------------------ pieces
function warningsBlock(warnings) {
  if (!warnings.length) return null;
  return h("div", { class: "warnings", role: "region", "aria-label": t("warnings.h") },
    h("h3", {}, t("warnings.h")), h("ul", {}, warnings.map((w) => h("li", {}, w))));
}

function kpiTiles(st) {
  const peak = isNum(st.peak_occupancy) ? int(st.peak_occupancy) : t("common.na");
  return h("div", { class: "kpis" },
    kpi(t("kpi.visitors"), int(st.visitors), t("kpi.visitors.unit")),
    kpi(t("kpi.groups"), int(st.groups)),
    kpi(t("kpi.median_visit"), mmss(st.median_visit_s), t("kpi.median_visit.unit")),
    kpi(t("kpi.browse"), pct(st.browse_rate), t("kpi.browse.unit")),
    kpi(t("kpi.assisted"), pct(st.assisted_share), t("kpi.assisted.unit")),
    kpi(t("kpi.unserved"), int(st.unserved_episodes)),
    kpi(isNum(st.peak_t) ? t("kpi.peak_at", { time: mmss(st.peak_t) }) : t("kpi.peak"), peak, t("kpi.visitors.unit")));
}

function carTable(rows) {
  if (!rows.length) return h("p", { class: "empty" }, t("cars.none"));
  const head = [t("cars.rank"), t("cars.car"), t("cars.stops"), t("cars.stop_rate"), t("cars.median_dwell"), t("cars.total_dwell"), t("cars.repeat"), t("cars.unassisted"), t("cars.sat"), t("cars.sold"), t("cars.per_sale")];
  return h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, head.map((label) => h("th", { scope: "col" }, label)))),
    h("tbody", {}, rows.map((c) => h("tr", {},
      h("td", {}, `#${c.rank ?? ""}`), h("td", {}, c.name || c.zone_id), h("td", {}, int(c.stops)), h("td", {}, pct(c.stop_rate)),
      h("td", {}, mmss(c.median_dwell)), h("td", {}, mmss(c.total_dwell)), h("td", {}, int(c.repeat_visits)),
      h("td", {}, pct(c.unassisted_share)), h("td", {}, int(c.sat_in_car ?? 0)),
      h("td", {}, isNum(c.sold) ? int(c.sold) : t("common.na")), h("td", {}, isNum(c.attention_to_sales) ? (Math.round(c.attention_to_sales * 100) / 100).toString() : t("common.na")))))));
}

const confLabel = (c) => (c === "high" ? t("ins.conf_high") : c === "medium" ? t("ins.conf_medium") : c === "low" ? t("ins.conf_low") : `${c}`);

function insightCard(ins, k, files, jobId, seek) {
  const insufficient = ins.confidence === "insufficient";
  const sheet = files[`contact_sheet_${k}`];
  const ev = Object.entries(ins.evidence || {});
  return h("article", { class: "insight" },
    h("div", { class: "tags" },
      h("span", { class: `badge ${insufficient ? "insufficient" : ins.confidence}` },
        insufficient ? t("ins.insufficient") : confLabel(ins.confidence)),
      ins.demo_only && h("span", { class: "badge demo" }, t("ins.demo_only")),
      h("span", { class: "badge insufficient" }, t("ins.sample", { n: ins.sample }))),
    h("h4", {}, ins.title),
    ev.length && h("dl", { class: "evidence" }, ev.map(([key, v]) =>
      h("div", {}, h("dt", {}, prettyKey(key) + ":"), h("dd", {}, evidenceValue(key, v))))),
    h("p", { class: "action" }, h("strong", {}, t("ins.action")), ins.action),
    ins.moments?.length ? h("div", {}, h("div", { class: "muted" }, t("ins.jump")),
      h("div", { class: "moments" }, ins.moments.slice(0, 8).map((m) =>
        h("button", { type: "button", onclick: () => seek(m.t) }, `${mmss(m.t)}`)))) : null,
    sheet && h("img", { class: "sheet", loading: "lazy", src: `/api/jobs/${jobId}/files/${sheet}`,
      alt: t("ins.sheet_alt", { title: ins.title }) }));
}

// ------------------------------------------------------------------ main
export function renderDashboard(root, report, ctx) {
  const { jobId } = ctx;
  const st = report.store || {}, files = report.files || {};
  const warnings = [...(report.warnings || [])];
  root.textContent = "";

  const video = h("video", { controls: true, preload: "metadata", "aria-label": t("dash.video_aria") });
  const videoMsg = h("p", { class: "muted", "aria-live": "polite" });
  const seek = (sec) => {
    if (!video.getAttribute("src")) { videoMsg.textContent = t("video.pending"); return; }
    video.currentTime = sec; video.scrollIntoView({ behavior: "smooth", block: "center" }); video.play?.().catch(() => {});
  };

  const head = h("div", { class: "dash-head" },
    h("h2", {}, t("dash.results")),
    h("div", { class: "actions" },
      h("button", { type: "button", onclick: ctx.onEditZones }, t("dash.edit_zones")),
      h("button", { type: "button", onclick: ctx.onEditRoles }, t("dash.edit_roles")),
      h("a", { href: `/api/jobs/${jobId}/export.json`, download: "" }, t("dash.export_json")),
      h("a", { href: `/api/jobs/${jobId}/export.csv`, download: "" }, t("dash.export_csv")),
      ctx.onSales && h("label", { class: "muted" }, t("dash.sales_csv"),
        h("input", { type: "file", accept: ".csv,text/csv", "aria-label": t("dash.sales_aria"),
                     onchange: (ev) => { const f = ev.target.files?.[0]; if (f) ctx.onSales(f); } })),
      files.report_html && h("a", { href: `/api/jobs/${jobId}/files/${files.report_html}`, target: "_blank", rel: "noopener" }, t("dash.printable"))));

  const insights = report.insights || [];
  const insightsBody = insights.length
    ? h("div", { class: "insights" }, insights.map((ins, k) => insightCard(ins, k, files, jobId, seek)))
    : h("p", { class: "empty" }, t("ins.none"));

  const flow = journeyFlowSvg(report.journey_flow || {});
  const top = (report.journey_flow?.top_paths || []).slice(0, 5);
  const flowCard = h("section", { class: "card" }, h("h3", {}, t("flow.h")),
    flow ? [flow.svg, flow.skipped ? h("p", { class: "legend" }, t("flow.legend_skipped", { skipped: flow.skipped }))
                                   : h("p", { class: "legend" }, t("flow.legend"))]
         : h("p", { class: "empty" }, t("flow.none")),
    top.length ? [h("h4", {}, t("flow.top")), h("ol", { class: "paths" }, top.map((p) => h("li", {}, t("flow.path", { path: p.path.join(" → "), count: p.count }))))] : null);

  const img = (key, alt) => {
    const rel = files[key];
    return rel ? h("img", { src: `/api/jobs/${jobId}/files/${rel}`, alt }) : h("p", { class: "empty" }, t("chart.na"));
  };

  root.append(head, warningsBlock(warnings), kpiTiles(st),
    h("section", { class: "card" }, h("h3", {}, t("cars.h")), carTable(report.per_car || [])),
    h("section", { class: "card" }, h("h3", {}, t("ins.h")), h("p", { class: "muted" }, h("small", {}, t("note.english"))), insightsBody),
    h("div", { class: "grid2" },
      h("section", { class: "card" }, h("h3", {}, t("heat.h")), img("heatmap", t("heat.alt")),
        h("p", { class: "legend" }, t("heat.legend"))),
      flowCard),
    h("div", { class: "grid2" },
      h("section", { class: "card" }, h("h3", {}, t("chart.occupancy_h")), img("occupancy_chart", t("chart.occupancy_alt"))),
      h("section", { class: "card" }, h("h3", {}, t("chart.waiting_h")), img("waiting_chart", t("chart.waiting_alt")))),
    h("section", { class: "card" }, h("h3", {}, t("dash.video")), video, videoMsg));

  // annotated video: poll while rendering, stop when the dashboard is replaced
  let stopped = false;
  const tick = async () => {
    if (stopped || !root.contains(video)) return;
    let state = "none";
    try { state = (await ctx.getStatus()).video_state; } catch { /* retry */ }
    if (state === "ready") {
      const rel = files.annotated_video || "annotated.mp4";
      video.src = `/api/jobs/${jobId}/files/${rel}?v=${Date.now()}`;
      videoMsg.textContent = ""; return;
    }
    if (state === "failed") { videoMsg.textContent = t("video.failed"); return; }
    if (state === "none") { videoMsg.textContent = t("video.none"); return; }
    videoMsg.textContent = t("video.rendering");
    setTimeout(tick, 2000);
  };
  tick();
  return () => { stopped = true; };
}
