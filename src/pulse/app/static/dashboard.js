// Results dashboard: renders report.json (see pulse/report.py) into the #dashboard element.
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
export const pct = (v) => (isNum(v) ? `${Math.round(v * 100)}%` : "n/a");
export const mmss = (v) => {
  if (!isNum(v)) return "n/a";
  const t = Math.round(v);
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
};
const int = (v) => (isNum(v) ? String(Math.round(v)) : "n/a");

function kpi(label, value, unit = "") {
  return h("div", { class: "kpi" }, h("div", { class: "v" }, value, unit && h("span", { class: "u" }, unit)), h("div", { class: "l" }, label));
}

function evidenceValue(key, v) {
  if (!isNum(v)) return v === null || v === undefined ? "n/a" : String(v);
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
  const svg = s("svg", { class: "flow", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Visitor journey flow between zones" });
  const sorted = [...fwd].sort((a, b) => depth[a.source] - depth[b.source] || b.count - a.count);
  for (const l of sorted) {
    const a = pos[l.source], b = pos[l.target]; if (!a || !b) continue;
    const w = Math.max(2, l.count * scale);
    const y0 = a.y + a.out + w / 2, y1 = b.y + b.inn + w / 2; a.out += w; b.inn += w;
    const x0 = a.x + boxW, x1 = b.x, mx = (x0 + x1) / 2;
    const path = s("path", { class: "link", d: `M${x0},${y0} C${mx},${y0} ${mx},${y1} ${x1},${y1}`, "stroke-width": w });
    path.append(s("title", {}, document.createTextNode(`${l.source} to ${l.target}: ${l.count}`)));
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
  return h("div", { class: "warnings", role: "region", "aria-label": "Warnings" },
    h("h3", {}, "Warnings"), h("ul", {}, warnings.map((w) => h("li", {}, w))));
}

function kpiTiles(st) {
  const peak = isNum(st.peak_occupancy) ? int(st.peak_occupancy) : "n/a";
  return h("div", { class: "kpis" },
    kpi("Visitors (excl. staff)", int(st.visitors), "people"),
    kpi("Groups", int(st.groups)),
    kpi("Median visit length", mmss(st.median_visit_s), "m:ss"),
    kpi("Browse rate", pct(st.browse_rate), "stopped at a car"),
    kpi("Assisted share", pct(st.assisted_share), "got staff contact"),
    kpi("Unserved-interest episodes", int(st.unserved_episodes)),
    kpi(isNum(st.peak_t) ? `Peak occupancy (at ${mmss(st.peak_t)})` : "Peak occupancy", peak, "people"));
}

function carTable(rows) {
  if (!rows.length) return h("p", { class: "empty" }, "No car zones defined, so there is no per-car table. Use Edit zones to mark car areas.");
  const head = ["Rank", "Car", "Stops", "Stop rate", "Median dwell", "Total dwell", "Repeat visits", "Unassisted share", "Sat in car"];
  return h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, head.map((t) => h("th", { scope: "col" }, t)))),
    h("tbody", {}, rows.map((c) => h("tr", {},
      h("td", {}, `#${c.rank ?? ""}`), h("td", {}, c.name || c.zone_id), h("td", {}, int(c.stops)), h("td", {}, pct(c.stop_rate)),
      h("td", {}, mmss(c.median_dwell)), h("td", {}, mmss(c.total_dwell)), h("td", {}, int(c.repeat_visits)),
      h("td", {}, pct(c.unassisted_share)), h("td", {}, int(c.sat_in_car ?? 0)))))));
}

function insightCard(ins, k, files, jobId, seek) {
  const insufficient = ins.confidence === "insufficient";
  const sheet = files[`contact_sheet_${k}`];
  const ev = Object.entries(ins.evidence || {});
  return h("article", { class: "insight" },
    h("div", { class: "tags" },
      h("span", { class: `badge ${insufficient ? "insufficient" : ins.confidence}` },
        insufficient ? "insufficient data" : `${ins.confidence} confidence`),
      ins.demo_only && h("span", { class: "badge demo" }, "demo only"),
      h("span", { class: "badge insufficient" }, `n = ${ins.sample}`)),
    h("h4", {}, ins.title),
    ev.length && h("dl", { class: "evidence" }, ev.map(([key, v]) =>
      h("div", {}, h("dt", {}, prettyKey(key) + ":"), h("dd", {}, evidenceValue(key, v))))),
    h("p", { class: "action" }, h("strong", {}, "Suggested action: "), ins.action),
    ins.moments?.length ? h("div", {}, h("div", { class: "muted" }, "Jump to moment in video:"),
      h("div", { class: "moments" }, ins.moments.slice(0, 8).map((m) =>
        h("button", { type: "button", onclick: () => seek(m.t) }, `${mmss(m.t)}`)))) : null,
    sheet && h("img", { class: "sheet", loading: "lazy", src: `/api/jobs/${jobId}/files/${sheet}`,
      alt: `Contact sheet of moments for: ${ins.title}` }));
}

// ------------------------------------------------------------------ main
export function renderDashboard(root, report, ctx) {
  const { jobId } = ctx;
  const st = report.store || {}, files = report.files || {};
  const warnings = [...(report.warnings || [])];
  root.textContent = "";

  const video = h("video", { controls: true, preload: "metadata", "aria-label": "Annotated showroom video" });
  const videoMsg = h("p", { class: "muted", "aria-live": "polite" });
  const seek = (t) => {
    if (!video.getAttribute("src")) { videoMsg.textContent = "The annotated video is still rendering; the moment link will work once it is ready."; return; }
    video.currentTime = t; video.scrollIntoView({ behavior: "smooth", block: "center" }); video.play?.().catch(() => {});
  };

  const head = h("div", { class: "dash-head" },
    h("h2", {}, "Results"),
    h("div", { class: "actions" },
      h("button", { type: "button", onclick: ctx.onEditZones }, "Edit zones"),
      h("button", { type: "button", onclick: ctx.onEditRoles }, "Edit roles"),
      h("a", { href: `/api/jobs/${jobId}/export.json`, download: "" }, "Export JSON"),
      h("a", { href: `/api/jobs/${jobId}/export.csv`, download: "" }, "Export CSV (per car)")));

  const insights = report.insights || [];
  const insightsBody = insights.length
    ? h("div", { class: "insights" }, insights.map((ins, k) => insightCard(ins, k, files, jobId, seek)))
    : h("p", { class: "empty" }, "No recommendations yet — not enough visits for confident insights.");

  const flow = journeyFlowSvg(report.journey_flow || {});
  const top = (report.journey_flow?.top_paths || []).slice(0, 5);
  const flowCard = h("section", { class: "card" }, h("h3", {}, "Visitor journey"),
    flow ? [flow.svg, flow.skipped ? h("p", { class: "legend" }, `Columns are steps from entry. Link width is the number of visitors. ${flow.skipped} return link(s) not drawn.`)
                                   : h("p", { class: "legend" }, "Columns are steps from entry. Link width is the number of visitors.")]
         : h("p", { class: "empty" }, "No customer journeys to draw yet."),
    top.length ? [h("h4", {}, "Top paths"), h("ol", { class: "paths" }, top.map((p) => h("li", {}, `${p.path.join(" → ")} (${p.count} ${p.count === 1 ? "visitor" : "visitors"})`)))] : null);

  const img = (key, alt) => {
    const rel = files[key];
    return rel ? h("img", { src: `/api/jobs/${jobId}/files/${rel}`, alt }) : h("p", { class: "empty" }, "Not available.");
  };

  root.append(head, warningsBlock(warnings), kpiTiles(st),
    h("section", { class: "card" }, h("h3", {}, "Per-car engagement"), carTable(report.per_car || [])),
    h("section", { class: "card" }, h("h3", {}, "Recommendations"), insightsBody),
    h("div", { class: "grid2" },
      h("section", { class: "card" }, h("h3", {}, "Where people walk and linger"), img("heatmap", "Heatmap of visitor positions over the showroom keyframe; warm colours mean more time spent"),
        h("p", { class: "legend" }, "Warm areas are where visitors spent the most time; white lines are typical walking paths.")),
      flowCard),
    h("div", { class: "grid2" },
      h("section", { class: "card" }, h("h3", {}, "Customers present over time"), img("occupancy_chart", "Line chart of customers present over time")),
      h("section", { class: "card" }, h("h3", {}, "Customers waiting unattended"), img("waiting_chart", "Line chart of customers waiting without staff contact"))),
    h("section", { class: "card" }, h("h3", {}, "Annotated video"), video, videoMsg));

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
    if (state === "failed") { videoMsg.textContent = "The annotated video could not be rendered. The numbers above are unaffected."; return; }
    if (state === "none") { videoMsg.textContent = "Video rendering was not requested for this run."; return; }
    videoMsg.textContent = "Rendering annotated video…";
    setTimeout(tick, 2000);
  };
  tick();
  return () => { stopped = true; };
}
