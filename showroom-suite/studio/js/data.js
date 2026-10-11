// One place where the page gets its data. Every view calls Suite.data.<fn>() and gets back the shapes
// documented below, whether the numbers come from the demo (mock.js) or from the real Coach and Pulse servers.
//
// To connect a section: fill in its functions in `Live` (path + adapter), then set its flag in config.js.
// If a live read fails, the section shows "not reachable" and falls back to demo data, so the page never goes blank.
// A failed save or upload shows its error instead.
(function () {
  const C = window.SUITE_CONFIG;

  async function get(base, path) {
    const r = await fetch(base + path, { headers: { Accept: "application/json" } });
    if (!r.ok) throw new Error(`${base}${path} answered HTTP ${r.status}`);
    return r.json();
  }
  async function send(base, path, body, method = "POST") {
    const isForm = body instanceof FormData;
    const r = await fetch(base + path, {
      method,
      headers: isForm ? {} : { "Content-Type": "application/json" },
      body: isForm ? body : JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`${base}${path} answered HTTP ${r.status}`);
    return r.json();
  }
  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.round(s) % 60).padStart(2, "0")}`; // 48 -> "0:48"
  const todo = (name) => { throw new Error(`${name} is not connected yet. See CONNECTING.md.`); };

  // ---------------------------------------------------------------------------------------------
  // Replays, connected: one upload goes to BOTH servers (Coach: salespeople, Pulse: customers). The studio has
  // no confirmation step, so each app's own staff suggestions and Pulse's automatic car zones are accepted as
  // they are; the apps' own pages (/coach/, /pulse/) can still correct them. A session id is "<coach>~<pulse>".
  // The session list lives in this browser (the servers keep jobs for 24 h).
  // ---------------------------------------------------------------------------------------------
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const STORE = "suite.sessions";
  const readSessions = () => { try { return JSON.parse(localStorage.getItem(STORE)) || []; } catch { return []; } };
  function saveSession(s) {
    const list = readSessions().filter((x) => x.id !== s.id);
    list.unshift(s);
    try { localStorage.setItem(STORE, JSON.stringify(list.slice(0, 50))); } catch { /* storage unavailable */ }
  }
  const sessionMeta = (id) => readSessions().find((x) => x.id === id);

  function videoDuration(file) {
    return new Promise((resolve) => {
      const v = document.createElement("video"), url = URL.createObjectURL(file);
      const done = (d) => { URL.revokeObjectURL(url); resolve(d); };
      v.preload = "metadata";
      v.onloadedmetadata = () => done(Number.isFinite(v.duration) ? v.duration : null);
      v.onerror = () => done(null);
      v.src = url;
    });
  }

  // Poll `read` every second until `isDone(status)`; `onTick(status)` reports progress. Throws on `isFailed`.
  async function until(read, isDone, isFailed, onTick, timeoutSec = 3600) {
    const t0 = Date.now();
    for (;;) {
      const st = await read();
      if (isFailed(st)) throw new Error(st.error || st.message || "analysis failed");
      if (isDone(st)) return st;
      onTick?.(st);
      if (Date.now() - t0 > timeoutSec * 1000) throw new Error("timed out");
      await wait(1000);
    }
  }

  async function coachPipeline(file, preset, step) {
    const { job_id: id } = await send(C.coachUrl, "/api/jobs", form(file, preset));
    step.coachId = id;
    const job = (s) => get(C.coachUrl, `/api/jobs/${id}`);
    await until(job, (s) => ["needs_confirm", "rendering", "done"].includes(s.status), (s) => s.status === "failed",
      (s) => step.tick("coach", s.message || s.status, s.progress));
    // Accept Coach's staff suggestions. If it suggests nobody, the "unsure" person on screen longest is taken as staff
    // (the same rule as Coach's own smoke test); without any staff there are no greets or scorecards at all.
    const { people } = await get(C.coachUrl, `/api/jobs/${id}/people`);
    const roles = {};
    let autoStaff = null;
    if (people.length > 1 && !people.some((p) => p.role === "staff")) {
      const pool = people.filter((p) => p.suggestion?.role === "unsure");
      const pick = (pool.length ? pool : people).reduce((a, b) => (b.t_last - b.t_first > a.t_last - a.t_first ? b : a));
      roles[pick.pid] = "staff";
      autoStaff = pick.pid;
    }
    const res = await send(C.coachUrl, `/api/jobs/${id}/roles`, { roles });
    step.tick("coach", "drawing the annotated video", 1);
    await until(job, (s) => s.video_status === "ready" || s.video_status === "failed", (s) => s.status === "failed");
    return { id, res, autoStaff };
  }

  async function pulsePipeline(file, preset, step) {
    const { job_id: id } = await send(C.pulseUrl, "/api/jobs", form(file, preset));
    step.pulseId = id;
    const job = () => get(C.pulseUrl, `/api/jobs/${id}`);
    await until(job, (s) => s.state === "ready_for_zones" || s.state === "scored", (s) => s.state === "failed",
      (s) => step.tick("pulse", s.message || s.state, s.progress));
    await send(C.pulseUrl, `/api/jobs/${id}/zones`, await get(C.pulseUrl, `/api/jobs/${id}/zones`), "PUT");   // automatic zones
    await send(C.pulseUrl, `/api/jobs/${id}/roles`, { overrides: {}, skip: false }, "PUT");                 // suggested roles
    const res = await send(C.pulseUrl, `/api/jobs/${id}/score`, {});
    step.tick("pulse", "drawing the annotated video", 1);
    await until(job, (s) => ["ready", "failed", "none"].includes(s.video_state), (s) => s.state === "failed");
    return { id, res };
  }

  function form(file, preset) {
    const f = new FormData();
    f.append("file", file);
    f.append("preset", preset);
    return f;
  }

  function parseCsv(text) {
    const rows = [];
    let row = [], cell = "", q = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (q) { if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (ch === '"') q = false; else cell += ch; }
      else if (ch === '"') q = true;
      else if (ch === ",") { row.push(cell); cell = ""; }
      else if (ch === "\n") { row.push(cell.replace(/\r$/, "")); rows.push(row); row = []; cell = ""; }
      else cell += ch;
    }
    if (cell || row.length) { row.push(cell); rows.push(row); }
    const [head, ...body] = rows;
    return body.filter((r) => r.length === head.length).map((r) => Object.fromEntries(head.map((h, i) => [h, r[i]])));
  }

  // A rough 0-100 score: greet speed 40, engaged share 40, walk-outs 20 (the studio's Team weighting).
  const staffScore = (mg, greets, engaged, walkouts) =>
    Math.round(Math.max(0, Math.min(100, 40 * (mg == null ? 0.5 : Math.min(1, 60 / Math.max(mg, 1)))
      + 40 * (greets ? Math.min(1, engaged / greets) : 0) + 20 - 7 * walkouts)));

  async function liveSession(id) {
    const [cid, pid] = id.split("~");
    const meta = sessionMeta(id) || { id, name: `Session ${id}`, recorded: "" };
    const [c, p, visits] = await Promise.all([
      cid ? get(C.coachUrl, `/api/jobs/${cid}/results`).catch(() => null) : null,
      pid ? get(C.pulseUrl, `/api/jobs/${pid}/results`).catch(() => null) : null,
      pid ? fetch(`${C.pulseUrl}/api/jobs/${pid}/files/visits.csv`).then((r) => (r.ok ? r.text() : "")).catch(() => "") : "",
    ]);
    if (!c && !p) throw new Error("neither server has this session any more (jobs are kept for 24 h)");

    const name = (who) => (c?.names?.[who] || `Salesperson ${who}`);
    const carName = Object.fromEntries((p?.per_car || []).map((z) => [z.zone_id, z.name]));
    const moments = [];
    for (const e of c?.events || []) {
      if (e.type === "greet") moments.push({ t: e.t, kind: "greet", lens: "coach",
        text: `${name(e.staff)} greeted customer ${e.customer}${e.data?.time_to_greet != null ? ` after ${mmss(e.data.time_to_greet)}` : ""}.` });
      else if (e.type === "engagement" && e.t_end != null) moments.push({ t: e.t, kind: "engagement", lens: "coach", durSec: e.t_end - e.t,
        text: `${name(e.staff)} stayed with customer ${e.customer} for ${mmss(e.t_end - e.t)}.` });
      else if (e.type === "walkout") moments.push({ t: e.t, kind: "walkout", lens: "coach", text: `Customer ${e.customer} left without being greeted.` });
    }
    const seen = new Set();
    for (const ins of p?.insights || []) {
      if (ins.rule !== "R1") continue;                       // R1 = unserved interest at a car
      for (const m of ins.moments || []) {
        const key = `${m.person_id}@${m.zone_id}`;
        if (seen.has(key)) continue;
        seen.add(key);
        moments.push({ t: m.t, kind: "unserved", lens: "pulse", car: carName[m.zone_id] || m.zone_id,
          text: `A customer studied the ${carName[m.zone_id] || m.zone_id} and no salesperson came within 2 m.` });
      }
    }
    moments.sort((a, b) => a.t - b.t);

    const counts = {};
    for (const v of visits ? parseCsv(visits) : []) {
      if (v.role === "staff" || !v.zones) continue;
      const path = ["Entrance", ...v.zones.split(" > ").map((z) => carName[z] || z), "Exit"].join("\u0000");
      counts[path] = (counts[path] || 0) + 1;
    }
    const routes = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 5)
      .map(([k, count]) => ({ path: k.split("\u0000"), count }));
    const car = [...(p?.per_car || [])].sort((a, b) => b.visitors - a.visitors || b.total_dwell - a.total_dwell)[0];
    const store = c?.scorecard?.store || {};
    const videos = [];
    if (c?.video_url) videos.push({ label: "Salespeople · Coach", url: C.coachUrl + c.video_url });
    if (p?.files?.annotated_video) videos.push({ label: "Customers · Pulse", url: `${C.pulseUrl}/api/jobs/${pid}/files/${p.files.annotated_video}` });

    return {
      ...meta, demo: false,
      lengthSec: c?.duration ?? p?.meta?.duration ?? meta.lengthSec,
      customers: store.customers ?? p?.store?.visitors ?? 0,
      medianGreetSec: store.median_time_to_greet ?? null,
      videos, moments,
      staff: (c?.scorecard?.staff || []).map((s) => ({
        name: name(s.staff), greets: s.greets, medianGreetSec: s.median_time_to_greet, engaged: s.customers_engaged,
        walkouts: s.walkouts_while_free, score: staffScore(s.median_time_to_greet, s.greets, s.customers_engaged, s.walkouts_while_free) })),
      routes,
      busiestCar: car && car.visitors ? { label: car.name, visits: car.visitors, medianDwellSec: car.median_dwell, unserved: car.unserved } : null,
    };
  }

  // ---------------------------------------------------------------------------------------------
  // Live: fill these in. Each must return exactly the shape described above it.
  // `adapt` functions turn whatever the server returns into that shape; start with x => x if the
  // server already matches.
  // ---------------------------------------------------------------------------------------------
  const Live = {
    // Floorplan, in metres, origin top-left, y grows toward the entrance:
    // { widthM, depthM, entrance:{x,y,w}, zones:[{id,label,x,y,w,h}], fixtures:[{label,x,y,w,h}], unitsPerMetre? }
    // Units do not have to be metres: pass camera pixels (widthM = frame width in px) and set unitsPerMetre
    // to roughly how many pixels make one metre on the floor, so the 2 m ring and the 1 m grid stay true.
    getFloorplan: async () => todo("getFloorplan"), // e.g. adapt(await get(C.pulseUrl, "/api/zones"))

    // Live snapshot, polled every config.pollMs:
    // { clock:"14:32:05",
    //   people:[{ id, role:"staff"|"customer", name|null, x, y (metres), trail:[[x,y],...],
    //             waitingSec|null, zone|null, zoneDwellSec|null, engagedWith|null, engagedSec|null }],
    //   kpis:{ medianGreetSec, greetTargetSec, waitingNow, longestWaitSec, onFloor, staffOnFloor, staffFree,
    //          hottestZone:{label, people, medianDwellSec}|null, unservedToday },
    //   heat?:{ cols, rows, values:[...] },               // optional, drawn under the Customers lens
    //   events:[{ time:"14:31:58", lens:"coach"|"pulse", text }] }   // newest first
    getLive: async () => todo("getLive"),

    // Replays
    // getSessions -> [{ id, name, recorded, lengthSec, customers, medianGreetSec, status:"ready"|"running"|"failed" }]
    getSessions: async () => readSessions(),
    // analyzeVideo(file, onStep) -> the new session (same shape). Call onStep(index 0-3, label) as work progresses.
    analyzeVideo: async (file, onStep) => {
      const lengthSec = await videoDuration(file);
      const preset = lengthSec && lengthSec <= 90 ? "short" : "dealership";   // both apps' preset names
      const recorded = new Date().toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
      const state = { coach: "", pulse: "" };
      const step = {
        tick(app, text, progress) {
          state[app] = `${app === "coach" ? "Salespeople" : "Customers"}: ${text}${progress != null && progress < 1 ? ` (${Math.round(progress * 100)}%)` : ""}`;
          const both = [state.coach, state.pulse].filter(Boolean).join(" · ");
          const drawing = ["coach", "pulse"].filter((a) => state[a].includes("drawing")).length;
          onStep(1 + drawing, both);   // 1 analysing, 2 one app drawing its video, 3 both
        },
      };
      onStep(0, "Uploading to both servers");
      const [cr, pr] = await Promise.allSettled([coachPipeline(file, preset, step), pulsePipeline(file, preset, step)]);
      if (cr.status === "rejected" && pr.status === "rejected") throw new Error(`Salespeople: ${cr.reason.message}. Customers: ${pr.reason.message}`);
      const id = `${cr.value?.id || step.coachId || ""}~${pr.value?.id || step.pulseId || ""}`;
      const store = cr.value?.res?.scorecard?.store || {};
      const s = { id, name: file.name.replace(/\.[^.]+$/, ""), recorded, lengthSec: Math.round(lengthSec || 0),
                  customers: store.customers ?? pr.value?.res?.store?.visitors ?? 0,
                  medianGreetSec: store.median_time_to_greet ?? null,
                  status: "ready",
                  staffNote: cr.value?.autoStaff != null
                    ? `Salesperson ${cr.value.autoStaff} was taken as staff automatically (on screen longest); correct it in the Coach app`
                    : null,
                  warning: cr.status === "rejected" ? `Salespeople analysis failed: ${cr.reason.message}`
                         : pr.status === "rejected" ? `Customer analysis failed: ${pr.reason.message}` : null };
      saveSession(s);
      return s;
    },
    // One session's results page (see CONNECTING.md, "Session results"):
    // { id, name, recorded, lengthSec, customers, medianGreetSec|null,
    //   demo: false,                                   // true only for invented results (the page then says so)
    //   videos:[{ label, url }],                       // annotated videos, e.g. Coach's and Pulse's; first one shows
    //   moments:[{ t (s), kind:"greet"|"engagement"|"walkout"|"unserved", lens:"coach"|"pulse", text, durSec? }],
    //   staff:[{ name, greets, medianGreetSec|null, engaged, walkouts, score (0-100) }],
    //   routes:[{ path:["Entrance","RAV4",...,"Exit"], count }],
    //   busiestCar:{ label, visits, medianDwellSec, unserved } | null }
    getSession: async (id) => liveSession(id),

    // Team (Coach)
    // { period, lead:[{label,value,text}] (4 items),
    //   staff:[{ id, name, since, greets, medianGreetSec, engaged, walkouts, score (0-100), trend:[7 numbers] }],
    //   suggestions:[{ trackId, reason, confidence (0-1), thumbUrl? }] }
    getTeam: async () => todo("getTeam"),
    // confirmStaff(trackId, isStaff, name) -> { ok:true }
    confirmStaff: async (trackId, isStaff, name) => todo("confirmStaff"),

    // Visitors (Pulse)
    // { period, lead:[{label,value,text}] (4 items),
    //   zones:[{ label, visits, medianDwellSec, over60 (0-1), approached (0-1), unserved }],
    //   journeys:[{ path:["Entrance","RAV4",...,"Exit"], count }],
    //   unserved:[{ time:"14:02", zone, dwellSec, note }],
    //   heat:{ cols, rows, values:[...] }  // row-major, any non-negative scale
    // }
    getVisitors: async () => todo("getVisitors"),

    // Demand (marketing; mostly Pulse, plus Coach for "talked to a salesperson")
    // { period,
    //   funnel:[{ label, sub?, count, fix? }],   // ordered stages; `fix` = what to do if this stage loses the most
    //   hours:{ days:["Mon",...], hours:[10,...,20], values:[[avg visitors per hour, ...] one row per day] },
    //   models:[{ label, considered (visits 60 s+ per week), approached (0-1), unserved }],
    //   campaign:{ model, campaignLabel, weeks:["17 Aug",...], startIndex, series:[{ label, values:[one per week] }] } }
    getDemand: async () => todo("getDemand"),

    // Insights (both)
    // [{ lens:"Customers × Salespeople"|"Salespeople"|"Customers", value, text, act }]
    getInsights: async () => todo("getInsights"),
  };

  const SECTION_OF = {
    getFloorplan: "floor", getLive: "floor",
    getSessions: "sessions", analyzeVideo: "sessions", getSession: "sessions",
    getTeam: "team", confirmStaff: "team",
    getVisitors: "visitors",
    getDemand: "demand",
    getInsights: "insights",
  };
  const WRITES = new Set(["confirmStaff", "analyzeVideo"]);
  const status = {}; // section -> { state:"demo"|"live"|"error", message }
  const listeners = [];

  function setStatus(section, state, message) {
    const prev = status[section];
    if (prev && prev.state === state && prev.message === message) return;
    status[section] = { state, message };
    listeners.forEach((fn) => fn(section, status[section]));
  }

  const data = {};
  for (const name of Object.keys(SECTION_OF)) {
    const section = SECTION_OF[name];
    data[name] = async (...args) => {
      if (!C.live[section]) { setStatus(section, "demo"); return window.Mock[name](...args); }
      try {
        const out = await Live[name](...args);
        setStatus(section, "live");
        return out;
      } catch (err) {
        console.warn(`[suite] ${name}:`, err);
        setStatus(section, "error", err.message);
        // Saves and uploads must not pretend to succeed against the demo; only reads fall back.
        if (WRITES.has(name)) throw err;
        return window.Mock[name](...args);
      }
    };
  }

  window.Suite = {
    data,
    status,
    onStatus: (fn) => listeners.push(fn),
    helpers: { get, send, coach: () => C.coachUrl, pulse: () => C.pulseUrl },
  };
})();
