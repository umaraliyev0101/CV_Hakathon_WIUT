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
    getSessions: async () => todo("getSessions"),
    // analyzeVideo(file, onStep) -> the new session (same shape). Call onStep(index 0-3, label) as work progresses.
    analyzeVideo: async (file, onStep) => todo("analyzeVideo"),
    // One session's results page (see CONNECTING.md, "Session results"):
    // { id, name, recorded, lengthSec, customers, medianGreetSec|null,
    //   demo: false,                                   // true only for invented results (the page then says so)
    //   videos:[{ label, url }],                       // annotated videos, e.g. Coach's and Pulse's; first one shows
    //   moments:[{ t (s), kind:"greet"|"engagement"|"walkout"|"unserved", lens:"coach"|"pulse", text, durSec? }],
    //   staff:[{ name, greets, medianGreetSec|null, engaged, walkouts, score (0-100) }],
    //   routes:[{ path:["Entrance","RAV4",...,"Exit"], count }],
    //   busiestCar:{ label, visits, medianDwellSec, unserved } | null }
    getSession: async (id) => todo("getSession"),

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
