// Demo data. Everything here is invented for the demo: a small simulation of a showroom floor plus
// fixed numbers for the report pages. data.js serves these whenever a section is not connected yet.
(function () {
  const FLOOR = {
    widthM: 24,
    depthM: 16,
    entrance: { x: 10.5, y: 16, w: 3 },
    zones: [
      { id: "camry", label: "Camry", x: 2.6, y: 2.6, w: 4.8, h: 1.9 },
      { id: "rav4", label: "RAV4", x: 9.8, y: 2.6, w: 4.6, h: 1.9 },
      { id: "lc300", label: "Land Cruiser 300", x: 16.6, y: 2.5, w: 5.0, h: 2.0 },
      { id: "cross", label: "Corolla Cross", x: 5.4, y: 8.2, w: 4.4, h: 1.8 },
      { id: "hilux", label: "Hilux", x: 13.6, y: 8.2, w: 5.4, h: 1.9 },
    ],
    fixtures: [
      { label: "Reception", x: 19.4, y: 12.8, w: 3.4, h: 1.3 },
      { label: "Lounge", x: 1.4, y: 12.2, w: 4.4, h: 2.6 },
    ],
  };
  const ZONE_WEIGHT = { camry: 3, rav4: 4, lc300: 5, cross: 2.5, hilux: 1.5 };
  const byId = Object.fromEntries(FLOOR.zones.map((z) => [z.id, z]));

  const SPEED = 6; // simulated seconds per real second, so a few minutes of floor play out per demo minute
  let t = 14 * 3600 + 20 * 60;
  let nextGuest = 26;
  const people = [];
  const events = [];
  const greets = [52, 38, 71, 44, 49, 90, 33, 58, 41, 47, 66, 39, 55, 62, 36];
  let unservedToday = 4;
  const HEAT_COLS = 48, HEAT_ROWS = 32;
  const heat = new Float32Array(HEAT_COLS * HEAT_ROWS);

  const rnd = (a, b) => a + Math.random() * (b - a);
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const pad2 = (n) => String(Math.floor(n)).padStart(2, "0");
  const clock = (s) => `${pad2(s / 3600)}:${pad2((s % 3600) / 60)}:${pad2(s % 60)}`;
  const mmss = (s) => `${Math.floor(s / 60)}:${pad2(s % 60)}`;
  const log = (lens, text) => { events.unshift({ time: clock(t), lens, text }); if (events.length > 40) events.pop(); };

  function pickZone(except) {
    const ids = FLOOR.zones.map((z) => z.id).filter((id) => id !== except);
    let total = ids.reduce((s, id) => s + ZONE_WEIGHT[id], 0), r = Math.random() * total;
    for (const id of ids) { r -= ZONE_WEIGHT[id]; if (r <= 0) return id; }
    return ids[0];
  }
  function zonePoint(z) {
    const p = 0.9, front = z.y < 6; // front-row cars sit against the window, so nobody stands behind them
    const side = front ? 1 + Math.floor(Math.random() * 3) : Math.floor(Math.random() * 4);
    if (side === 0) return [rnd(z.x, z.x + z.w), z.y - p];
    if (side === 1) return [rnd(z.x, z.x + z.w), z.y + z.h + p];
    if (side === 2) return [z.x - p, rnd(z.y, z.y + z.h)];
    return [z.x + z.w + p, rnd(z.y, z.y + z.h)];
  }
  function moveToward(p, speed, sdt) {
    const dx = p.tx - p.x, dy = p.ty - p.y, d = Math.hypot(dx, dy);
    if (d < 0.05) return true;
    const step = Math.min(d, speed * sdt);
    p.x += (dx / d) * step; p.y += (dy / d) * step;
    const last = p.trail[p.trail.length - 1];
    if (!last || Math.hypot(last[0] - p.x, last[1] - p.y) > 0.25) { p.trail.push([p.x, p.y]); if (p.trail.length > 22) p.trail.shift(); }
    return d - step < 0.05;
  }

  // Customers walk the aisles instead of through the cars: down to the nearest aisle, across, and up to the car.
  const AISLES = [6.3, 11.6], CORRIDOR_X = 11.6;
  const nearestAisle = (y) => AISLES.reduce((a, b) => (Math.abs(b - y) < Math.abs(a - y) ? b : a));
  function goTo(p, tx, ty) {
    const a = nearestAisle(p.y), b = nearestAisle(ty);
    p.route = [[p.x, a]];
    if (a !== b) p.route.push([CORRIDOR_X, a], [CORRIDOR_X, b]);
    p.route.push([tx, b], [tx, ty]);
    [p.tx, p.ty] = p.route.shift();
  }
  function walk(p, speed, sdt) {
    if (!moveToward(p, speed, sdt)) return false;
    if (!p.route || !p.route.length) return true;
    [p.tx, p.ty] = p.route.shift();
    return false;
  }

  function addGuest(opts = {}) {
    const zone = opts.zone || pickZone();
    const [tx, ty] = zonePoint(byId[zone]);
    const g = {
      id: `C${nextGuest++}`, role: "customer",
      x: opts.x ?? rnd(10.8, 13.2), y: opts.y ?? 15.6, tx, ty, trail: [],
      state: opts.state || "walk", zone, dwellLeft: opts.dwellLeft || 0, zoneDwell: opts.zoneDwell || 0,
      enteredAt: t - (opts.age || 0), greetedAt: opts.greeted ? t - 60 : null, servedBy: null,
      flaggedUnserved: false, visited: [zone],
    };
    if (g.state === "dwell") { g.x = tx; g.y = ty; }
    if (g.state === "walk") goTo(g, tx, ty);
    people.push(g);
    return g;
  }

  const staff = [
    { id: "S1", name: "Aziz", home: [18.6, 11.6] },
    { id: "S2", name: "Dilnoza", home: [12.2, 6.2] },
    { id: "S3", name: "Jasur", home: [4.2, 6.6] },
  ].map((s) => ({ ...s, role: "staff", x: s.home[0], y: s.home[1], tx: s.home[0], ty: s.home[1], trail: [], state: "idle", target: null, engagedSec: 0, endAt: 0, idleT: rnd(5, 20) }));
  people.push(...staff);

  // Start mid-afternoon with a few people already on the floor.
  addGuest({ zone: "lc300", state: "dwell", dwellLeft: 70, zoneDwell: 48, age: 160 });
  const engagedGuest = addGuest({ zone: "rav4", state: "dwell", dwellLeft: 200, zoneDwell: 80, age: 260, greeted: true });
  addGuest({ zone: "camry", state: "dwell", dwellLeft: 40, zoneDwell: 15, age: 90, greeted: true });
  addGuest({ zone: "cross", state: "walk", x: 9.5, y: 12, age: 30 });
  Object.assign(staff[1], { state: "engaged", target: engagedGuest, engagedSec: 70, endAt: 260, x: engagedGuest.x + 1.1, y: engagedGuest.y });
  engagedGuest.servedBy = "S2";
  [
    ["coach", "Dilnoza greeted C24 after 0:39"],
    ["pulse", "C23 left after 8:12, visited Camry, RAV4, reception"],
    ["coach", "Aziz and C21 passed 2 min at Land Cruiser 300, counts as engaged"],
    ["pulse", "C25 came in"],
    ["coach", "Dilnoza greeted C25 after 0:51"],
  ].forEach(([lens, text], i) => events.unshift({ time: clock(t - (5 - i) * 47), lens, text }));

  function sendHome(s) { s.state = "return"; s.target = null; goTo(s, s.home[0], s.home[1]); }

  function freeStaffFor(g) {
    return staff.filter((s) => s.state === "idle").sort((a, b) => dist(a, g) - dist(b, g))[0];
  }

  function tick(sdt) {
    t += sdt;
    const guests = people.filter((p) => p.role === "customer");

    if (guests.length < 7 && Math.random() < sdt / 22) { const g = addGuest(); log("pulse", `${g.id} came in`); }

    for (const g of guests) {
      const engaged = staff.find((s) => s.target === g && s.state === "engaged");
      if (g.state === "walk") {
        if (walk(g, 1.0, sdt)) { g.state = "dwell"; g.dwellLeft = rnd(30, 190); g.zoneDwell = 0; }
      } else if (g.state === "dwell") {
        g.zoneDwell += sdt;
        g.dwellLeft -= engaged ? 0 : sdt;
        const nearStaff = staff.some((s) => dist(s, g) < 2);
        if (g.zoneDwell > 60 && !nearStaff && !g.greetedAt && !g.flaggedUnserved) {
          g.flaggedUnserved = true;
          log("pulse", `${g.id} has been at the ${byId[g.zone].label} for 1:00, no salesperson nearby`);
        }
        if (g.dwellLeft <= 0) {
          if (Math.random() < 0.5 && g.visited.length < 3) {
            g.zone = pickZone(g.zone); g.visited.push(g.zone);
            goTo(g, ...zonePoint(byId[g.zone])); g.state = "walk";
          } else { g.state = "leave"; goTo(g, rnd(10.8, 13.2), 16.4); }
        }
      } else if (g.state === "leave") {
        if (walk(g, 1.1, sdt)) {
          people.splice(people.indexOf(g), 1);
          const stay = mmss(Math.round(t - g.enteredAt));
          if (!g.greetedAt) { unservedToday++; log("coach", `${g.id} left after ${stay} without being greeted`); }
          else log("pulse", `${g.id} left after ${stay}, visited ${g.visited.map((z) => byId[z].label).join(", ")}`);
          staff.forEach((s) => { if (s.target === g) sendHome(s); });
          continue;
        }
      }
      const c = Math.min(HEAT_COLS - 1, Math.max(0, Math.floor((g.x / FLOOR.widthM) * HEAT_COLS)));
      const r = Math.min(HEAT_ROWS - 1, Math.max(0, Math.floor((g.y / FLOOR.depthM) * HEAT_ROWS)));
      if (g.state === "dwell") heat[r * HEAT_COLS + c] += sdt;
    }

    for (const s of staff) {
      const g = s.target;
      if (s.state === "idle") {
        s.idleT -= sdt;
        if (s.idleT <= 0) { s.tx = s.home[0] + rnd(-1.2, 1.2); s.ty = s.home[1] + rnd(-0.8, 0.8); s.idleT = rnd(15, 35); }
        moveToward(s, 0.5, sdt);
      } else if (s.state === "approach") {
        if (!people.includes(g)) { sendHome(s); continue; }
        s.tx = g.x + 1.0; s.ty = g.y + 0.3;
        moveToward(s, 1.4, sdt);
        if (dist(s, g) < 2) {
          const waited = Math.round(t - g.enteredAt);
          g.greetedAt = t; g.servedBy = s.id; greets.push(waited); if (greets.length > 40) greets.shift();
          log("coach", `${s.name} greeted ${g.id} after ${mmss(waited)}`);
          s.state = "engaged"; s.engagedSec = 0; s.endAt = rnd(110, 360);
        }
      } else if (s.state === "engaged") {
        if (!people.includes(g)) { sendHome(s); continue; }
        s.tx = g.x + 1.1; s.ty = g.y + 0.2;
        moveToward(s, 1.2, sdt);
        const before = s.engagedSec;
        s.engagedSec += sdt;
        if (before < 120 && s.engagedSec >= 120) log("coach", `${s.name} and ${g.id} passed 2 min at the ${byId[g.zone].label}, counts as engaged`);
        if (s.engagedSec >= s.endAt) {
          if (Math.random() < 0.45) { log("pulse", `${g.id} walked to reception with ${s.name}`); goTo(g, 20.6, 12.2); g.state = "leave"; }
          else g.dwellLeft = Math.min(g.dwellLeft, 8);
          sendHome(s);
        }
      } else if (s.state === "return") {
        if (walk(s, 1.1, sdt)) { s.state = "idle"; s.idleT = rnd(5, 20); }
      }
    }

    for (const g of people) {
      if (g.role !== "customer" || g.greetedAt || g.state === "leave") continue;
      if (t - g.enteredAt < 12 || staff.some((s) => s.target === g)) continue;
      const s = freeStaffFor(g);
      if (s) { s.state = "approach"; s.target = g; }
    }
  }

  // Start the live heat from a typical day so the Customers lens is not empty on first load.
  dayHeat().values.forEach((v, i) => (heat[i] = v * 40));

  let lastReal = performance.now();
  function advance() {
    const now = performance.now();
    let dt = Math.min(0.25, (now - lastReal) / 1000);
    lastReal = now;
    let sdt = dt * SPEED;
    while (sdt > 0) { const step = Math.min(0.5, sdt); tick(step); sdt -= step; }
  }

  const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : 0; };

  function liveSnapshot() {
    advance();
    const guests = people.filter((p) => p.role === "customer");
    const waiting = guests.filter((g) => !g.greetedAt && g.state !== "leave");
    const counts = {};
    guests.filter((g) => g.state === "dwell").forEach((g) => { (counts[g.zone] ||= []).push(g.zoneDwell); });
    const hot = Object.entries(counts).sort((a, b) => b[1].length - a[1].length || median(b[1]) - median(a[1]))[0];
    return {
      clock: clock(t),
      people: people.map((p) => ({
        id: p.id, role: p.role, name: p.name || null, x: p.x, y: p.y,
        trail: p.trail.slice(),
        waitingSec: p.role === "customer" && !p.greetedAt ? Math.round(t - p.enteredAt) : null,
        zone: p.zone || null,
        zoneDwellSec: p.role === "customer" && p.state === "dwell" ? Math.round(p.zoneDwell) : null,
        engagedWith: p.role === "staff" && p.state === "engaged" ? p.target.id : null,
        engagedSec: p.role === "staff" && p.state === "engaged" ? Math.round(p.engagedSec) : null,
      })),
      kpis: {
        medianGreetSec: median(greets),
        greetTargetSec: 60,
        waitingNow: waiting.length,
        longestWaitSec: waiting.length ? Math.max(...waiting.map((g) => Math.round(t - g.enteredAt))) : 0,
        onFloor: guests.length,
        staffOnFloor: staff.length,
        staffFree: staff.filter((s) => s.state === "idle").length,
        hottestZone: hot ? { label: byId[hot[0]].label, people: hot[1].length, medianDwellSec: Math.round(median(hot[1])) } : null,
        unservedToday,
      },
      heat: { cols: HEAT_COLS, rows: HEAT_ROWS, values: Array.from(heat) },
      events: events.slice(0, 14),
    };
  }

  // A day's worth of customer heat for the Visitors page: warm rings around each car, weighted by interest.
  function dayHeat() {
    const values = new Array(HEAT_COLS * HEAT_ROWS).fill(0);
    const w = { camry: 0.55, rav4: 0.8, lc300: 1, cross: 0.45, hilux: 0.22 };
    for (let r = 0; r < HEAT_ROWS; r++) for (let c = 0; c < HEAT_COLS; c++) {
      const x = ((c + 0.5) / HEAT_COLS) * FLOOR.widthM, y = ((r + 0.5) / HEAT_ROWS) * FLOOR.depthM;
      let v = 0;
      for (const z of FLOOR.zones) {
        const dx = Math.max(z.x - x, 0, x - (z.x + z.w)), dy = Math.max(z.y - y, 0, y - (z.y + z.h));
        const d = Math.hypot(dx, dy);
        v += w[z.id] * Math.exp(-((d - 0.8) ** 2) / 0.9) * (d > 0 ? 1 : 0.15);
      }
      v += 0.35 * Math.exp(-((x - 21) ** 2 + (y - 12) ** 2) / 4);   // reception
      v += 0.25 * Math.exp(-((x - 3.6) ** 2 + (y - 13.5) ** 2) / 5); // lounge
      v += 0.12 * Math.exp(-((x - 12) ** 2) / 3) * (y > 10 ? 1 : 0.2); // entrance lane
      values[r * HEAT_COLS + c] = v;
    }
    return { cols: HEAT_COLS, rows: HEAT_ROWS, values };
  }

  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const sessions = [
    { id: "s-0311", name: "Camera 2 · afternoon", recorded: "11 Oct, 12:00", lengthSec: 7200, customers: 16, medianGreetSec: 51, status: "ready" },
    { id: "s-0310", name: "Camera 2 · morning", recorded: "11 Oct, 10:00", lengthSec: 7200, customers: 19, medianGreetSec: 44, status: "ready" },
    { id: "s-0309", name: "Camera 1 · entrance", recorded: "10 Oct, 16:00", lengthSec: 5400, customers: 23, medianGreetSec: 63, status: "ready" },
    { id: "s-0308", name: "Camera 2 · Saturday peak", recorded: "5 Oct, 11:00", lengthSec: 10800, customers: 41, medianGreetSec: 82, status: "ready" },
  ];

  window.Mock = {
    getFloorplan: async () => FLOOR,
    getLive: async () => liveSnapshot(),

    getTeam: async () => ({
      period: "Today, 10:00–14:20 · 3 salespeople on shift · 35 visits",
      lead: [
        { label: "Top score", value: "Aziz", text: "Fastest greets on the floor and one walk-out all day." },
        { label: "Team median greet", value: "0:48", text: "Target is 1:00. Met on 26 of 35 visits." },
        { label: "Engaged visits", value: "21 of 35", text: "A salesperson stayed within 2 m for over 2 minutes." },
        { label: "Walk-outs", value: "5", text: "Three happened while all three salespeople were busy." },
      ],
      staff: [
        { id: "S1", name: "Aziz", since: "On shift since 09:45", greets: 14, medianGreetSec: 41, engaged: 9, walkouts: 1, score: 86, trend: [72, 75, 80, 78, 83, 84, 86] },
        { id: "S2", name: "Dilnoza", since: "On shift since 10:00", greets: 12, medianGreetSec: 47, engaged: 8, walkouts: 1, score: 81, trend: [70, 74, 73, 77, 79, 80, 81] },
        { id: "S3", name: "Jasur", since: "On shift since 11:30", greets: 9, medianGreetSec: 74, engaged: 4, walkouts: 3, score: 58, trend: [66, 63, 61, 60, 59, 57, 58] },
      ],
      suggestions: [
        { trackId: "T41", reason: "Behind the reception desk 64% of the time. Same dark jacket as Aziz and Dilnoza.", confidence: 0.91 },
        { trackId: "T57", reason: "Visited four cars in six minutes and stood next to three different customers.", confidence: 0.78 },
        { trackId: "T63", reason: "Was already inside at 09:52, before opening. Never crossed the entrance.", confidence: 0.7 },
      ],
    }),
    confirmStaff: async (trackId, isStaff, name) => { await wait(250); return { ok: true, trackId, isStaff, name }; },

    getVisitors: async () => ({
      period: "Today, 10:00–14:20 · one camera covering the main floor",
      lead: [
        { label: "Visitors", value: "35", text: "Peak was 9 at once, at 12:40." },
        { label: "Median visit", value: "7:40", text: "Visitors who reached reception stayed 14:05." },
        { label: "Considered a car", value: "24", text: "Spent over 60 s at one car. Most often the Land Cruiser 300." },
        { label: "Left unserved", value: "5", text: "Lingered over a minute and nobody came within 2 m." },
      ],
      zones: [
        { label: "Land Cruiser 300", visits: 21, medianDwellSec: 96, over60: 0.62, approached: 0.59, unserved: 3 },
        { label: "RAV4", visits: 26, medianDwellSec: 74, over60: 0.48, approached: 0.77, unserved: 1 },
        { label: "Camry", visits: 18, medianDwellSec: 51, over60: 0.33, approached: 0.83, unserved: 0 },
        { label: "Corolla Cross", visits: 14, medianDwellSec: 43, over60: 0.29, approached: 0.71, unserved: 1 },
        { label: "Hilux", visits: 9, medianDwellSec: 22, over60: 0.11, approached: 0.67, unserved: 0 },
      ],
      journeys: [
        { path: ["Entrance", "RAV4", "Land Cruiser 300", "Exit"], count: 9 },
        { path: ["Entrance", "Camry", "RAV4", "Reception"], count: 7 },
        { path: ["Entrance", "Land Cruiser 300", "Exit"], count: 6 },
        { path: ["Entrance", "Corolla Cross", "Camry", "Lounge", "Reception"], count: 5 },
        { path: ["Entrance", "Hilux", "Lounge", "Exit"], count: 3 },
      ],
      unserved: [
        { time: "14:02", zone: "Land Cruiser 300", dwellSec: 158, note: "Photographed the price board, then left." },
        { time: "13:51", zone: "Corolla Cross", dwellSec: 71, note: "Two salespeople were in engagements." },
        { time: "13:05", zone: "Land Cruiser 300", dwellSec: 96, note: "Sat in the driver's seat for 40 s." },
        { time: "12:40", zone: "RAV4", dwellSec: 134, note: "Couple with a child. All three salespeople were busy." },
        { time: "11:12", zone: "Land Cruiser 300", dwellSec: 212, note: "Walked around the car twice." },
      ],
      heat: dayHeat(),
    }),

    getInsights: async () => ([
      { lens: "Customers × Salespeople", value: "41%", text: "of Land Cruiser 300 visitors who stayed over a minute were never approached. It is the most-considered car and the least-served.", act: "Keep one salesperson between the Land Cruiser and the RAV4 after 12:00." },
      { lens: "Salespeople × Customers", value: "1:38", text: "Median greet between 12:30 and 13:30, double the daily figure. Two of three salespeople were already in engagements, and four of five walk-outs fell in this hour.", act: "Move one lunch break out of this window." },
      { lens: "Customers × Salespeople", value: "3.1×", text: "Visitors engaged for over 2 minutes were 3.1 times as likely to walk to reception as visitors who were only greeted.", act: "Coach greetings that turn into a walkaround, not just a hello." },
      { lens: "Salespeople", value: "0:48", text: "Team median greet today, inside the 1:00 target on 26 of 35 visits. Jasur's median is 1:14.", act: "Pair Jasur with Aziz for the next Saturday peak." },
      { lens: "Customers", value: "0:22", text: "Median dwell at the Hilux. Most visitors pass it on the way to the lounge without stopping.", act: "Swap the Hilux and the Corolla Cross for a week and compare." },
    ]),

    getSessions: async () => sessions.slice(),
    analyzeVideo: async (file, onStep) => {
      const steps = ["Finding people", "Following each person across frames", "Suggesting who is staff", "Scoring visits and routes"];
      for (let i = 0; i < steps.length; i++) { onStep(i, steps[i]); await wait(900 + Math.random() * 600); }
      const s = { id: `s-${Date.now()}`, name: file.name.replace(/\.[^.]+$/, ""), recorded: "Just now", lengthSec: 1260, customers: 6, medianGreetSec: 57, status: "ready" };
      sessions.unshift(s);
      return s;
    },
  };
})();
