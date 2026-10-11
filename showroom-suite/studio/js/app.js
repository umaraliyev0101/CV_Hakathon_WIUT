// Views: routing, rendering, and the few interactions (lens, upload, staff confirmation).
(function () {
  const C = window.SUITE_CONFIG;
  const D = window.Suite.data;
  const $ = (s, el = document) => el.querySelector(s);
  const pad2 = (n) => String(n).padStart(2, "0");
  const mmss = (s) => (s == null ? "–" : `${Math.floor(s / 60)}:${pad2(Math.round(s) % 60)}`);
  const hmm = (s) => (s >= 3600 ? `${Math.floor(s / 3600)} h ${pad2(Math.round((s % 3600) / 60))} min` : `${Math.round(s / 60)} min`);
  const pct = (x) => `${Math.round(x * 100)}%`;
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const lensTag = (lens) => (lens === "coach" ? `<span class="tag t-staff">Salespeople</span>` : `<span class="tag t-guest">Customers</span>`);

  $("#site-name").textContent = C.showroom;

  // ---- source chips -------------------------------------------------------------------------
  const SOURCE_TEXT = {
    floor: "Coach + Pulse", sessions: "Coach + Pulse", team: "Coach", visitors: "Pulse", demand: "Pulse + Coach", insights: "Coach + Pulse",
  };
  function paintSource(section, st) {
    document.querySelectorAll(`[data-source="${section}"]`).forEach((el) => {
      el.classList.toggle("is-live", st.state === "live");
      el.classList.toggle("is-error", st.state === "error");
      el.textContent = st.state === "live" ? `Live · ${SOURCE_TEXT[section]}` : st.state === "error" ? "Server not reachable · showing demo data" : "Demo data";
      el.title = st.message || "";
    });
  }
  window.Suite.onStatus(paintSource);
  Object.keys(SOURCE_TEXT).forEach((s) => paintSource(s, { state: C.live[s] ? "live" : "demo" }));

  let toastTimer;
  function toast(text) {
    const t = $("#toast"); t.textContent = text; t.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), 2600);
  }

  // ---- routing ------------------------------------------------------------------------------
  const VIEWS = ["live", "replays", "team", "visitors", "demand", "insights"];
  const loaded = {};
  function route() {
    const hash = location.hash.slice(1);
    const sessionId = hash.startsWith("session/") ? decodeURIComponent(hash.slice(8)) : null;
    const name = sessionId ? "session" : VIEWS.includes(hash) ? hash : "live";
    const navName = sessionId ? "replays" : name;   // a session page belongs to Replays in the nav
    if (name !== "session") player.pause();
    document.querySelectorAll("[data-view]").forEach((v) => (v.hidden = v.dataset.view !== name));
    document.querySelectorAll(".nav a").forEach((a) => a.toggleAttribute("aria-current", a.getAttribute("href") === `#${navName}`));
    document.querySelectorAll(".nav a[aria-current]").forEach((a) => a.setAttribute("aria-current", "page"));
    if (sessionId) { openSession(sessionId); return; }
    if (!loaded[name]) { loaded[name] = true; LOADERS[name](); }
    if (name === "visitors" && visitorsData) requestAnimationFrame(drawVisitorsHeat);
    if (name === "demand" && demandData) requestAnimationFrame(drawDemand);
  }
  window.addEventListener("hashchange", route);

  // ---- live floor ---------------------------------------------------------------------------
  let floorplan = null, live = null, lens = "all";
  const floorCanvas = $("#floor");

  document.querySelectorAll("[data-lens]").forEach((b) => b.addEventListener("click", () => {
    lens = b.dataset.lens;
    document.querySelectorAll("[data-lens]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderKpis();
  }));

  function renderKpis() {
    if (!live) return;
    const k = live.kpis;
    const greetOk = k.medianGreetSec <= k.greetTargetSec;
    const tiles = [
      { lens: "coach", label: "Median greet today", value: mmss(k.medianGreetSec),
        text: greetOk ? `Inside the ${mmss(k.greetTargetSec)} target.` : `Over the ${mmss(k.greetTargetSec)} target. Waits are building up.` },
      { lens: "coach", label: "Waiting to be greeted", value: k.waitingNow, unit: k.waitingNow === 1 ? "person" : "people",
        text: k.waitingNow ? `Longest wait ${mmss(k.longestWaitSec)}. ${k.staffFree} of ${k.staffOnFloor} salespeople free.` : `Everyone on the floor has been greeted. ${k.staffFree} of ${k.staffOnFloor} salespeople free.` },
      { lens: "pulse", label: "Customers on the floor", value: k.onFloor,
        text: `${k.unservedToday} left unserved today after lingering over a minute.` },
      { lens: "pulse", label: "Holding attention now", value: k.hottestZone ? k.hottestZone.label : "–", small: true,
        text: k.hottestZone ? `${k.hottestZone.people} ${k.hottestZone.people === 1 ? "person" : "people"} at this car, longest stay ${mmss(k.hottestZone.medianDwellSec)}.` : "Nobody is stopped at a car right now." },
    ];
    const order = lens === "pulse" ? [2, 3, 0, 1] : [0, 1, 2, 3];
    $("#live-kpis").innerHTML = order.map((i) => tiles[i]).map((t) => `
      <div class="kpi">
        <div class="label"><span>${esc(t.label)}</span>${lensTag(t.lens)}</div>
        <div class="value"${t.small ? ' style="font-size:clamp(30px,3vw,40px);letter-spacing:-0.02em"' : ""}>${esc(t.value)}${t.unit ? `<small>${esc(t.unit)}</small>` : ""}</div>
        <p class="finding">${esc(t.text)}</p>
      </div>`).join("");
  }

  let lastLogKey = "";
  function renderLog() {
    const ev = live.events.filter((e) => lens === "all" || e.lens === lens).slice(0, 8);
    const key = ev.map((e) => e.time + e.text).join("|");
    if (key === lastLogKey) return;
    lastLogKey = key;
    $("#live-log").innerHTML = ev.map((e) => `
      <div class="log-row"><span class="mono">${esc(e.time)}</span>${lensTag(e.lens)}<span>${esc(e.text)}</span></div>`).join("")
      || `<p class="note">Nothing yet for this lens.</p>`;
  }

  let lastKpi = 0;
  async function pollLive() {
    try {
      live = await D.getLive();
      $("#clock").textContent = live.clock;
      const now = performance.now();
      if (now - lastKpi > 900) { renderKpis(); lastKpi = now; }
      renderLog();
    } finally {
      setTimeout(pollLive, C.live.floor ? C.pollMs : 250);
    }
  }
  function frame() {
    if (!$("#view-live").hidden) Floor.drawStage(floorCanvas, floorplan, live, lens);
    requestAnimationFrame(frame);
  }

  // ---- replays ------------------------------------------------------------------------------
  async function renderSessions() {
    const list = await D.getSessions();
    $("#session-count").textContent = `${list.length} sessions`;
    $("#sessions").innerHTML = list.map((s) => `
      <tr class="row-link" data-session="${esc(s.id)}" tabindex="0" title="Open this session's results">
        <td><a class="who" href="#session/${encodeURIComponent(s.id)}">${esc(s.name)}</a><span class="sub mono">${esc(s.id)}</span></td>
        <td>${esc(s.recorded)}</td>
        <td class="n">${hmm(s.lengthSec)}</td>
        <td class="n">${esc(s.customers)}</td>
        <td class="n">${mmss(s.medianGreetSec)}</td>
        <td><span class="status-pill ${esc(s.status)}">${s.status === "ready" ? "Ready" : s.status === "running" ? "Analysing" : "Failed"}</span></td>
      </tr>`).join("");
  }
  async function analyze(file) {
    if (!file) return;
    const prog = $("#progress"), bars = prog.querySelectorAll(".steps div");
    prog.hidden = false; bars.forEach((b) => b.classList.remove("on"));
    try {
      const s = await D.analyzeVideo(file, (i, label) => {
        bars.forEach((b, j) => b.classList.toggle("on", j <= i));
        $("#progress-text").textContent = `${label}…`;
      });
      $("#progress-text").textContent = `Done. ${s.customers} customers, median greet ${mmss(s.medianGreetSec)}.`;
      toast(s.warning || `“${s.name}” is ready`);
      renderSessions();
      location.hash = `#session/${encodeURIComponent(s.id)}`;   // straight to its results
    } catch (err) {
      $("#progress-text").textContent = `Could not analyse this video: ${err.message}`;
    }
  }
  $("#video-input").addEventListener("change", (e) => analyze(e.target.files[0]));
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
  drop.addEventListener("drop", (e) => analyze(e.dataTransfer.files[0]));

  // ---- team ---------------------------------------------------------------------------------
  function spark(values) {
    const w = 88, h = 24, min = Math.min(...values) - 2, max = Math.max(...values) + 2;
    const pts = values.map((v, i) => [(i / (values.length - 1)) * (w - 4) + 2, h - 2 - ((v - min) / (max - min)) * (h - 4)]);
    const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
    const [ex, ey] = pts[pts.length - 1];
    return `<svg class="spark" viewBox="0 0 ${w} ${h}" aria-hidden="true"><path d="${d}" fill="none" stroke="currentColor" stroke-width="1.4"/><circle cx="${ex}" cy="${ey}" r="2.4" fill="currentColor"/></svg>`;
  }
  function leadHtml(items) {
    return items.map((l) => `<div><span class="label">${esc(l.label)}</span><div class="value">${esc(l.value)}</div><p>${esc(l.text)}</p></div>`).join("");
  }
  async function renderTeam() {
    const t = await D.getTeam();
    $("#team-period").textContent = t.period;
    $("#team-lead").innerHTML = leadHtml(t.lead);
    $("#team-rows").innerHTML = t.staff.map((s) => {
      const up = s.trend[s.trend.length - 1] >= s.trend[0];
      return `<tr>
        <td><span class="who">${esc(s.name)}</span><span class="sub">${esc(s.since)}</span></td>
        <td class="n">${esc(s.greets)}</td>
        <td class="n ${s.medianGreetSec > 60 ? "poor" : ""}">${mmss(s.medianGreetSec)}</td>
        <td class="n">${esc(s.engaged)}</td>
        <td class="n ${s.walkouts >= 3 ? "poor" : ""}">${esc(s.walkouts)}</td>
        <td class="${up ? "good" : "poor"}">${spark(s.trend)}</td>
        <td class="n"><div class="score"><div class="bar-s"><i style="width:${s.score}%"></i></div><span>${esc(s.score)}</span></div></td>
      </tr>`;
    }).join("");
    renderSuggestions(t.suggestions);
  }
  function renderSuggestions(list) {
    $("#suggest-count").textContent = `${list.length} to review`;
    $("#suggest").innerHTML = list.map((s) => `
      <div class="suggest-row" data-track="${esc(s.trackId)}">
        <div class="thumb">${s.thumbUrl ? `<img src="${esc(s.thumbUrl)}" alt="">` : ""}</div>
        <div>
          <span class="who">Person ${esc(s.trackId)}</span> <span class="label">· ${Math.round(s.confidence * 100)}% sure</span>
          <p>${esc(s.reason)}</p>
          <div class="actions">
            <button type="button" class="btn" data-act="yes">Yes, staff</button>
            <button type="button" class="link" data-act="no">Customer</button>
          </div>
        </div>
      </div>`).join("");
  }
  $("#suggest").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const row = btn.closest(".suggest-row"), id = row.dataset.track, isStaff = btn.dataset.act === "yes";
    row.querySelectorAll("button").forEach((b) => (b.disabled = true));
    try {
      await D.confirmStaff(id, isStaff, null);
      row.querySelector(".thumb").classList.toggle("is-staff", isStaff);
      row.querySelector(".actions").innerHTML = `<span class="done">${isStaff ? "Marked as staff. Scoring updated." : "Marked as customer."}</span>`;
      toast(isStaff ? `Person ${id} added to the team` : `Person ${id} kept as a customer`);
    } catch (err) {
      row.querySelectorAll("button").forEach((b) => (b.disabled = false));
      toast(`Could not save: ${err.message}`);
    }
  });

  // ---- visitors -----------------------------------------------------------------------------
  let visitorsData = null;
  function drawVisitorsHeat() { Floor.drawHeat($("#heat"), floorplan, visitorsData.heat); }
  async function renderVisitors() {
    const v = await D.getVisitors();
    visitorsData = v;
    if (!floorplan) floorplan = await D.getFloorplan();
    $("#visitors-period").textContent = v.period;
    $("#visitors-lead").innerHTML = leadHtml(v.lead);
    const maxJ = Math.max(...v.journeys.map((j) => j.count));
    $("#journeys").innerHTML = v.journeys.map((j) => `
      <div class="journey">
        <div><div class="path">${j.path.map((p) => `<span>${esc(p)}</span>`).join("")}</div><div class="bar-j" style="width:${(j.count / maxJ) * 100}%"></div></div>
        <span class="num">${esc(j.count)}</span>
      </div>`).join("");
    $("#zone-rows").innerHTML = v.zones.map((z) => `
      <tr>
        <td class="who">${esc(z.label)}</td>
        <td class="n">${esc(z.visits)}</td>
        <td class="n">${mmss(z.medianDwellSec)}</td>
        <td class="n">${pct(z.over60)}</td>
        <td class="n ${z.approached < 0.65 ? "poor" : ""}">${pct(z.approached)}</td>
        <td class="n ${z.unserved >= 3 ? "poor" : ""}">${esc(z.unserved)}</td>
      </tr>`).join("");
    $("#unserved").innerHTML = v.unserved.map((u) => `
      <div class="log-row"><span class="mono">${esc(u.time)}</span><span class="tag">${mmss(u.dwellSec)} at car</span><span><b style="font-weight:500">${esc(u.zone)}</b>. ${esc(u.note)}</span></div>`).join("");
    requestAnimationFrame(drawVisitorsHeat);
  }

  // ---- demand -------------------------------------------------------------------------------
  // Each chart's "next step" is written from the computed numbers, so it stays true when live data arrives.
  let demandData = null;
  function drawDemand() {
    const d = demandData, Ch = window.Charts;
    $("#demand-period").textContent = d.period;

    const f = Ch.funnel($("#c-funnel"), d.funnel);
    $("#c-funnel .act").textContent = `${f.lost} visitors (${Ch.pct(f.worstLoss)}) stopped between “${f.from.label.toLowerCase()}” and “${f.to.label.toLowerCase()}”. ${f.to.fix || ""}`;

    const h = Ch.hours($("#c-hours"), d.hours);
    const dayTotals = d.hours.values.map((r) => r.reduce((a, b) => a + b, 0));
    const week = dayTotals.reduce((a, b) => a + b, 0);
    const topDay = dayTotals.indexOf(Math.max(...dayTotals));
    const fullDay = { Mon: "Monday", Tue: "Tuesday", Wed: "Wednesday", Thu: "Thursday", Fri: "Friday", Sat: "Saturday", Sun: "Sunday" };
    $("#c-hours .act").textContent = `${fullDay[d.hours.days[topDay]] || d.hours.days[topDay]} brings ${Ch.pct(dayTotals[topDay] / week)} of the week's visitors, peaking at ${h.peakHour}:00 with ${h.peakValue.toFixed(1)} per hour. End paid ads the evening before, run test-drive events from ${h.peakHour - 2}:00 to ${h.peakHour + 2}:00, and add one floater salesperson for that window.`;

    const sc = Ch.demandScatter($("#c-demand"), d.models);
    const many = sc.neglected.length > 1;
    $("#c-demand .act").textContent = sc.neglected.length
      ? `${sc.neglected.map((m) => m.label).join(" and ")} ${many ? "pull" : "pulls"} above-median interest, but only ${sc.neglected.map((m) => Ch.pct(m.approached)).join(" / ")} of those visitors get approached. Fix floor coverage there before advertising ${many ? "them" : "it"}, or the extra traffic walks out unserved.`
      : `Every high-interest car is well covered. Put ad money behind the model with the most interest: ${[...d.models].sort((a, b) => b.considered - a.considered)[0].label}.`;

    const cl = Ch.campaign($("#c-campaign"), d.campaign);
    $("#c-campaign .act").textContent = cl.lift - cl.control > 0.1
      ? `The campaign brought people to the car: ${d.campaign.model} rose ${Ch.signed(cl.lift)} while the rest of the floor moved ${Ch.signed(cl.control)}. Keep it running, and keep a salesperson by the ${d.campaign.model} while it does.`
      : `No clear effect yet: ${d.campaign.model} moved ${Ch.signed(cl.lift)} while the rest of the floor moved ${Ch.signed(cl.control)}. Change the creative or the offer before adding budget.`;
  }
  async function renderDemand() {
    demandData = await D.getDemand();
    requestAnimationFrame(drawDemand);
  }

  // ---- insights -----------------------------------------------------------------------------
  async function renderInsights() {
    const list = await D.getInsights();
    $("#insight-list").innerHTML = list.map((i) => `
      <article class="insight">
        <span class="label">${esc(i.lens)}</span>
        <div class="value">${esc(i.value)}</div>
        <p>${esc(i.text)}</p>
        <p class="act">${esc(i.act)}</p>
      </article>`).join("");
  }

  const openRow = (e) => {
    const tr = e.target.closest("tr[data-session]");
    if (!tr || e.target.closest("a")) return;
    if (e.type === "keydown" && e.key !== "Enter") return;
    location.hash = `#session/${encodeURIComponent(tr.dataset.session)}`;
  };
  $("#sessions").addEventListener("click", openRow);
  $("#sessions").addEventListener("keydown", openRow);

  // ---- session results ----------------------------------------------------------------------
  // The video with its moments on a timeline underneath, plus that session's numbers. Demo sessions play the user's
  // own upload with invented results, and say so; live sessions play the servers' annotated video.
  const player = $("#player");
  const KIND = { greet: "Greet", engagement: "Engagement", walkout: "Walk-out", unserved: "Unserved customer" };
  let session = null, sessionToken = 0;

  const duration = () => (Number.isFinite(player.duration) && player.duration > 0 ? player.duration : session?.lengthSec || 1);

  function seek(t) {
    if (!session?.videos.length) return;
    player.currentTime = Math.max(0, t - 1);          // a second of lead-in, so the moment is seen arriving
    player.play().catch(() => {});
  }

  function showVideo(i) {
    const v = session.videos[i], at = player.currentTime || 0, playing = !player.paused;
    document.querySelectorAll("#video-switch button").forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
    player.src = v.url;
    player.addEventListener("loadedmetadata", () => { player.currentTime = Math.min(at, duration()); if (playing) player.play().catch(() => {}); drawTimeline(); }, { once: true });
  }

  function drawTimeline() {
    const d = duration();
    const left = (t) => `${Math.min(100, (t / d) * 100)}%`;
    // Top lane: one clickable mark per moment. Bottom lane: how long engagements and unserved waits lasted
    // (drawn only, so a long span never covers another moment's mark).
    $("#tl-track").innerHTML = `<i class="tl-head" id="tl-head"></i>`
      + session.moments.filter((m) => m.durSec).map((m) => `
      <i class="tl-span m-${esc(m.kind)}" style="left:${left(m.t)};width:${Math.max(0.4, (m.durSec / d) * 100)}%"></i>`).join("")
      + session.moments.map((m, i) => `
      <button type="button" class="tl-mark m-${esc(m.kind)}" data-moment="${i}" style="left:${left(m.t)}"
              title="${mmss(m.t)} · ${esc(KIND[m.kind] || m.kind)}: ${esc(m.text)}"
              aria-label="${mmss(m.t)}, ${esc(KIND[m.kind] || m.kind)}. ${esc(m.text)}"></button>`).join("");
    const step = d > 1800 ? 600 : d > 600 ? 120 : d > 180 ? 30 : 10;
    const ticks = [];
    for (let t = 0; t <= d; t += step) ticks.push(`<span style="left:${left(t)}">${mmss(t)}</span>`);
    $("#tl-axis").innerHTML = ticks.join("");
    tick();
  }

  function tick() {
    if (!session) return;
    const t = player.currentTime || 0, d = duration();
    const head = $("#tl-head");
    if (head) head.style.left = `${Math.min(100, (t / d) * 100)}%`;
    $("#player-time").textContent = `${mmss(t)} / ${mmss(d)}`;
    // the moment on screen right now, shown over the video (a few seconds, or the whole span)
    const now = session.moments.filter((m) => t >= m.t && t < m.t + Math.max(4, m.durSec || 0)).pop();
    const cap = $("#moment-caption");
    cap.hidden = !now || !session.videos.length;
    if (now) cap.innerHTML = `<span class="mono">${mmss(now.t)}</span><b>${esc(KIND[now.kind] || now.kind)}</b>${esc(now.text)}${session.demo ? `<em>demo</em>` : ""}`;
    document.querySelectorAll("#moment-list .log-row").forEach((r) => r.classList.toggle("is-now", now != null && +r.dataset.moment === session.moments.indexOf(now)));
  }
  player.addEventListener("timeupdate", tick);
  player.addEventListener("seeked", tick);
  player.addEventListener("loadedmetadata", drawTimeline);

  function onMoment(e) {
    const el = e.target.closest("[data-moment]");
    if (!el || (e.type === "keydown" && e.key !== "Enter")) return;
    seek(session.moments[+el.dataset.moment].t);
  }
  $("#tl-track").addEventListener("click", (e) => {
    if (e.target.closest("[data-moment]")) return onMoment(e);
    const r = e.currentTarget.getBoundingClientRect();         // a click on the bare track scrubs there
    if (session?.videos.length) player.currentTime = ((e.clientX - r.left) / r.width) * duration();
  });
  $("#moment-list").addEventListener("click", onMoment);
  $("#moment-list").addEventListener("keydown", onMoment);

  async function openSession(id) {
    const token = ++sessionToken;
    $("#session-name").textContent = "Loading…";
    let s;
    try { s = await D.getSession(id); } catch (err) { $("#session-name").textContent = `Could not open this session: ${err.message}`; return; }
    if (token !== sessionToken) return;                      // the user already went elsewhere
    session = s;
    const coach = s.moments.filter((m) => m.lens === "coach").length, pulse = s.moments.length - coach;
    $("#session-name").textContent = s.name;
    $("#session-meta").textContent = [s.recorded, s.lengthSec ? `${mmss(s.lengthSec)} of video` : "", `${coach} salespeople moments, ${pulse} customer moments`, s.staffNote].filter(Boolean).join(" · ");
    $("#session-demo").hidden = !s.demo;
    $("#demo-stamp").hidden = !s.demo || !s.videos.length;

    // video (+ a switch when the servers give more than one annotated video)
    $("#player-empty").hidden = s.videos.length > 0;
    player.hidden = !s.videos.length;
    $("#video-switch").innerHTML = s.videos.length > 1
      ? s.videos.map((v, i) => `<button type="button" data-video="${i}" aria-pressed="${i === 0}">${esc(v.label)}</button>`).join("")
      : s.videos.length ? `<button type="button" aria-pressed="true" disabled>${esc(s.videos[0].label)}</button>` : "";
    player.removeAttribute("src"); player.load();
    if (s.videos.length) showVideo(0);

    const walkouts = s.moments.filter((m) => m.kind === "walkout").length, unserved = s.moments.filter((m) => m.kind === "unserved").length;
    $("#session-kpis").innerHTML = [
      { lens: "pulse", label: "Customers", value: s.customers, text: `${unserved} left a car unserved after lingering.` },
      { lens: "coach", label: "Median greet", value: mmss(s.medianGreetSec), text: "From walking in to a salesperson within 1.5 m." },
      { lens: "coach", label: "Walk-outs", value: walkouts, text: walkouts ? "Left without being greeted. Click one on the timeline to see why." : "Every customer was greeted." },
    ].map((t) => `
      <div class="kpi">
        <div class="label"><span>${esc(t.label)}</span>${lensTag(t.lens)}</div>
        <div class="value">${esc(t.value)}</div>
        <p class="finding">${esc(t.text)}</p>
      </div>`).join("");

    $("#moment-count").textContent = `${s.moments.length} moments${s.demo ? " · demo" : ""}`;
    $("#moment-list").innerHTML = s.moments.map((m, i) => `
      <div class="log-row" data-moment="${i}" tabindex="0" role="button" aria-label="Jump to ${mmss(m.t)}">
        <span class="mono">${mmss(m.t)}</span><span class="tag m-tag m-${esc(m.kind)}">${esc(KIND[m.kind] || m.kind)}</span><span>${esc(m.text)}</span>
      </div>`).join("") || `<p class="note">Nothing happened in this video.</p>`;

    const b = s.busiestCar;
    $("#busiest").innerHTML = b ? `
      <div class="value">${esc(b.label)}</div>
      <dl>
        <div><dt>Visits</dt><dd>${esc(b.visits)}</dd></div>
        <div><dt>Median dwell</dt><dd>${mmss(b.medianDwellSec)}</dd></div>
        <div><dt>Left unserved</dt><dd class="${b.unserved ? "poor" : ""}">${esc(b.unserved)}</dd></div>
      </dl>` : `<p class="note">No car held anyone's attention.</p>`;

    $("#session-staff").innerHTML = s.staff.map((p) => `<tr>
        <td><span class="who">${esc(p.name)}</span></td>
        <td class="n">${esc(p.greets)}</td>
        <td class="n ${p.medianGreetSec > 60 ? "poor" : ""}">${mmss(p.medianGreetSec)}</td>
        <td class="n">${esc(p.engaged)}</td>
        <td class="n ${p.walkouts >= 2 ? "poor" : ""}">${esc(p.walkouts)}</td>
        <td class="n"><div class="score"><div class="bar-s"><i style="width:${p.score}%"></i></div><span>${esc(p.score)}</span></div></td>
      </tr>`).join("") || `<tr><td colspan="6" class="note">No salespeople confirmed for this session.</td></tr>`;

    const maxR = Math.max(1, ...s.routes.map((r) => r.count));
    $("#session-routes").innerHTML = s.routes.map((r) => `
      <div class="journey">
        <div><div class="path">${r.path.map((p) => `<span>${esc(p)}</span>`).join("")}</div><div class="bar-j" style="width:${(r.count / maxR) * 100}%"></div></div>
        <span class="num">${esc(r.count)}</span>
      </div>`).join("") || `<p class="note">No routes in this session.</p>`;

    drawTimeline();
  }
  $("#video-switch").addEventListener("click", (e) => { const b = e.target.closest("[data-video]"); if (b) showVideo(+b.dataset.video); });

  const LOADERS = {
    live: async () => { floorplan = floorplan || (await D.getFloorplan()); pollLive(); requestAnimationFrame(frame); },
    replays: renderSessions,
    team: renderTeam,
    visitors: renderVisitors,
    demand: renderDemand,
    insights: renderInsights,
  };

  let resizeT;
  window.addEventListener("resize", () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(() => {
      if (visitorsData && !$("#view-visitors").hidden) drawVisitorsHeat();
      if (demandData && !$("#view-demand").hidden) drawDemand();
    }, 120);
  });
  route();
})();
