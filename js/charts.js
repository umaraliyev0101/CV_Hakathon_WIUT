// Demand charts: funnel, busy hours, car demand vs service, campaign lift. Plain SVG/HTML, sized to their container.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
  const signed = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(Math.round(x * 100))}%`;

  // ---- shared tooltip: any element with data-tip shows it on hover or keyboard focus ----------
  const tip = document.createElement("div");
  tip.className = "chart-tip"; tip.hidden = true; tip.setAttribute("role", "tooltip");
  document.body.appendChild(tip);
  function place(x, y) {
    const r = tip.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > innerHeight - 8) top = y - r.height - 14;
    tip.style.left = `${Math.max(8, left)}px`; tip.style.top = `${Math.max(8, top)}px`;
  }
  function show(html, x, y) { tip.innerHTML = html; tip.hidden = false; place(x, y); }
  function hide() { tip.hidden = true; }
  function bindTips(root) {
    root.addEventListener("pointermove", (e) => {
      const el = e.target.closest("[data-tip]");
      el && root.contains(el) ? show(el.dataset.tip, e.clientX, e.clientY) : hide();
    });
    root.addEventListener("pointerleave", hide);
    root.addEventListener("focusin", (e) => {
      const el = e.target.closest("[data-tip]"); if (!el) return;
      const r = el.getBoundingClientRect(); show(el.dataset.tip, r.right, r.top);
    });
    root.addEventListener("focusout", hide);
  }

  const svgEl = (tag, attrs = {}, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  };
  function niceMax(v, step) { return Math.ceil(v / step) * step; }
  function table(head, rows) {
    return `<table><thead><tr>${head.map((h, i) => `<th${i ? ' class="n"' : ""}>${esc(h)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr>${r.map((c, i) => `<td${i ? ' class="n"' : ""}>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }

  // ---- 1. Funnel -----------------------------------------------------------------------------
  function funnel(root, stages) {
    const box = root.querySelector(".funnel");
    const top = stages[0].count;
    let worst = -1, worstLoss = -1;
    stages.forEach((s, i) => { if (i) { const loss = 1 - s.count / stages[i - 1].count; if (loss > worstLoss) { worstLoss = loss; worst = i; } } });
    box.innerHTML = stages.map((s, i) => {
      const prev = i ? stages[i - 1].count : null;
      const lost = prev != null ? prev - s.count : 0;
      const drop = i ? `<div class="f-drop${i === worst ? " worst" : ""}">↓ ${lost} lost here (${pct(lost / prev)})${i === worst ? " · biggest drop" : ""}</div>` : "";
      const t = `<b>${esc(s.label)}</b><br>${s.count} visitors · ${pct(s.count / top)} of everyone who came in${i ? `<br>${pct(lost / prev)} dropped since the step before` : ""}`;
      return `${drop}<div class="f-row" role="listitem" tabindex="0" data-tip="${esc(t)}">
          <div class="f-label">${esc(s.label)}${s.sub ? `<small>${esc(s.sub)}</small>` : ""}</div>
          <div class="f-track"><div class="f-bar" style="width:${(s.count / top) * 100}%"></div></div>
          <div class="num">${s.count}</div>
        </div>`;
    }).join("");
    bindTips(box);
    return { worst, worstLoss, lost: stages[worst - 1].count - stages[worst].count, from: stages[worst - 1], to: stages[worst] };
  }

  // ---- 2. Busy hours -------------------------------------------------------------------------
  function hours(root, h) {
    const grid = root.querySelector(".hours"), ramp = root.querySelector(".ramp");
    const steps = [1, 2, 3, 4, 5].map((i) => css(`--heat-${i}`));
    const flat = h.values.flat(), max = Math.max(...flat), min = Math.min(...flat);
    const bin = (v) => Math.min(4, Math.floor(((v - min) / (max - min || 1)) * 5));
    let peak = [0, 0];
    h.values.forEach((row, d) => row.forEach((v, k) => { if (v > h.values[peak[0]][peak[1]]) peak = [d, k]; }));
    grid.style.setProperty("--cols", h.hours.length);
    let html = `<span></span>${h.hours.map((hr, k) => `<span class="h-col">${k % 2 === 0 ? hr : ""}</span>`).join("")}`;
    h.days.forEach((day, d) => {
      html += `<span class="h-day">${esc(day)}</span>`;
      h.values[d].forEach((v, k) => {
        const t = `<b>${esc(day)} ${h.hours[k]}:00–${h.hours[k] + 1}:00</b><br>${v.toFixed(1)} visitors per hour on average`;
        html += `<span class="cell${d === peak[0] && k === peak[1] ? " peak" : ""}" tabindex="0" style="background:${steps[bin(v)]}" data-tip="${esc(t)}" aria-label="${esc(day)} ${h.hours[k]}:00, ${v.toFixed(1)} visitors"></span>`;
      });
    });
    grid.innerHTML = html;
    const edges = [0, 1, 2, 3, 4, 5].map((i) => min + ((max - min) * i) / 5);
    ramp.innerHTML = `<span>${edges[0].toFixed(0)}</span><span class="sw">${steps.map((c, i) => `<i style="background:${c}" title="${edges[i].toFixed(1)}–${edges[i + 1].toFixed(1)}"></i>`).join("")}</span><span>${edges[5].toFixed(0)} visitors / hour</span><span>· outlined = busiest hour</span>`;
    bindTips(grid);
    return { peakDay: h.days[peak[0]], peakHour: h.hours[peak[1]], peakValue: h.values[peak[0]][peak[1]] };
  }

  // ---- 3. Car demand vs service (scatter with median quadrants) -------------------------------
  function demandScatter(root, models) {
    const plot = root.querySelector(".plot");
    const W = Math.max(300, plot.clientWidth), H = Math.round(Math.min(380, Math.max(260, W * 0.62)));
    const m = { l: 44, r: 16, t: 14, b: 40 };
    const xMax = niceMax(Math.max(...models.map((d) => d.considered)) * 1.15, 10);
    const yMin = 0.4, yMax = 1;
    const X = (v) => m.l + (v / xMax) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - yMin) / (yMax - yMin)) * (H - m.t - m.b);
    const med = (k) => { const s = models.map((d) => d[k]).sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
    const mx = med("considered"), my = med("approached");
    const ink = css("--ink"), bad = css("--bad"), paper = css("--paper");

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img",
      "aria-label": `Scatter of ${models.length} car models: weekly considered visits against share approached by a salesperson.` });
    for (let v = 0; v <= xMax; v += 10) {
      svg.append(svgEl("line", { class: "grid", x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }));
      svg.append(svgEl("text", { class: "tick", x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, v));
    }
    for (let v = yMin; v <= yMax + 1e-9; v += 0.1) {
      svg.append(svgEl("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
      svg.append(svgEl("text", { class: "tick", x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, pct(v)));
    }
    // median split lines: solid, darker than the grid, so the quadrants read
    svg.append(svgEl("line", { class: "axis", x1: X(mx), x2: X(mx), y1: m.t, y2: H - m.b }));
    svg.append(svgEl("line", { class: "axis", x1: m.l, x2: W - m.r, y1: Y(my), y2: Y(my) }));
    const q = (x, y, anchor, text, color) => svg.append(svgEl("text", { x, y, "text-anchor": anchor, style: `fill:${color || css("--ink-3")};font-size:11.5px` }, text));
    q(W - m.r - 4, m.t + 14, "end", "Hot, covered");
    q(W - m.r - 4, H - m.b - 8, "end", "Hot but neglected", bad);
    q(m.l + 6, m.t + 14, "start", "Quiet, covered");
    q(m.l + 6, H - m.b - 8, "start", "Quiet, ignored");
    svg.append(svgEl("text", { class: "tick", x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, "Visitors who stayed 60 s+ at the car, per week →"));
    svg.append(svgEl("text", { class: "tick", x: 12, y: m.t + (H - m.t - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${m.t + (H - m.t - m.b) / 2})` }, "Approached by a salesperson →"));

    models.forEach((d) => {
      const neglected = d.considered > mx && d.approached < my;
      const x = X(d.considered), y = Y(d.approached);
      const g = svgEl("g", { tabindex: 0, "data-tip": `<b>${esc(d.label)}</b><br>${d.considered} considered visits / week<br>${pct(d.approached)} approached · ${d.unserved} left unserved${neglected ? "<br>Hot but neglected" : ""}` });
      g.append(svgEl("circle", { cx: x, cy: y, r: 14, fill: "transparent" })); // 28px hit target
      g.append(svgEl("circle", { cx: x, cy: y, r: 6, fill: neglected ? bad : ink, stroke: paper, "stroke-width": 2 }));
      const right = x < W - 120;
      g.append(svgEl("text", { x: right ? x + 10 : x - 10, y: y - 8, "text-anchor": right ? "start" : "end", style: `fill:${css("--ink")};font-weight:${neglected ? 500 : 400}` }, d.label));
      svg.append(g);
    });
    plot.replaceChildren(svg);
    bindTips(svg);
    root.querySelector("details .table-wrap").innerHTML = table(["Model", "Considered / week", "Approached", "Left unserved"],
      models.map((d) => [d.label, d.considered, pct(d.approached), d.unserved]));
    return { neglected: models.filter((d) => d.considered > mx && d.approached < my), medianX: mx, medianY: my };
  }

  // ---- 4. Campaign lift (highlight one line, the rest in grey context) ------------------------
  function campaign(root, c) {
    const plot = root.querySelector(".plot");
    const W = Math.max(300, plot.clientWidth), H = Math.round(Math.min(340, Math.max(240, W * 0.55)));
    const m = { l: 34, r: 112, t: 22, b: 30 };
    const n = c.weeks.length, s0 = c.startIndex, win = Math.min(s0, n - s0);
    const all = c.series.flatMap((s) => s.values);
    const yMax = niceMax(Math.max(...all) * 1.1, 10);
    const X = (i) => m.l + (i / (n - 1)) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - v / yMax) * (H - m.t - m.b);
    const ink3 = css("--ink-3"), data = css("--data"), context = "#aab0b4";
    const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const lift = (vals) => avg(vals.slice(s0, s0 + win)) / avg(vals.slice(s0 - win, s0)) - 1;
    const focus = c.series.find((s) => s.label === c.model);
    const others = c.series.filter((s) => s !== focus);
    const sum = (i) => others.reduce((t, s) => t + s.values[i], 0);
    const otherVals = c.weeks.map((_, i) => sum(i));
    const result = { lift: lift(focus.values), control: lift(otherVals), window: win, before: avg(focus.values.slice(s0 - win, s0)), after: avg(focus.values.slice(s0, s0 + win)) };

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img",
      "aria-label": `Weekly considered visits per model; ${c.model} highlighted, campaign from ${c.weeks[s0]}.` });
    // campaign band
    svg.append(svgEl("rect", { x: X(s0) - (X(1) - X(0)) / 2, y: m.t, width: X(n - 1) - X(s0) + (X(1) - X(0)) / 2, height: H - m.t - m.b, fill: data, opacity: 0.07 }));
    svg.append(svgEl("text", { x: X(s0) - (X(1) - X(0)) / 2 + 6, y: m.t - 8, style: `fill:${data};font-size:11.5px;font-weight:500` }, c.campaignLabel));
    for (let v = 0; v <= yMax; v += 10) {
      svg.append(svgEl("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
      svg.append(svgEl("text", { class: "tick", x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, v));
    }
    c.weeks.forEach((w, i) => { if (i % 2 === 0 || i === n - 1) svg.append(svgEl("text", { class: "tick", x: X(i), y: H - 10, "text-anchor": "middle" }, w)); });

    const path = (vals) => vals.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join(" ");
    others.forEach((s) => svg.append(svgEl("path", { d: path(s.values), fill: "none", stroke: context, "stroke-width": 1.5 })));
    svg.append(svgEl("path", { d: path(focus.values), fill: "none", stroke: data, "stroke-width": 2.5 }));
    svg.append(svgEl("circle", { cx: X(n - 1), cy: Y(focus.values[n - 1]), r: 4, fill: data, stroke: css("--paper"), "stroke-width": 2 }));

    // end labels, nudged apart so they never collide
    const labels = c.series.map((s) => ({ s, y: Y(s.values[n - 1]) })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 14) labels[i].y = labels[i - 1].y + 14;
    labels.forEach(({ s, y }) => svg.append(svgEl("text", { x: X(n - 1) + 8, y: y + 4,
      style: s === focus ? `fill:${css("--ink")};font-weight:500` : `fill:${ink3}` }, `${s.label} ${s.values[n - 1]}`)));

    // crosshair + tooltip
    const cross = svgEl("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, stroke: css("--ink"), "stroke-width": 1, visibility: "hidden" });
    svg.append(cross);
    const hit = svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
    svg.append(hit);
    hit.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect(), sx = ((e.clientX - r.left) / r.width) * W;
      const i = Math.max(0, Math.min(n - 1, Math.round(((sx - m.l) / (W - m.l - m.r)) * (n - 1))));
      cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
      const rows = [...c.series].sort((a, b) => b.values[i] - a.values[i])
        .map((s) => `<div class="tip-row${s === focus ? " on" : ""}"><span>${esc(s.label)}</span><span>${s.values[i]}</span></div>`).join("");
      show(`<b>Week of ${esc(c.weeks[i])}</b>${i >= s0 ? " · campaign running" : ""}${rows}`, e.clientX, e.clientY);
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hide(); });
    plot.replaceChildren(svg);

    root.querySelector(".lift").innerHTML = `<span class="big">${signed(result.lift)}</span>
      <p>${esc(c.model)} considered visits per week, ${win} weeks after the campaign started vs ${win} weeks before (${result.before.toFixed(1)} → ${result.after.toFixed(1)}). The other models moved ${signed(result.control)} over the same weeks.</p>`;
    root.querySelector("details .table-wrap").innerHTML = table(["Week", ...c.series.map((s) => s.label)],
      c.weeks.map((w, i) => [w, ...c.series.map((s) => s.values[i])]));
    return result;
  }

  window.Charts = { funnel, hours, demandScatter, campaign, signed, pct };
})();
