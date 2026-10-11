// Draws the showroom floor: the dark live stage (with lenses) and the light heatmap on the Visitors page.
(function () {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.round(s) % 60).padStart(2, "0")}`;

  function fit(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function projector(floor, w, h, pad) {
    const s = Math.min((w - pad * 2) / floor.widthM, (h - pad * 2) / floor.depthM);
    const ox = (w - floor.widthM * s) / 2, oy = (h - floor.depthM * s) / 2;
    return { s, X: (x) => ox + x * s, Y: (y) => oy + y * s, ox, oy };
  }

  function heatColor(v) {
    // pale amber -> orange -> brick red, matching the legend bands
    const stops = [[243, 201, 139], [229, 138, 58], [184, 50, 31]];
    const t = Math.min(1, Math.max(0, v)) * 2, i = Math.min(1, Math.floor(t)), f = t - i;
    const a = stops[i], b = stops[i + 1];
    return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * f));
  }

  function drawHeatLayer(ctx, P, floor, heat, alpha) {
    if (!heat || !heat.values || !heat.values.length) return;
    const max = Math.max(...heat.values) || 1;
    const cw = (floor.widthM / heat.cols) * P.s, ch = (floor.depthM / heat.rows) * P.s;
    const off = document.createElement("canvas");
    off.width = heat.cols; off.height = heat.rows;
    const octx = off.getContext("2d"), img = octx.createImageData(heat.cols, heat.rows);
    heat.values.forEach((raw, i) => {
      const v = Math.sqrt(raw / max);
      const [r, g, b] = heatColor(v);
      img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = v < 0.08 ? 0 : Math.round(255 * alpha * Math.min(1, v * 1.4));
    });
    octx.putImageData(img, 0, 0);
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.filter = `blur(${Math.max(2, cw * 0.6)}px)`;
    ctx.drawImage(off, P.ox, P.oy, cw * heat.cols, ch * heat.rows);
    ctx.restore();
  }

  function drawPlan(ctx, P, floor, theme, labels) {
    const { line, faint, text } = theme;
    // 1 m grid
    ctx.strokeStyle = faint; ctx.lineWidth = 1;
    ctx.beginPath();
    const m = floor.unitsPerMetre || 1;
    for (let x = 0; x <= floor.widthM; x += m) { ctx.moveTo(P.X(x) + 0.5, P.Y(0)); ctx.lineTo(P.X(x) + 0.5, P.Y(floor.depthM)); }
    for (let y = 0; y <= floor.depthM; y += m) { ctx.moveTo(P.X(0), P.Y(y) + 0.5); ctx.lineTo(P.X(floor.widthM), P.Y(y) + 0.5); }
    ctx.stroke();
    // walls with the entrance gap
    const e = floor.entrance;
    ctx.strokeStyle = line; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(P.X(e.x), P.Y(floor.depthM));
    ctx.lineTo(P.X(0), P.Y(floor.depthM)); ctx.lineTo(P.X(0), P.Y(0));
    ctx.lineTo(P.X(floor.widthM), P.Y(0)); ctx.lineTo(P.X(floor.widthM), P.Y(floor.depthM));
    ctx.lineTo(P.X(e.x + e.w), P.Y(floor.depthM));
    ctx.stroke();
    ctx.fillStyle = text; ctx.font = `500 11px ${css("--f-mono")}`; ctx.textAlign = "center";
    ctx.fillText("Entrance", P.X(e.x + e.w / 2), P.Y(floor.depthM) - 6);

    // cars: a body outline with a cabin, so the zones read as cars rather than boxes
    for (const z of floor.zones) {
      const x = P.X(z.x), y = P.Y(z.y), w = z.w * P.s, h = z.h * P.s;
      ctx.strokeStyle = line; ctx.lineWidth = 1.25;
      roundRect(ctx, x, y, w, h, h * 0.42); ctx.stroke();
      ctx.strokeStyle = faint; roundRect(ctx, x + w * 0.3, y + h * 0.16, w * 0.42, h * 0.68, h * 0.2); ctx.stroke();
      if (labels) {
        ctx.fillStyle = text; ctx.textAlign = "left"; ctx.font = `500 11px ${css("--f-mono")}`;
        ctx.fillText(labels(z), x, y + h + 15);
      }
    }
    for (const f of floor.fixtures || []) {
      ctx.strokeStyle = faint; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      ctx.strokeRect(P.X(f.x), P.Y(f.y), f.w * P.s, f.h * P.s); ctx.setLineDash([]);
      ctx.fillStyle = text; ctx.textAlign = "left"; ctx.font = `400 11px ${css("--f-mono")}`;
      ctx.fillText(f.label, P.X(f.x) + 4, P.Y(f.y) - 5);
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

  function chip(ctx, x, y, text, bg, fg) {
    ctx.font = `500 11px ${css("--f-mono")}`;
    const w = ctx.measureText(text).width + 10;
    ctx.fillStyle = bg; ctx.fillRect(x - w / 2, y - 9, w, 17);
    ctx.fillStyle = fg; ctx.textAlign = "center"; ctx.fillText(text, x, y + 3.5);
  }

  // Live stage. lens: "all" | "coach" | "pulse"
  function drawStage(canvas, floor, live, lens) {
    const { ctx, w, h } = fit(canvas);
    const ink = css("--stage-ink"), dim = css("--stage-dim"), accent = css("--accent"), guest = css("--guest");
    ctx.fillStyle = css("--stage"); ctx.fillRect(0, 0, w, h);
    if (!floor) return;
    const P = projector(floor, w, h, 22);
    const people = live ? live.people : [];

    if (lens === "pulse" && live && live.heat) drawHeatLayer(ctx, P, floor, live.heat, 0.55);

    const dwellers = {};
    people.forEach((p) => { if (p.role === "customer" && p.zoneDwellSec != null) (dwellers[p.zone] ||= []).push(p.zoneDwellSec); });
    drawPlan(ctx, P, floor, { line: "rgba(230,232,228,0.42)", faint: "rgba(230,232,228,0.07)", text: "rgba(230,232,228,0.55)" },
      (z) => {
        if (lens !== "pulse") return z.label;
        const d = dwellers[z.id];
        return d ? `${z.label} · ${d.length} here · ${mmss(Math.max(...d))}` : z.label;
      });

    const byId = Object.fromEntries(people.map((p) => [p.id, p]));
    const M = floor.unitsPerMetre || 1;
    const R = Math.max(4, P.s * M * 0.22);

    // trails
    for (const p of people) {
      if (!p.trail || p.trail.length < 2) continue;
      const show = lens === "pulse" ? p.role === "customer" : lens === "coach" ? p.role === "staff" : true;
      if (!show) continue;
      ctx.strokeStyle = p.role === "staff" ? "rgba(224,122,31,0.35)" : lens === "pulse" ? "rgba(185,203,216,0.5)" : "rgba(185,203,216,0.22)";
      ctx.lineWidth = lens === "pulse" ? 1.6 : 1.1;
      ctx.beginPath();
      p.trail.forEach(([x, y], i) => (i ? ctx.lineTo(P.X(x), P.Y(y)) : ctx.moveTo(P.X(x), P.Y(y))));
      ctx.lineTo(P.X(p.x), P.Y(p.y));
      ctx.stroke();
    }

    // engagement rings: 2 m radius around the pair, with progress toward the 2-minute rule
    if (lens !== "pulse") {
      for (const s of people) {
        if (s.role !== "staff" || !s.engagedWith || !byId[s.engagedWith]) continue;
        const g = byId[s.engagedWith];
        const cx = P.X((s.x + g.x) / 2), cy = P.Y((s.y + g.y) / 2), rr = 2 * M * P.s;
        const done = s.engagedSec >= 120;
        ctx.strokeStyle = done ? "rgba(230,232,228,0.75)" : "rgba(230,232,228,0.35)";
        ctx.lineWidth = 1; ctx.setLineDash(done ? [] : [4, 4]);
        ctx.beginPath(); ctx.arc(cx, cy, rr, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
        ctx.strokeStyle = accent; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(cx, cy, rr, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, s.engagedSec / 120)); ctx.stroke();
        if (lens === "coach") chip(ctx, cx, cy - rr - 4, done ? `engaged ${mmss(s.engagedSec)}` : `${mmss(s.engagedSec)} / 2:00`, done ? ink : "rgba(230,232,228,0.16)", done ? css("--stage") : ink);
      }
    }

    // people
    for (const p of people) {
      const x = P.X(p.x), y = P.Y(p.y);
      if (p.role === "staff") {
        const faded = lens === "pulse";
        ctx.fillStyle = faded ? "rgba(224,122,31,0.35)" : accent;
        if (faded) ctx.fillRect(x - R * 0.7, y - R * 0.7, R * 1.4, R * 1.4);
        else { ctx.beginPath(); ctx.arc(x, y, R + 1, 0, Math.PI * 2); ctx.fill(); }
        if (!faded) {
          ctx.fillStyle = ink; ctx.font = `500 11px ${css("--f-body")}`; ctx.textAlign = "left";
          ctx.fillText(p.name || p.id, x + R + 5, y + 4);
        }
      } else {
        const waiting = p.waitingSec != null;
        const late = waiting && p.waitingSec > 60;
        const dimmed = lens === "coach" && !waiting;
        ctx.fillStyle = dimmed ? "rgba(185,203,216,0.3)" : guest;
        ctx.beginPath(); ctx.arc(x, y, R, 0, Math.PI * 2); ctx.fill();
        if (late && lens !== "pulse") { ctx.strokeStyle = css("--bad"); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, R + 4, 0, Math.PI * 2); ctx.stroke(); }
        if (waiting && lens === "coach") chip(ctx, x, y - R - 12, `waiting ${mmss(p.waitingSec)}`, late ? css("--bad") : "rgba(230,232,228,0.16)", ink);
        if (lens === "pulse" && p.zoneDwellSec >= 20) {
          ctx.fillStyle = dim; ctx.font = `500 10.5px ${css("--f-mono")}`; ctx.textAlign = "center";
          ctx.fillText(mmss(p.zoneDwellSec), x, y - R - 6);
        }
      }
    }
  }

  // Visitors page heatmap on the light paper surface.
  function drawHeat(canvas, floor, heat) {
    const { ctx, w, h } = fit(canvas);
    ctx.fillStyle = css("--paper"); ctx.fillRect(0, 0, w, h);
    if (!floor) return;
    const P = projector(floor, w, h, 22);
    drawHeatLayer(ctx, P, floor, heat, 0.9);
    drawPlan(ctx, P, floor, { line: "rgba(18,20,22,0.7)", faint: "rgba(18,20,22,0.08)", text: "rgba(18,20,22,0.7)" }, (z) => z.label);
  }

  window.Floor = { drawStage, drawHeat };
})();
