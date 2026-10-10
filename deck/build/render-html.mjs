#!/usr/bin/env node
/* Clio Teardown — JSON slide specs -> single print-ready HTML.
   Usage: node render-html.mjs <slidesDir> <outFile> */

import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dir = dirname(fileURLToPath(import.meta.url));
const [, , SLIDES_DIR = join(__dir, '..', 'slides'), OUT = join(__dir, 'deck.html')] = process.argv;

/* ---------------- helpers ---------------- */

const esc = (s) => String(s ?? '')
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// tiny inline markup: **bold**, `code`, [[tag:confirmed]] chips
const rich = (s) => esc(s)
  .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
  .replace(/`([^`]+?)`/g, '<code style="font-family:var(--mono);font-size:.88em">$1</code>');

const N = (n) => Number(n).toLocaleString('en-US');
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// one hue per deck part, walked around the wheel at even lightness
const ACCENTS = ['#C2492D','#BC7A22','#6F7F32','#3F7D4E','#2E7D74','#2F6E93','#4A56A0','#6E4C9B','#8E3F77','#B23A63','#55606E'];
const accentFor = (file) => ACCENTS[(parseInt(String(file).slice(0, 2), 10) || 0) % ACCENTS.length];
const PALETTE = ['var(--acc)', '#2E7D74', '#BC7A22', '#3F7D4E', '#2F6E93'];
const HEX = { 'var(--acc)': 'var(--acc)', '#2E7D74': '#2E7D74', '#BC7A22': '#BC7A22', '#3F7D4E': '#3F7D4E', '#2F6E93': '#2F6E93' };

const defs = `<defs>
<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0.5 L9,5 L0,9.5" fill="none" stroke="#8C8880" stroke-width="1.8"/></marker>
<marker id="ah-a" style="color:var(--acc)" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0.5 L9,5 L0,9.5" fill="none" stroke="currentColor" stroke-width="2.2"/></marker>
<marker id="ah-m" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0.5 L9,5 L0,9.5" fill="none" stroke="#7C7871" stroke-width="1.4"/></marker>
</defs>`;

/* ---------------- charts (deterministic, drawn from data) ---------------- */

function chartHBar(c) {
  const W = 1090, rowH = c.data.length > 8 ? 42 : 54, padL = c.labelWidth || 250;
  const H = c.data.length * rowH + 30;
  const max = Math.max(...c.data.map(d => Math.abs(d.value))) * 1.02 || 1;
  const barW = W - padL - 110;
  const rows = c.data.map((d, i) => {
    const y = i * rowH + 8, w = Math.max(2, Math.abs(d.value) / max * barW);
    const col = d.color ? `var(--${d.color})` : (d.highlight ? 'var(--acc-dk)' : 'var(--acc)');
    const lab = d.display ?? N(d.value);
    return `<text class="ax-lab" x="${padL - 14}" y="${y + rowH / 2 - 4}" text-anchor="end" dominant-baseline="middle" ${d.highlight ? 'style="fill:var(--acc-dk);font-weight:700"' : ''}>${esc(d.label)}</text>
<rect x="${padL}" y="${y + rowH / 2 - 17}" width="${w}" height="26" fill="${col}"/>
<text class="val-lab" x="${padL + w + 12}" y="${y + rowH / 2 - 3}" dominant-baseline="middle">${esc(lab)}</text>`;
  }).join('\n');
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${defs}${rows}
<line class="ax-base" x1="${padL}" y1="4" x2="${padL}" y2="${H - 18}"/></svg>`;
}

function chartBar(c) {
  const W = 1080, H = 430, padL = 78, padB = c.data.some(d => String(d.label).length > 9) ? 78 : 56, padT = 40;
  const max = (c.max ?? Math.max(...c.data.map(d => d.value))) * 1.12 || 1;
  const n = c.data.length, slot = (W - padL - 30) / n, bw = Math.min(slot * 0.56, 92);
  const y = v => H - padB - (v / max) * (H - padB - padT);
  const grid = (c.gridlines ?? 4);
  const gl = Array.from({ length: grid + 1 }, (_, i) => {
    const v = max / grid * i;
    return `<line class="ax-line" x1="${padL}" y1="${y(v)}" x2="${W - 20}" y2="${y(v)}"/>
<text class="ax-lab" x="${padL - 12}" y="${y(v)}" text-anchor="end" dominant-baseline="middle">${esc(c.fmt ? c.fmt.replace('{}', N(Math.round(v))) : N(Math.round(v)))}</text>`;
  }).join('\n');
  const bars = c.data.map((d, i) => {
    const cx = padL + slot * i + slot / 2, h = Math.max(2, (d.value / max) * (H - padB - padT));
    const col = d.color ? `var(--${d.color})` : (d.highlight ? 'var(--acc-dk)' : 'var(--acc)');
    const lab = String(d.label).length > 9 ? `<text class="ax-lab" transform="translate(${cx},${H - padB + 16}) rotate(-32)" text-anchor="end">${esc(d.label)}</text>`
      : `<text class="ax-lab" x="${cx}" y="${H - padB + 22}" text-anchor="middle">${esc(d.label)}</text>`;
    return `<rect x="${cx - bw / 2}" y="${y(d.value)}" width="${bw}" height="${h}" fill="${col}"/>
<text class="val-lab" x="${cx}" y="${y(d.value) - 11}" text-anchor="middle">${esc(d.display ?? N(d.value))}</text>${lab}`;
  }).join('\n');
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${defs}${gl}
<line class="ax-base" x1="${padL}" y1="${H - padB}" x2="${W - 20}" y2="${H - padB}"/>${bars}</svg>`;
}

function chartFunnel(c) {
  const W = 1060, H = 400, padT = 34, n = c.data.length;
  const rowH = (H - padT - 40) / n, maxV = c.data[0].value;
  const maxW = 660, xC = 330;
  const rows = c.data.map((d, i) => {
    const w = Math.max(40, d.value / maxV * maxW), y = padT + i * rowH;
    const col = HEX[PALETTE[Math.min(i, 4)]];
    const pct = Math.round(d.value / maxV * 100);
    return `<rect x="${xC - w / 2}" y="${y}" width="${w}" height="${rowH - 13}" fill="${col}" opacity="${1 - i * 0.13}"/>
<text class="bar-lab" x="${xC}" y="${y + (rowH - 13) / 2}" text-anchor="middle" dominant-baseline="middle" style="font-size:17px">${esc(d.display ?? d.value)}</text>
<text class="dg-t-b" x="${xC + maxW / 2 + 34}" y="${y + (rowH - 13) / 2 - 8}" dominant-baseline="middle">${esc(d.label)}</text>
<text class="dg-t-s" x="${xC + maxW / 2 + 34}" y="${y + (rowH - 13) / 2 + 13}" dominant-baseline="middle">${esc(d.note ?? pct + '% of start')}</text>`;
  }).join('\n');
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${defs}${rows}</svg>`;
}

function chartLine(c) {
  const W = 1060, H = 400, padL = 92, padB = 58, padT = 36;
  const pts = c.data;
  const max = (c.max ?? Math.max(...pts.map(d => d.value))) * 1.14 || 1;
  const min = c.min ?? 0;
  const x = i => padL + (W - padL - 40) * (pts.length === 1 ? .5 : i / (pts.length - 1));
  const y = v => H - padB - ((v - min) / (max - min)) * (H - padB - padT);
  const grid = Array.from({ length: 5 }, (_, i) => {
    const v = min + (max - min) / 4 * i;
    return `<line class="ax-line" x1="${padL}" y1="${y(v)}" x2="${W - 30}" y2="${y(v)}"/>
<text class="ax-lab" x="${padL - 12}" y="${y(v)}" text-anchor="end" dominant-baseline="middle">${esc(c.fmt ? c.fmt.replace('{}', N(Math.round(v))) : N(Math.round(v)))}</text>`;
  }).join('\n');
  const path = pts.map((d, i) => `${i ? 'L' : 'M'}${x(i)},${y(d.value)}`).join(' ');
  const dots = pts.map((d, i) => `<circle cx="${x(i)}" cy="${y(d.value)}" r="6" fill="var(--acc)"/>
<text class="val-lab" x="${x(i)}" y="${y(d.value) - 16}" text-anchor="middle">${esc(d.display ?? N(d.value))}</text>
<text class="ax-lab" x="${x(i)}" y="${H - padB + 24}" text-anchor="middle">${esc(d.label)}</text>`).join('\n');
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${defs}${grid}
<line class="ax-base" x1="${padL}" y1="${H - padB}" x2="${W - 30}" y2="${H - padB}"/>
<path d="${path}" fill="none" stroke="var(--acc)" stroke-width="2.8"/>${dots}</svg>`;
}

function chartStack(c) {
  // data: [{label, parts:[{name,value}]}], series legend from first row
  const W = 1060, H = 400, padL = 80, padB = 76, padT = 30;
  const totals = c.data.map(d => d.parts.reduce((a, p) => a + p.value, 0));
  const max = Math.max(...totals) * 1.1;
  const n = c.data.length, slot = (W - padL - 30) / n, bw = Math.min(slot * .5, 88);
  const y = v => H - padB - (v / max) * (H - padB - padT);
  const bars = c.data.map((d, i) => {
    const cx = padL + slot * i + slot / 2;
    let acc = 0;
    return d.parts.map((p, j) => {
      const y0 = y(acc + p.value), h = y(acc) - y0; acc += p.value;
      return `<rect x="${cx - bw / 2}" y="${y0}" width="${bw}" height="${Math.max(1, h)}" fill="${HEX[PALETTE[j % 5]]}"/>`;
    }).join('') + `<text class="ax-lab" x="${cx}" y="${H - padB + 22}" text-anchor="middle">${esc(d.label)}</text>`;
  }).join('\n');
  const leg = (c.data[0]?.parts || []).map((p, j) =>
    `<rect x="${padL + j * 190}" y="${H - 30}" width="13" height="13" fill="${HEX[PALETTE[j % 5]]}"/>
<text class="ax-lab" x="${padL + j * 190 + 20}" y="${H - 19}">${esc(p.name)}</text>`).join('\n');
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${defs}
<line class="ax-base" x1="${padL}" y1="${H - padB}" x2="${W - 30}" y2="${H - padB}"/>${bars}${leg}</svg>`;
}

function chartTimeline(c) {
  const W = 1090, rowGap = 62, H = c.data.length * rowGap + 46;
  const axisX = 250;
  const rows = c.data.map((d, i) => {
    const y = 26 + i * rowGap;
    const col = d.highlight ? 'var(--acc-dk)' : 'var(--acc)';
    return `<text class="dg-t-b" x="${axisX - 26}" y="${y}" text-anchor="end" dominant-baseline="middle" ${d.highlight ? 'style="fill:var(--acc-dk)"' : ''}>${esc(d.date)}</text>
<circle cx="${axisX}" cy="${y}" r="${d.highlight ? 8 : 5.5}" fill="${col}"/>
<text class="dg-t" x="${axisX + 26}" y="${y - 9}" dominant-baseline="middle" style="font-weight:600">${esc(d.label)}</text>
<text class="dg-t-s" x="${axisX + 26}" y="${y + 12}" dominant-baseline="middle">${esc(d.note ?? '')}</text>`;
  }).join('\n');
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${defs}
<line x1="${axisX}" y1="16" x2="${axisX}" y2="${H - 22}" class="dg-rule"/>${rows}</svg>`;
}

const CHARTS = { hbar: chartHBar, bar: chartBar, funnel: chartFunnel, line: chartLine, stack: chartStack, timeline: chartTimeline };

/* ---------------- layouts ---------------- */

function furniture(s, no) {
  const src = s.source ? `<div class="src">${rich(s.source)}</div>` : '<div class="src"></div>';
  return `<div class="foot">${src}<div class="no">${no}</div></div>`;
}

const head = (s) => {
  const t = s.title ? `<h1 class="slide-title${String(s.title).length > 52 ? ' long' : ''}">${rich(s.title)}</h1>` : '';
  const e = s.eyebrow ? `<div class="eyebrow">${rich(s.eyebrow)}</div>` : '';
  const sub = s.subtitle ? `<div class="subtitle">${rich(s.subtitle)}</div>` : '';
  return e + t + sub;
};

const LAYOUTS = {
  title: (s) => `<div class="rule"></div>${s.eyebrow ? `<div class="eyebrow">${rich(s.eyebrow)}</div>` : ''}
<h1>${rich(s.title)}</h1>
${s.subtitle ? `<div class="subtitle">${rich(s.subtitle)}</div>` : ''}
${s.meta ? `<div class="meta">${s.meta.map(m => rich(m)).join('<br>')}</div>` : ''}`,

  section: (s) => `${s.num ? `<div class="num">${rich(s.num)}</div>` : ''}
<h1>${rich(s.title)}</h1>
${s.subtitle ? `<div class="subtitle">${rich(s.subtitle)}</div>` : ''}`,

  statement: (s) => {
    const len = String(s.big ?? '').length;
    const cls = len > 16 ? ' xs' : len > 9 ? ' sm' : '';
    return `<div class="statement">
${s.eyebrow ? `<div class="eyebrow">${rich(s.eyebrow)}</div>` : ''}
${s.big ? `<div class="big${cls}">${rich(s.big)}</div>` : ''}
${s.claim ? `<div class="claim">${rich(s.claim)}</div>` : ''}
${s.note ? `<div class="note">${rich(s.note)}</div>` : ''}</div>`;
  },

  bullets: (s) => `${head(s)}<div class="body">
<ul class="bullets${s.items && s.items.length > 5 ? ' tight' : ''}">${(s.items || []).map(i => `<li>${rich(i)}</li>`).join('')}</ul></div>`,

  steps: (s) => `${head(s)}<div class="body">
<ol class="steps">${(s.items || []).map(i => `<li>${rich(i)}</li>`).join('')}</ol></div>`,

  twoCol: (s) => `${head(s)}<div class="body"><div class="two-col">
<div class="${s.leftTone === 'pos' ? 'pos-card' : s.leftTone === 'neg' ? 'neg-card' : ''}"><div class="col-head${s.leftTone ? ' ' + s.leftTone : ''}">${rich(s.leftHead || '')}</div>
<ul class="bullets">${(s.left || []).map(i => `<li>${rich(i)}</li>`).join('')}</ul></div>
<div class="${s.rightTone === 'pos' ? 'pos-card' : s.rightTone === 'neg' ? 'neg-card' : ''}"><div class="col-head${s.rightTone ? ' ' + s.rightTone : ''}">${rich(s.rightHead || '')}</div>
<ul class="bullets">${(s.right || []).map(i => `<li>${rich(i)}</li>`).join('')}</ul></div>
</div></div>`,

  table: (s) => {
    const rows = s.rows || [];
    const dense = rows.length > 8 ? ' xs' : rows.length > 5 ? ' sm' : '';
    return `${head(s)}<div class="body">
<table class="grid${dense}">
${s.head ? `<thead><tr>${s.head.map(h => `<th>${rich(h)}</th>`).join('')}</tr></thead>` : ''}
<tbody>${rows.map(r => `<tr>${r.map(c => {
      const cell = typeof c === 'object' && c !== null ? c : { t: c };
      return `<td class="${cell.tone || ''}${cell.num ? ' num' : ''}">${rich(cell.t)}</td>`;
    }).join('')}</tr>`).join('')}</tbody></table>
${s.note ? `<div class="fig-note">${rich(s.note)}</div>` : ''}</div>`;
  },

  quote: (s) => `${s.eyebrow ? `<div class="eyebrow">${rich(s.eyebrow)}</div>` : ''}
<blockquote class="q${String(s.quote || '').length > 150 ? ' sm' : ''}">&ldquo;${rich(s.quote)}&rdquo;</blockquote>
<div class="attrib">${rich(s.attrib || '')}</div>`,

  callout: (s) => `${head(s)}<div class="body"><div class="callout">
<div class="lead">${rich(s.lead)}</div>
${s.detail ? `<div class="detail">${rich(s.detail)}</div>` : ''}</div></div>`,

  kpis: (s) => `${head(s)}<div class="body"><div class="kpis">
${(s.items || []).map(k => `<div><div class="kpi-val${String(k.value).length > 6 ? ' sm' : ''}">${rich(k.value)}</div><div class="kpi-lab">${rich(k.label)}</div></div>`).join('')}
</div>${s.note ? `<div class="fig-note" style="margin-top:34px">${rich(s.note)}</div>` : ''}</div>`,

  chart: (s) => {
    const fn = CHARTS[s.chart?.type];
    const svg = fn ? fn(s.chart) : `<text>unknown chart ${esc(s.chart?.type)}</text>`;
    return `${head(s)}<div class="body"><div class="figure">${svg}</div>
${s.note ? `<div class="fig-note">${rich(s.note)}</div>` : ''}</div>`;
  },

  explain: (s) => `${head(s)}<div class="body"><div class="explain">
<div><div class="col-head">${rich(s.mainHead || 'What it does')}</div>
<p class="ex-p">${rich(s.story || '')}</p>
${s.why ? `<div class="col-head ex-gap">${rich(s.whyHead || 'Why it matters')}</div><p class="ex-p">${rich(s.why)}</p>` : ''}</div>
<div><div class="col-head">${rich(s.listHead || 'Use cases')}</div>
<ul class="bullets">${(s.uses || []).map(i => `<li>${rich(i)}</li>`).join('')}</ul>
<div class="col-head ex-gap">${rich(s.buyersHead || 'Who buys it')}</div>
<p class="ex-buyers">${rich(s.buyers || '')}</p></div>
</div></div>`,

  diagram: (s) => `${head(s)}<div class="body"><div class="figure">
<svg viewBox="${s.viewBox || '0 0 1100 430'}" width="100%">${defs}${s.svg || ''}</svg></div>
${s.note ? `<div class="fig-note">${rich(s.note)}</div>` : ''}</div>`,
};

/* ---------------- assemble ---------------- */

const files = readdirSync(SLIDES_DIR).filter(f => f.endsWith('.json')).sort();
let all = [];
for (const f of files) {
  try {
    const j = JSON.parse(readFileSync(join(SLIDES_DIR, f), 'utf8'));
    const arr = Array.isArray(j) ? j : (j.slides || []);
    arr.forEach(s => all.push({ ...s, _file: f }));
  } catch (e) { console.error(`!! ${f}: ${e.message}`); }
}

const html = [];
all.forEach((s, i) => {
  const no = i + 1;
  const L = LAYOUTS[s.layout];
  if (!L) { console.error(`!! slide ${no} (${s._file}): unknown layout "${s.layout}"`); return; }
  const extra = [s.layout === 'title' ? 'title-slide' : '', s.layout === 'section' ? 'section-slide' : '',
  s.layout === 'statement' ? 'statement-slide' : '', s.layout === 'quote' ? 'quote-slide' : '',
  s.theme || ''].filter(Boolean).join(' ');
  const foot = (s.layout === 'title') ? '' : furniture(s, no);
  html.push(`<section class="slide ${extra}" style="--acc:${accentFor(s._file)}" data-n="${no}" data-src="${esc(s._file)}">\n${L(s)}\n${foot}\n</section>`);
});

const css = readFileSync(join(__dir, 'design.css'), 'utf8');
writeFileSync(OUT, `<!doctype html><html><head><meta charset="utf-8">
<title>Clio Teardown</title><style>${css}</style></head><body>
${html.join('\n')}
</body></html>`);

console.log(`${all.length} slides from ${files.length} files -> ${OUT}`);
