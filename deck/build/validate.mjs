#!/usr/bin/env node
/* Validate slide spec files against the contract. Usage: node validate.mjs <file.json|dir> */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const TARGET = process.argv[2];

// Composition thresholds. Override per deck with env vars when a deck genuinely
// warrants it — but raise them deliberately, not to silence a failing build.
const BULLET_CAP   = Number(process.env.BULLET_CAP   ?? 0.30);
const FIGURE_FLOOR = Number(process.env.FIGURE_FLOOR ?? 0.10);
const CAPS = {
  title: 60, subtitle: 150, note: 170, claim: 90, big: 14, lead: 95, detail: 300,
  quote: 240, attrib: 130, eyebrow: 46, source: 190,
};
const LAYOUTS = ['title', 'section', 'statement', 'bullets', 'steps', 'twoCol', 'table', 'quote', 'callout', 'kpis', 'chart', 'diagram', 'explain'];
const CHARTS = ['hbar', 'bar', 'funnel', 'line', 'stack', 'timeline'];
const DG_CLASSES = ['dg-box', 'dg-box-a', 'dg-box-t', 'dg-box-m', 'dg-box-d', 'dg-t', 'dg-t-b', 'dg-t-s', 'dg-t-w', 'dg-t-a', 'dg-arrow', 'dg-arrow-a', 'dg-dash', 'dg-rule'];

let errs = [], warns = [], count = 0;

// Composition tracking. In the run this skill came from, the first draft of a
// 267-slide deck came back 41% bullet lists because character caps were ENFORCED
// while layout mix was merely suggested. What you validate is what you get.
const stats = { total: 0, byLayout: {}, byFile: {}, noSource: 0 };
const FIGURE = new Set(['chart', 'diagram']);

function checkFile(path) {
  let arr;
  const raw = readFileSync(path, 'utf8');
  try { arr = JSON.parse(raw); }
  catch (e) { errs.push(`${path}: INVALID JSON — ${e.message}`); return; }
  if (!Array.isArray(arr)) { errs.push(`${path}: top level must be an ARRAY of slides`); return; }

  const short = path.split('/').pop();
  stats.byFile[short] ??= { n: 0, figures: 0, bullets: 0, prose: 0 };

  arr.forEach((s, i) => {
    count++;
    const at = `${path} #${i + 1}`;
    stats.total++;
    stats.byLayout[s.layout] = (stats.byLayout[s.layout] || 0) + 1;
    stats.byFile[short].n++;
    if (FIGURE.has(s.layout)) stats.byFile[short].figures++;
    if (s.layout === 'bullets') stats.byFile[short].bullets++;
    // "prose" = a layout that is text a reader must wade through. A file made only of
    // covers, big numbers and KPI cards is already visual and needs no figure.
    if (['bullets', 'steps', 'table'].includes(s.layout)) stats.byFile[short].prose++;
    if (!s.layout) return errs.push(`${at}: missing "layout"`);
    if (!LAYOUTS.includes(s.layout)) return errs.push(`${at}: unknown layout "${s.layout}" (use ${LAYOUTS.join('/')})`);

    for (const [k, cap] of Object.entries(CAPS)) {
      if (typeof s[k] === 'string' && s[k].length > cap)
        errs.push(`${at}: "${k}" is ${s[k].length} chars, cap ${cap} — "${s[k].slice(0, 50)}..."`);
    }
    // a quote carries its citation in `attrib`, so it is not missing a source
    const cited = s.source || (s.layout === 'quote' && s.attrib);
    if (s.layout !== 'title' && s.layout !== 'section' && !cited) {
      stats.noSource++;
      warns.push(`${at} (${s.layout}): no "source"`);
    }

    const listCap = (key, max, len) => {
      const L = s[key];
      if (!L) return;
      if (!Array.isArray(L)) return errs.push(`${at}: "${key}" must be an array`);
      if (L.length > max) errs.push(`${at}: "${key}" has ${L.length} items, max ${max}`);
      L.forEach((it, j) => {
        if (typeof it === 'string' && it.length > len)
          errs.push(`${at}: ${key}[${j}] is ${it.length} chars, cap ${len} — "${it.slice(0, 46)}..."`);
      });
    };

    switch (s.layout) {
      case 'bullets': if (!s.items?.length) errs.push(`${at}: bullets needs "items"`); listCap('items', 6, 95); break;
      case 'steps': if (!s.items?.length) errs.push(`${at}: steps needs "items"`); listCap('items', 7, 110); break;
      case 'twoCol': listCap('left', 6, 62); listCap('right', 6, 62);
        if (!s.leftHead || !s.rightHead) errs.push(`${at}: twoCol needs leftHead and rightHead`); break;
      case 'table': {
        if (!s.rows?.length) { errs.push(`${at}: table needs "rows"`); break; }
        if (s.rows.length > 9) errs.push(`${at}: table has ${s.rows.length} rows, max 9`);
        const w = s.head?.length || s.rows[0].length;
        if (w > 5) errs.push(`${at}: table has ${w} columns, max 5`);
        s.rows.forEach((r, j) => {
          if (r.length !== w) errs.push(`${at}: row ${j + 1} has ${r.length} cells, header has ${w}`);
          r.forEach((c, k) => {
            const t = typeof c === 'object' && c !== null ? c.t : c;
            if (typeof t === 'string' && t.length > 42)
              errs.push(`${at}: cell [${j + 1},${k + 1}] is ${t.length} chars, cap 42 — "${String(t).slice(0, 40)}"`);
          });
        });
        break;
      }
      case 'kpis': {
        if (!s.items?.length) { errs.push(`${at}: kpis needs "items"`); break; }
        if (s.items.length < 2 || s.items.length > 4) errs.push(`${at}: kpis needs 2-4 items, has ${s.items.length}`);
        s.items.forEach((k, j) => {
          if (String(k.value ?? '').length > 8) errs.push(`${at}: kpi[${j}].value "${k.value}" over 8 chars`);
          if (String(k.label ?? '').length > 60) errs.push(`${at}: kpi[${j}].label over 60 chars`);
        });
        break;
      }
      case 'statement': if (!s.big && !s.claim) errs.push(`${at}: statement needs "big" or "claim"`); break;
      case 'quote': if (!s.quote) errs.push(`${at}: quote needs "quote"`); break;
      case 'callout': if (!s.lead) errs.push(`${at}: callout needs "lead"`); break;
      case 'chart': {
        const c = s.chart;
        if (!c) { errs.push(`${at}: chart needs "chart"`); break; }
        if (!CHARTS.includes(c.type)) { errs.push(`${at}: chart.type "${c.type}" invalid (${CHARTS.join('/')})`); break; }
        if (!Array.isArray(c.data) || !c.data.length) { errs.push(`${at}: chart.data must be a non-empty array`); break; }
        if (c.data.length > 12) errs.push(`${at}: chart has ${c.data.length} points, max 12`);
        c.data.forEach((d, j) => {
          if (c.type === 'timeline') { if (!d.date || !d.label) errs.push(`${at}: timeline[${j}] needs date+label`); }
          else if (c.type === 'stack') { if (!Array.isArray(d.parts)) errs.push(`${at}: stack[${j}] needs parts[]`); }
          else if (typeof d.value !== 'number' || Number.isNaN(d.value)) errs.push(`${at}: chart data[${j}].value must be a number`);
          if (c.type !== 'timeline' && c.type !== 'stack' && !d.label) errs.push(`${at}: chart data[${j}] needs a label`);
        });
        break;
      }
      case 'explain': {
        if (!s.story) { errs.push(`${at}: explain needs "story"`); break; }
        if (s.story.length > 520) errs.push(`${at}: story is ${s.story.length} chars, cap 520`);
        if (typeof s.why === 'string' && s.why.length > 260) errs.push(`${at}: why is ${s.why.length} chars, cap 260`);
        listCap('uses', 4, 120);
        if (typeof s.buyers === 'string' && s.buyers.length > 200) errs.push(`${at}: buyers is ${s.buyers.length} chars, cap 200`);
        const words = [s.subtitle, s.story, s.why, ...(s.uses || []), s.buyers].filter(Boolean).join(' ').split(/\s+/).length;
        if (words > 150) errs.push(`${at}: ${words} words on the page, cap 150`);
        break;
      }
      case 'diagram': {
        if (!s.svg) { errs.push(`${at}: diagram needs "svg"`); break; }
        if (/\b(fill|stroke)\s*=\s*['"]#|\b(fill|stroke)\s*=\s*['"](red|blue|green|black|white|orange|purple|gray|grey)/i.test(s.svg))
          errs.push(`${at}: diagram uses literal colours — use only the dg-* classes`);
        const used = [...s.svg.matchAll(/class=['"]([^'"]+)['"]/g)].flatMap(m => m[1].split(/\s+/));
        used.filter(c => c && !DG_CLASSES.includes(c)).forEach(c => errs.push(`${at}: unknown diagram class "${c}"`));
        const nums = [...s.svg.matchAll(/\b(?:x|cx)=['"](-?[\d.]+)/g)].map(m => +m[1]);
        const ys = [...s.svg.matchAll(/\b(?:y|cy)=['"](-?[\d.]+)/g)].map(m => +m[1]);
        if (nums.some(n => n < 0 || n > 1095)) errs.push(`${at}: diagram has x outside 0..1095 (canvas is 1100 wide)`);
        if (ys.some(n => n < 0 || n > 425)) errs.push(`${at}: diagram has y outside 0..425 (canvas is 430 tall)`);
        if (/dg-t-w/.test(s.svg) && !/dg-box-d/.test(s.svg))
          errs.push(`${at}: uses dg-t-w (white text) with no dg-box-d (dark box) — white on a light fill is unreadable`);
        if (!/<text/.test(s.svg)) warns.push(`${at}: diagram has no text labels`);
        break;
      }
    }
  });
}

const st = statSync(TARGET);
const DIR_MODE = st.isDirectory();
if (DIR_MODE) readdirSync(TARGET).filter(f => f.endsWith('.json')).sort().forEach(f => checkFile(join(TARGET, f)));
else checkFile(TARGET);

// ---------------- composition gate ----------------
// Per-file: any part of real length needs something drawn in it.
for (const [f, s] of Object.entries(stats.byFile)) {
  if (s.n >= 4 && s.figures === 0 && s.prose >= 2)
    errs.push(`${f}: ${s.n} slides, ${s.prose} of them prose, and not one chart or diagram — every part of real length needs something drawn`);
}

// Deck-wide: only meaningful across the whole deck, so directory targets only.
if (DIR_MODE && stats.total >= 12) {
  const b = stats.byLayout.bullets || 0;
  const share = b / stats.total;
  if (share > BULLET_CAP)
    errs.push(`composition: ${b}/${stats.total} slides are bullet lists (${Math.round(share * 100)}%), cap is ${Math.round(BULLET_CAP * 100)}% — convert the weakest runs to tables, diagrams or comparisons`);

  const figures = (stats.byLayout.chart || 0) + (stats.byLayout.diagram || 0);
  const figShare = figures / stats.total;
  if (figShare < FIGURE_FLOOR)
    errs.push(`composition: only ${figures}/${stats.total} slides (${Math.round(figShare * 100)}%) are charts or diagrams, floor is ${Math.round(FIGURE_FLOOR * 100)}%`);

  const uncited = stats.noSource / stats.total;
  if (uncited > 0.15)
    warns.push(`composition: ${stats.noSource}/${stats.total} slides carry no source (${Math.round(uncited * 100)}%)`);

  console.log('\nlayout mix:');
  Object.entries(stats.byLayout).sort((a, b2) => b2[1] - a[1])
    .forEach(([k, v]) => console.log(`  ${k.padEnd(10)} ${String(v).padStart(3)}  ${String(Math.round(v / stats.total * 100)).padStart(3)}%`));
  console.log('');
}

warns.forEach(w => console.log(`WARN  ${w}`));
errs.forEach(e => console.log(`ERROR ${e}`));
console.log(`\n${count} slides checked — ${errs.length} errors, ${warns.length} warnings`);
process.exit(errs.length ? 1 : 0);
