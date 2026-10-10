#!/usr/bin/env node
/* Compute slide indices. NEVER type these by hand.
 *
 * In the run this skill came from, 9 of 24 hand-derived diagram indices were wrong;
 * QA agents were sent to bullet slides and ~400k tokens were wasted. The data to
 * compute them was always right there.
 *
 * Usage:
 *   node indices.mjs <slidesDir>                 # inventory: layout mix + every slide
 *   node indices.mjs <slidesDir> diagram         # JSON args array for a QA workflow
 *   node indices.mjs <slidesDir> diagram,chart   # several layouts
 *   node indices.mjs <slidesDir> --figures       # shorthand for diagram,chart
 */
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const [, , DIR, SEL] = process.argv;
if (!DIR) { console.error('usage: indices.mjs <slidesDir> [layout,layout|--figures]'); process.exit(1); }

const want = SEL === '--figures' ? ['diagram', 'chart'] : SEL ? SEL.split(',') : null;

let n = 0;
const all = [];
for (const f of readdirSync(DIR).filter(x => x.endsWith('.json')).sort()) {
  const arr = JSON.parse(readFileSync(join(DIR, f), 'utf8'));
  arr.forEach((s, i) => {
    n++;
    all.push({ file: f, idx: i + 1, n, layout: s.layout,
               title: (s.title || s.claim || s.quote || s.lead || '').slice(0, 46) });
  });
}

if (want) {
  // exactly the shape a Workflow `args` array needs
  console.log(JSON.stringify(all.filter(s => want.includes(s.layout))
    .map(({ file, idx, n }) => ({ file, idx, n }))));
} else {
  const by = {};
  all.forEach(s => by[s.layout] = (by[s.layout] || 0) + 1);
  console.log(`${n} slides in ${DIR}\n`);
  Object.entries(by).sort((a, b) => b[1] - a[1])
    .forEach(([k, v]) => console.log(`  ${k.padEnd(10)} ${String(v).padStart(3)}  ${String(Math.round(v / n * 100)).padStart(3)}%`));
  console.log('');
  all.forEach(s => console.log(
    `${String(s.n).padStart(3)}  ${s.file.padEnd(24)} #${String(s.idx).padEnd(3)} ${s.layout.padEnd(9)} ${s.title}`));
}
