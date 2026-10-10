#!/usr/bin/env node
/* Screenshot individual slides as PNG. Usage: node shoot.mjs <deck.html> <outDir> [n1,n2,...] */
import { readFileSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';

const [, , DECK, OUTDIR, WHICH] = process.argv;
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const html = readFileSync(DECK, 'utf8');
const style = html.match(/<style>([\s\S]*?)<\/style>/)[1];
const slides = html.match(/<section class="slide[\s\S]*?\n<\/section>/g) || [];
mkdirSync(OUTDIR, { recursive: true });
const tmp = join(OUTDIR, '_t');
mkdirSync(tmp, { recursive: true });
const want = WHICH ? WHICH.split(',').map(Number) : slides.map((_, i) => i + 1);
for (const n of want) {
  const s = slides[n - 1]; if (!s) continue;
  const f = join(tmp, `s${n}.html`);
  writeFileSync(f, `<!doctype html><html><head><meta charset="utf-8"><style>${style}
  html,body{background:#fff!important;padding:0!important;margin:0!important}
  .slide{margin:0!important;box-shadow:none!important}</style></head><body>${s}</body></html>`);
  execFileSync(CHROME, ['--headless', '--disable-gpu', '--hide-scrollbars',
    '--window-size=1280,720', `--screenshot=${join(OUTDIR, `slide-${String(n).padStart(3, '0')}.png`)}`, f],
    { stdio: 'ignore' });
}
rmSync(tmp, { recursive: true, force: true });
console.log(`shot ${want.length} slides -> ${OUTDIR}`);
