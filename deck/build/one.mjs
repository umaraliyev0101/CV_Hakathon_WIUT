#!/usr/bin/env node
/* Render ONE slide from a spec file to PNG, for iterating on a fix.
   Usage: node one.mjs <slides/file.json> <1-based index in that file> <out.png> */
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const __dir = dirname(fileURLToPath(import.meta.url));
const [, , FILE, IDX, OUT] = process.argv;
const tmp = mkdtempSync(join(tmpdir(), 'slide-'));
const one = join(tmp, 'one.json');
const arr = JSON.parse(readFileSync(FILE, 'utf8'));
const s = arr[Number(IDX) - 1];
if (!s) { console.error(`no slide #${IDX} in ${FILE} (has ${arr.length})`); process.exit(1); }
writeFileSync(one, JSON.stringify([s]));
execFileSync('node', [join(__dir, 'render-html.mjs'), tmp, join(tmp, 'd.html')], { stdio: 'inherit' });

// Strip the on-screen page chrome before screenshotting. design.css adds a dark page
// background, 24px of body padding and a drop shadow under @media screen — without this
// the capture shows a dark band on top, sits 24px low, and CLIPS THE BOTTOM 24px, which
// is exactly where the footer and source line live. A QA agent would then miss a real
// bottom-edge clip, or report one that isn't there. shoot.mjs already does this.
const deck = join(tmp, 'd.html');
writeFileSync(deck, readFileSync(deck, 'utf8').replace('</head>',
  '<style>html,body{background:#fff!important;padding:0!important;margin:0!important}' +
  '.slide{margin:0!important;box-shadow:none!important}</style></head>'));
execFileSync('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ['--headless', '--disable-gpu', '--hide-scrollbars', '--window-size=1280,720',
   `--screenshot=${OUT}`, join(tmp, 'd.html')], { stdio: 'ignore' });
console.log(`-> ${OUT}`);
