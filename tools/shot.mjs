// Self-serve screenshot harness: headless Edge via puppeteer-core.
// Usage: node tools/shot.mjs "<url>" out.png [waitMs]
// Writes the PNG even on WebGL/context trouble; prints console errors.
import puppeteer from 'puppeteer-core';
import { PNG } from 'pngjs';
import fs from 'fs';

const url = process.argv[2];
const out = process.argv[3];
const waitMs = Number(process.argv[4] || 12000);
if (!url || !out) { console.error('need <url> <out.png> [waitMs]'); process.exit(2); }

const EDGE_PATHS = [
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
];
const executablePath = EDGE_PATHS.find(p => fs.existsSync(p));
if (!executablePath) { console.error('Edge not found'); process.exit(2); }

const browser = await puppeteer.launch({
  executablePath,
  headless: 'new',
  args: [
    '--use-angle=default', '--enable-unsafe-swiftshader', '--disable-lcd-text',
    '--window-size=1600,900', '--hide-scrollbars', '--force-color-profile=srgb',
  ],
  defaultViewport: { width: 1600, height: 900 },
});
const page = await browser.newPage();
const logs = [];
page.on('console', m => { const t = m.text(); if (t) logs.push(t.slice(0, 300)); });
page.on('pageerror', e => logs.push('PAGEERROR: ' + String(e).slice(0, 300)));

try {
  await page.goto(url, { waitUntil: 'load', timeout: 30000 });
} catch (e) { logs.push('GOTO: ' + String(e).slice(0, 200)); }
await new Promise(r => setTimeout(r, waitMs));

// probe readout (HUD) if present
let hud = '';
try {
  hud = await page.evaluate(() => {
    const els = [...document.querySelectorAll('div,pre')].filter(e =>
      /fps|alt\s|drawcalls/i.test(e.textContent || '') && e.children.length === 0);
    return els.map(e => e.textContent).join(' | ').slice(0, 500);
  });
} catch {}

let shot = false, size = 0;
try {
  const data = await page.screenshot({ type: 'png' });
  fs.writeFileSync(out, data);
  shot = true; size = data.length;
} catch (e) { logs.push('SHOT: ' + String(e).slice(0, 200)); }

// quick stats from the PNG itself: mean luminance + unique-ish colors
let stats = '';
try {
  const png = PNG.sync.read(fs.readFileSync(out));
  const n = png.width * png.height; let sum = 0, bright = 0, dark = 0;
  for (let i = 0; i < n; i++) {
    const l = 0.299 * png.data[i*4] + 0.587 * png.data[i*4+1] + 0.114 * png.data[i*4+2];
    sum += l; if (l > 200) bright++; if (l < 20) dark++;
  }
  stats = `lum=${(sum/n).toFixed(1)} bright%=${(100*bright/n).toFixed(1)} dark%=${(100*dark/n).toFixed(1)}`;
} catch {}

console.log(JSON.stringify({ shot, size, out, stats, hud: hud.slice(0, 300), logs: logs.slice(0, 8) }, null, 1));
await browser.close();
