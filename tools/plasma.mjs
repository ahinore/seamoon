// Plasma probe: watch the #plasma overlay + heat during an entry.
import puppeteer from 'puppeteer-core';
import { PNG } from 'pngjs';
import fs from 'fs';

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const url = process.argv[2] || 'http://127.0.0.1:5173/?demo=orbital&alt=200000&pe=20000&notli=1&warp=100&probe=d&v=950&atmo=1';
const browser = await puppeteer.launch({
  executablePath: EDGE, headless: 'new',
  args: ['--use-angle=default', '--enable-unsafe-swiftshader', '--window-size=1600,900'],
  defaultViewport: { width: 1600, height: 900 },
});
const page = await browser.newPage();
await page.goto(url, { waitUntil: 'load', timeout: 30000 });

for (let t = 0; t < 34; t++) {
  await new Promise(r => setTimeout(r, 1000));
  if (t < 18 || t > 32) continue;
  const info = await page.evaluate(() => {
    const p = document.getElementById('plasma');
    const f = window.__flight;
    return { op: p ? p.style.opacity : null, heat: f ? Number(f.heat.toFixed(3)) : null, note: f ? f.note : '' };
  });
  const data = await page.screenshot({ type: 'png' });
  const png = PNG.sync.read(data);
  let orange = 0, n = 0;
  for (let i = 0; i < png.width * png.height; i += 11) {
    const r = png.data[i*4], g = png.data[i*4+1], b = png.data[i*4+2];
    n++;
    if (r > 150 && g < r * 0.75 && b < r * 0.5) orange++;
  }
  console.log(`t${t}: op=${info.op} heat=${info.heat} note=${info.note} orangePx=${(100*orange/n).toFixed(1)}%`);
  if (t === 23 || t === 25) fs.writeFileSync(`tmp/plasma_t${t}.png`, data);
}
await browser.close();
