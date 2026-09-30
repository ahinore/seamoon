// Reentry telemetry grab: load an orbital run, sample HUD + plasma
// pixel stats over time. Usage: node tools/reentry.mjs "<url>" [seconds]
import puppeteer from 'puppeteer-core';
import { PNG } from 'pngjs';
import fs from 'fs';

const url = process.argv[2];
const secs = Number(process.argv[3] || 40);
const EDGE_PATHS = [
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
];
const executablePath = EDGE_PATHS.find(p => fs.existsSync(p));
const browser = await puppeteer.launch({
  executablePath, headless: 'new',
  args: ['--use-angle=default', '--enable-unsafe-swiftshader', '--disable-lcd-text', '--window-size=1600,900', '--hide-scrollbars', '--force-color-profile=srgb'],
  defaultViewport: { width: 1600, height: 900 },
});
const page = await browser.newPage();
await page.goto(url, { waitUntil: 'load', timeout: 30000 });

const t0 = Date.now();
let shotN = 0;
while ((Date.now() - t0) / 1000 < secs) {
  await new Promise(r => setTimeout(r, 4000));
  const hud = await page.evaluate(() => {
    const el = document.getElementById('hud');
    return el ? el.textContent.replace(/\s+/g, ' ').slice(0, 400) : '';
  });
  const data = await page.screenshot({ type: 'png' });
  const png = PNG.sync.read(data);
  const W = png.width, H = png.height;
  let n = 0, orange = 0, white = 0, dark = 0;
  for (let i = 0; i < W * H; i += 7) {
    const r = png.data[i*4], g = png.data[i*4+1], b = png.data[i*4+2];
    n++;
    if (r > 190 && g > 60 && g < 190 && b < 90) orange++;
    if (0.299*r + 0.587*g + 0.114*b > 235) white++;
    if (0.299*r + 0.587*g + 0.114*b < 20) dark++;
  }
  const entry = /ENTRY/.test(hud) ? ' ENTRY!' : '';
  const heat = /HEAT ([\d.]+MW\/m2\s+[\d.]+g)/.exec(hud);
  console.log(`t=${((Date.now()-t0)/1000).toFixed(0)}s plasma=${(100*orange/n).toFixed(1)}% white=${(100*white/n).toFixed(0)}% dark=${(100*dark/n).toFixed(0)}%${entry} ${heat ? heat[1] : ''}`);
  if (shotN < 4 && (orange / n > 0.02 || /ENTRY/.test(hud))) {
    fs.writeFileSync(`tmp/reentry_${shotN++}.png`, await page.screenshot({ type: 'png' }));
  }
}
await browser.close();
