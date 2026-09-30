// Same-session A/B: shot1 (both hulls) → toggle near hull off → shot2.
// Requires the page to expose window.__clouds (set in main.ts when
// ?abshot=1).
import puppeteer from 'puppeteer-core';
import { PNG } from 'pngjs';
import fs from 'fs';

const url = process.argv[2];
const out1 = process.argv[3];
const out2 = process.argv[4];
const waitMs = Number(process.argv[5] || 15000);
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
await new Promise(r => setTimeout(r, waitMs));
fs.writeFileSync(out1, await page.screenshot({ type: 'png' }));
await page.evaluate(() => { window.__clouds.children[0].visible = false; });
await new Promise(r => setTimeout(r, 1500));
fs.writeFileSync(out2, await page.screenshot({ type: 'png' }));
await browser.close();
console.log('saved', out1, out2);
