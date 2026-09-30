// shot2.mjs — two screenshots from ONE page session (measures temporal
// motion exactly as a user sees it, no reload between frames).
// usage: node tools/shot2.mjs "<url>" outA.png outB.png [wait1Ms] [wait2Ms]
import puppeteer from 'puppeteer-core';
import fs from 'fs';

const url = process.argv[2];
const outA = process.argv[3];
const outB = process.argv[4];
const wait1 = Number(process.argv[5] ?? 15000);
const wait2 = Number(process.argv[6] ?? 6000);
const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';

const browser = await puppeteer.launch({
  executablePath: EDGE,
  headless: 'new',
  args: ['--window-size=1740,900', '--disable-gpu-sandbox', '--use-angle=default'],
  defaultViewport: { width: 1728, height: 863 },
});
const page = await browser.newPage();
page.on('console', (m) => {
  const t = m.text();
  if (/error|ERROR/i.test(t)) console.log('[console]', t.slice(0, 200));
});
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
await new Promise((r) => setTimeout(r, wait1));
const hudA = await page.evaluate(() => document.getElementById('hud')?.innerText ?? '');
await page.screenshot({ path: outA });
await new Promise((r) => setTimeout(r, wait2));
await page.screenshot({ path: outB });
const hudB = await page.evaluate(() => document.getElementById('hud')?.innerText ?? '');
console.log(JSON.stringify({ hud: hudA.replace(/\n/g, ' | ').slice(0, 300), hud2: hudB.replace(/\n/g, ' | ').slice(0, 120) }));
await browser.close();
