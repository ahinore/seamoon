// fieldDrift.mjs — offline mirror of the shell field: how many points
// cross the density threshold over a 30s window at alt=218km?
const hash13 = (x, y, z) => {
  let p3 = [x * 0.1031, y * 0.1031, z * 0.1031].map((v) => v - Math.floor(v));
  const d = p3[0] * (p3[2] + 31.32) + p3[1] * (p3[0] + 31.32) + p3[2] * (p3[1] + 31.32);
  p3 = p3.map((v) => v + d);
  return ((p3[0] + p3[1]) * p3[2]) % 1;
};
const wrap = (v) => ((v % 2048) + 2048) % 2048;
const sm = (t) => t * t * (3 - 2 * t);
function noise3(xv, yv, zv) {
  xv = wrap(xv); yv = wrap(yv); zv = wrap(zv);
  const ix = Math.floor(xv), iy = Math.floor(yv), iz = Math.floor(zv);
  const fx = sm(xv - ix), fy = sm(yv - iy), fz = sm(zv - iz);
  const i1x = wrap(ix + 1), i1y = wrap(iy + 1), i1z = wrap(iz + 1);
  const m = (a, b, t) => a + (b - a) * t;
  return m(
    m(m(hash13(ix, iy, iz), hash13(i1x, iy, iz), fx), m(hash13(ix, i1y, iz), hash13(i1x, i1y, iz), fx), fy),
    m(m(hash13(ix, iy, i1z), hash13(i1x, iy, i1z), fx), m(hash13(ix, i1y, i1z), hash13(i1x, i1y, i1z), fx), fy),
    fz);
}
function fbmLod(px, py, pz, cellKm, kmps) {
  let a = 0.5, s = 0, asum = 0;
  const fade = (ck) => Math.min(1, Math.max(0, (ck * 0.5 / kmps - 1) / 2));
  let f = fade(cellKm); s += a * noise3(px, py, pz) * f; asum += a * f;
  px *= 2.17; py *= 2.17; pz *= 2.17; a *= 0.5; cellKm /= 2.17;
  f = fade(cellKm); s += a * noise3(px, py, pz) * f; asum += a * f;
  px *= 2.17; py *= 2.17; pz *= 2.17; a *= 0.5; cellKm /= 2.17;
  f = fade(cellKm); s += a * noise3(px, py, pz) * f; asum += a * f;
  return s / Math.max(asum, 0.15);
}
const kmPerPx = 0.292, cellKm = 8.75; // alt 218km
const DT = 30; // seconds
let cross = 0, n = 0, dmax = 0, dsum = 0;
const thr = 0.42;
for (let i = 0; i < 30000; i++) {
  // a fixed world point: lattice base around (700,300,500) cells (world km)
  const wx = 7000 + (i % 200) * 0.05, wy = 3000 + ((i * 7) % 200) * 0.05, wz = 5000 + ((i * 13) % 200) * 0.05; // km
  const f0 = fbmLod(
    wx / cellKm + 0.0,
    wy / cellKm + 0.0,
    wz / cellKm + 0.0, cellKm, kmPerPx);
  const windKm = 18 * DT / 1000; // 18 m/s * DT s, in km
  const f1 = fbmLod(
    wx / cellKm + windKm / cellKm,
    wy / cellKm + windKm * 1.3 / cellKm,
    wz / cellKm + -windKm * 0.8 / cellKm, cellKm, kmPerPx);
  const a0 = f0 > thr, a1 = f1 > thr;
  if (a0 !== a1) cross++;
  const dd = Math.abs(f1 - f0); dsum += dd; if (dd > dmax) dmax = dd;
  n++;
}
console.log(`threshold crossings over ${DT}s: ${(100 * cross / n).toFixed(2)}%  mean|df|=${(dsum / n).toFixed(4)} max|df|=${dmax.toFixed(4)}`);
// Distribution of the renormalized field vs threshold:
const vals = [];
for (let i = 0; i < 20000; i++) {
  const wx = 7000 + (i % 200) * 0.05, wy = 3000 + ((i * 7) % 200) * 0.05, wz = 5000 + ((i * 13) % 200) * 0.05;
  vals.push(fbmLod(wx / cellKm, wy / cellKm, wz / cellKm, cellKm, kmPerPx));
}
vals.sort((a, b) => a - b);
const q = (p) => vals[Math.floor(p * vals.length)];
console.log(`renorm f1: min=${vals[0].toFixed(3)} q25=${q(0.25)} q50=${q(0.5)} q70=${q(0.7)} q85=${q(0.85)} max=${vals[vals.length - 1].toFixed(3)}`);
console.log(`thr ${thr} sits at quantile ${(vals.filter((v) => v < thr).length / vals.length).toFixed(2)}`);
// NOTE: wind is in METERS here but wx in KM — the wind/cellKm term is
// 18*30/8.75 = 61.7 CELL UNITS?! In the shader: pwF = shellP(m)/(cellKm*1000)
// + wind(m)/(cellKm*1000) — both in meters, divided by cellKm*1000 (meters
// per cell). Here wx is in km so the wind term must be km/cellKm.
// 18 m/s * 30 s = 540 m = 0.54 km → 0.54/8.75 = 0.062 cells. But I wrote
// 18*DT/cellKm = 61.7 — 1000x TOO BIG in this mirror (mixing m and km).
// That means the SHADER might have the same bug! Check: shader wind =
// driftM = uTime*18 (meters). pwF = shellP/(cellKm*1000) + wind/(cellKm*1000).
// wind/(cellKm*1000) = 540/8750 = 0.062 cells — CORRECT in shader.
// My mirror: wx/cellKm with wx in km = right; wind must be 0.54 km → 0.54/cellKm.
