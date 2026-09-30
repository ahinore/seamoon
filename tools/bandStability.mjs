// bandStability.mjs — why does the deck rebuild when altitude changes
// between ~2.5Mm and ~224km, but is stable above and below?
// Hypothesis: cellKm = clamp(kmPerPx*30, 6, 70) is UNCLAMPED in exactly
// that band — the world→lattice mapping RESCALES continuously with
// altitude, so every altitude step re-samples the field at a different
// scale → threshold crossings everywhere → "clouds rebuilt".
// Compare coverage agreement at matched world points for:
//   A) current: cellKm follows pixels (rescales in the band)
//   B) fixed octaves 70/32/15 km (world-anchored, fades only)
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
function fbm4(x, y, z) { let a = 0.5, s = 0; for (let i = 0; i < 4; i++) { s += a * noise3(x, y, z); x *= 2.13; y *= 2.13; z *= 2.13; a *= 0.5; } return s; }
const tan30 = 0.5774, VH = 863;
const fadePx = (cellKm, kmPerPx) => Math.min(1, Math.max(0, (cellKm * 0.5 / kmPerPx - 3) / 5));
// three-octave LOD fbm with EXPLICIT cells (world-anchored):
function fbmFixed(px, py, pz, cells, kmPerPx) {
  let s = 0, asum = 0;
  const amps = [0.5, 0.25, 0.125];
  for (let o = 0; o < 3; o++) {
    const f = fadePx(cells[o], kmPerPx);
    s += amps[o] * noise3(px, py, pz) * f; asum += amps[o] * f;
    px *= 2.17; py *= 2.17; pz *= 2.17;
  }
  return s / Math.max(asum, 0.15);
}
// current per-pixel-cell version (single lattice + fades):
function fbmCur(px, py, pz, cellKm, kmPerPx) {
  let s = 0, asum = 0, ck = cellKm;
  const amps = [0.5, 0.25, 0.125];
  for (let o = 0; o < 3; o++) {
    const f = fadePx(ck, kmPerPx);
    s += amps[o] * noise3(px, py, pz) * f; asum += amps[o] * f;
    px *= 2.17; py *= 2.17; pz *= 2.17; ck /= 2.17;
  }
  return s / Math.max(asum, 0.15);
}
const R = 6374;
function coverAt(up, alt, mode) {
  const kmPerPx = 2 * alt * tan30 / VH / 1000;
  const weather = fbm4(up[0] * 2.2, up[1] * 2.2, up[2] * 2.2);
  const lat = Math.asin(Math.max(-1, Math.min(1, up[1])));
  const bands = 0.55 + 0.45 * Math.cos(lat * 6) * 0.5 + 0.25 * Math.exp(-Math.pow((Math.abs(lat) - 0.15) * 3, 2));
  let cv = 0.42 * bands * 1.6 * weather + (weather - 0.5) * 0.4;
  cv = Math.min(1, Math.max(0, cv)); cv = Math.pow(cv, 0.7);
  const sp = [up[0] * R, up[1] * R, up[2] * R];
  let f1;
  if (mode === 'fixed') {
    f1 = fbmFixed(sp[0] / 70, sp[1] / 70, sp[2] / 70, [70, 32.2, 14.8], kmPerPx);
  } else {
    const cellKm = Math.min(70, Math.max(6, kmPerPx * 30));
    f1 = fbmCur(sp[0] / cellKm, sp[1] / cellKm, sp[2] / cellKm, cellKm, kmPerPx);
  }
  const sys = 0.62 * Math.min(1, Math.max(0, (weather - 0.30) / 0.32)) + 0.38 * f1;
  const thr = 0.60 + (0.50 - 0.60) * cv;
  let d = (sys - thr) / 0.18; d = Math.min(1, Math.max(0, d));
  const gate = Math.max(Math.min(1, Math.max(0, (weather - 0.36) / 0.19)), 0.12 * Math.min(1, Math.max(0, (weather - 0.15) / 0.2)));
  d *= gate; d = Math.min(1, d * 1.5);
  return d > 0.3;
}
// altitude pairs spanning the user's rebuild band:
const pairs = [[2670000, 2580000], [2580000, 1000000], [1000000, 500000], [500000, 224000], [224000, 200000]];
for (const [a1, a2] of pairs) {
  for (const mode of ['cur', 'fixed']) {
    let agree = 0, n = 0, w1 = 0;
    for (let i = 0; i < 15000; i++) {
      const th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
      const up = [Math.sin(ph) * Math.cos(th), Math.cos(ph), Math.sin(ph) * Math.sin(th)];
      const c1 = coverAt(up, a1, mode), c2 = coverAt(up, a2, mode);
      if (c1) w1++;
      if (c1 === c2) agree++;
      n++;
    }
    console.log(`${(a1 / 1e6).toFixed(2)}→${(a2 / 1e6).toFixed(2)}Mm  ${mode === 'cur' ? 'CURRENT(pixel-cell)' : 'FIXED(70/32/15km)'}: agreement=${(100 * agree / n).toFixed(1)}%  white@top=${(100 * w1 / n).toFixed(1)}%`);
  }
}
