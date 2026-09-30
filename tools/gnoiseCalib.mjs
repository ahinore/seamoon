// gnoiseCalib.mjs — 3D gradient noise (Perlin) mirror of the planned GLSL:
// measure the renormalized 3-octave distribution to calibrate thresholds.
const hash13 = (x, y, z) => {
  let p3 = [x * 0.1031, y * 0.1031, z * 0.1031].map((v) => v - Math.floor(v));
  const d = p3[0] * (p3[2] + 31.32) + p3[1] * (p3[0] + 31.32) + p3[2] * (p3[1] + 31.32);
  p3 = p3.map((v) => v + d);
  return ((p3[0] + p3[1]) * p3[2]) % 1;
};
const wrapv = (v) => ((v % 2048) + 2048) % 2048;
const quintic = (t) => t * t * t * (t * (t * 6 - 15) + 10);
// gradient direction on the unit sphere from two hashes (same as GLSL plan)
function gradAt(cx, cy, cz) {
  const h1 = hash13(cx, cy, cz);
  const h2 = hash13(cx + 19.19, cy + 19.19, cz + 19.19);
  const z = h1 * 2 - 1;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  const th = h2 * Math.PI * 2;
  return [r * Math.cos(th), r * Math.sin(th), z];
}
function gnoise3(xv, yv, zv) {
  xv = wrapv(xv); yv = wrapv(yv); zv = wrapv(zv);
  const ix = Math.floor(xv), iy = Math.floor(yv), iz = Math.floor(zv);
  const fx = xv - ix, fy = yv - iy, fz = zv - iz;
  const ux = quintic(fx), uy = quintic(fy), uz = quintic(fz);
  const i1x = wrapv(ix + 1), i1y = wrapv(iy + 1), i1z = wrapv(iz + 1);
  const m = (a, b, t) => a + (b - a) * t;
  const d = (g, dx, dy, dz) => g[0] * dx + g[1] * dy + g[2] * dz;
  const n000 = d(gradAt(ix, iy, iz), fx, fy, fz);
  const n100 = d(gradAt(i1x, iy, iz), fx - 1, fy, fz);
  const n010 = d(gradAt(ix, i1y, iz), fx, fy - 1, fz);
  const n110 = d(gradAt(i1x, i1y, iz), fx - 1, fy - 1, fz);
  const n001 = d(gradAt(ix, iy, i1z), fx, fy, fz - 1);
  const n101 = d(gradAt(i1x, iy, i1z), fx - 1, fy, fz - 1);
  const n011 = d(gradAt(ix, i1y, i1z), fx, fy - 1, fz - 1);
  const n111 = d(gradAt(i1x, i1y, i1z), fx - 1, fy - 1, fz - 1);
  return m(
    m(m(n000, n100, ux), m(n010, n110, ux), uy),
    m(m(n001, n101, ux), m(n011, n111, ux), uy),
    uz);
}
// fbm: 3 octaves 70/32.2/14.8 km, warp, rotation — mirror of fbm3oLod
function fbmLod(px, py, pz, kmps) {
  let a = 0.5, s = 0, asum = 0;
  const fade = (ck) => Math.min(1, Math.max(0, (ck * 0.5 / kmps - 3) / 5));
  const w = gnoise3(px * 0.35 + 17.3, py * 0.35 + 17.3, pz * 0.35 + 17.3) * 0.7;
  let f = fade(70);
  s += a * (0.5 + gnoise3(px + w, py + w * 0.8, pz + w * 1.1) * 1.2) * f; asum += a * f;
  let q = [py * 2.17, pz * 2.17, px * 2.17]; // ROT3 swap
  px = q[0]; py = q[1]; pz = q[2]; a *= 0.5;
  f = fade(70 / 2.17);
  s += a * (0.5 + gnoise3(px + w, py + w * 0.8, pz + w * 1.1) * 1.2) * f; asum += a * f;
  q = [py * 2.17, pz * 2.17, px * 2.17];
  px = q[0]; py = q[1]; pz = q[2]; a *= 0.5;
  f = fade(70 / (2.17 * 2.17));
  s += a * (0.5 + gnoise3(px + w, py + w * 0.8, pz + w * 1.1) * 1.2) * f; asum += a * f;
  return s / Math.max(asum, 0.15);
}
const R = 6374;
const vals = [];
for (let i = 0; i < 30000; i++) {
  const th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
  const x = R * Math.sin(ph) * Math.cos(th), y = R * Math.cos(ph), z = R * Math.sin(ph) * Math.sin(th);
  vals.push(fbmLod(x / 70, y / 70, z / 70, 2.51));
}
vals.sort((a, b) => a - b);
const q = (p) => vals[Math.floor(p * vals.length)];
console.log(`gnoise fbm: min=${vals[0].toFixed(3)} q10=${q(0.1).toFixed(3)} q30=${q(0.3).toFixed(3)} q50=${q(0.5).toFixed(3)} q70=${q(0.7).toFixed(3)} q90=${q(0.9).toFixed(3)} max=${vals[vals.length - 1].toFixed(3)}`);
const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
const sd = Math.sqrt(vals.reduce((s, v) => s + (v - mean) * (v - mean), 0) / vals.length);
console.log(`mean=${mean.toFixed(3)} std=${sd.toFixed(3)}`);
// grid-node structure check: sample along a straight line in cell units —
// gradient noise must be ~0 at integer nodes (no plateaus):
let nodeVals = [];
for (let i = 0; i < 200; i++) nodeVals.push(gnoise3(i, i * 0.3, i * 0.7));
const nodeAbsMean = nodeVals.reduce((a, b) => a + Math.abs(b), 0) / nodeVals.length;
console.log(`|gnoise| at lattice nodes (should be ~0): ${nodeAbsMean.toFixed(4)}`);
