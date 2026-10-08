import * as THREE from 'three';
import { fbm3, ridged3, hash3i, hash3a } from './noise';

/**
 * Moon terrain function (Phase 9) — the lunar counterpart of terrain.ts.
 *
 * Same contract as Earth's terrainHeight: unit direction in, elevation in
 * meters out, deterministic, CPU-only, sampled by the tile generator (and
 * later by collision). Reuses the SAME cube-sphere LOD pipeline — the moon
 * is "the planet system with different parameters" (no ocean, no
 * atmosphere-driven fog, grayscale regolith palette).
 *
 * Composition:
 *  - maria: low-frequency mask; inside basins the relief is damped and the
 *    palette darkens (basaltic plains).
 *  - highlands: gentle ridged relief + a medium roughness octave.
 *  - craters: 4 lattice scales (giant basins -> 20 m bowllets), each cell
 *    proposing ONE crater with a power-law radius tied to the cell size
 *    (rad <= 0.42 * cell). Profile = flat-floored bowl + gaussian rim ring
 *    + central peak for large radii. Depth scales DOWN for basin scales
 *    (real basins are shallow relative to diameter).
 *
 * Placement geometry: the lattice lives in moon-centered meters; a cell's
 * crater center is its jittered 3D center PROJECTED onto the sphere
 * (direction preserved, radius normalized). Distances are chords on the
 * sphere. With jitter <= 0.15 cell and rad <= 0.42 cell the profile support
 * (1.8 * rad <= 0.76 cell) never reaches past the 3x3x3 neighborhood —
 * proven bound, so no seams and no missed craters.
 */

export const MOON_SEED = 4242;

/** Moon radius, meters (real: 1,737,400 m). */
export const R_MOON = 1_737_000;
/** Max lunar elevation for color mapping. */
export const MOON_MAX_ELEV = 10800;

const R_M = R_MOON; // for wavelength math only

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
}

/** Octave visibility for mesh vertex spacing — same contract as Earth's. */
function moonOctaveFade(freq: number, spacing: number): number {
  if (spacing <= 0) return 1;
  const wl = R_M / freq;
  return smoothstep(2 * spacing, 4 * spacing, wl);
}

/**
 * One crater's elevation at chord distance `s` from its center.
 * Negative bowl + gaussian rim (+ central peak for big radii).
 */
function craterProfile(s: number, radius: number, depth: number): number {
  const r = s / radius;
  if (r >= 1.8) return 0;
  const bowl = -depth * (1 - smoothstep(0.55, 0.95, r));
  const rim = depth * 0.4 * Math.exp(-Math.pow((r - 1.0) / 0.22, 2));
  let v = bowl + rim;
  // central peak for complex craters (rad > 6 km); sigma = 0.16·radius
  if (radius > 6000) v += depth * 0.22 * Math.exp(-Math.pow(r / 0.16, 2));
  return v;
}

/**
 * Craters from one lattice scale. Returns the summed elevation contribution.
 * cellM: lattice cell size (m). dScale: depth multiplier for this scale.
 * occupancy: hash threshold — a cell hosts a crater when hash <= threshold.
 *
 * M11w14 de-latticing (the user read the old field as "too regular"):
 *  - radius law flattened to pow(h,1.6): per-scale sizes spread wide.
 *  - jitter amplitude 1.3 cell (±0.65 cell displacement): the nearest-
 *    neighbor graph of the lattice breaks — no more visible rows.
 *    Seam bound (per axis): a missed ring-2 center is ≥ 1.5 − jAmp/2 =
 *    0.85 cells away from any sample while the profile support is at most
 *    1.8·radC = 0.72 cells — continuous across loop edges.
 *  - occupancy lowered (sparse fields) and modulated by a regional clump
 *    mask: clusters + voids instead of an even sprinkle.
 *  - 18% of occupied cells add a SECOND smaller crater anywhere in the
 *    cell, killing the one-per-cell uniformity.
 * (A first attempt used a flat ±0.45 jitter with a ±2 loop — correct but
 * 4.6× the height-field cost; tile builds stalled the workers AND the
 * inline-urgent main-thread budget at fps 9, so it was rolled back.)
 */
export function craterScale(
  px: number, py: number, pz: number,
  cellM: number, dScale: number, occupy: number,
  spacing: number, seedBase: number,
): number {
  // Regional density mask, anchored to super-blocks of the crater lattice
  // (block ≈ 400 km via a per-scale bit shift, plus a 4x-finer medium
  // layer). M11w14 lesson: the first version evaluated the cluster mask at
  // the SAMPLE, so occupancy swayed with the sample position and whole
  // craters popped in/out mid-cell — hard 100+ m steps in the height
  // field. Anchoring the mask to the CELL makes occupancy sample-independent
  // and the field continuous everywhere.
  const sh = Math.max(1, Math.round(Math.log2(400000 / cellM)));
  // occupancy sways ±0.22 around the base threshold
  const occFor = (ix: number, iy: number, iz: number): number =>
    occupy - (hash3a(ix >> sh, iy >> sh, iz >> sh, seedBase + 91) - 0.5) * 0.30 -
    (hash3a(ix >> (sh - 2), iy >> (sh - 2), iz >> (sh - 2), seedBase + 92) - 0.5) * 0.14;
  const gx = Math.floor(px / cellM);
  const gy = Math.floor(py / cellM);
  const gz = Math.floor(pz / cellM);
  // mesh can't resolve craters below ~3*spacing (LOD fade window)
  const minRad = spacing > 0 ? 3 * spacing : 0;
  let sum = 0;
  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const ix = gx + dx, iy = gy + dy, iz = gz + dz;
        // Poisson-like count per cell (M11w14): a cell holds 0..3 craters —
        // empty cells and multi-crater cells both occur, so nearest-neighbor
        // spacing varies from ~0 to ~1.5 cells. This (not jitter alone) is
        // what finally erases the one-crater-per-cell lattice read. 12% of
        // occupied cells hold 2-3 craters; the regional mask sways occEff so
        // dense regions get more of the multi-crater cells.
        const occEff = occFor(ix, iy, iz);
        const h0 = hash3a(ix, iy, iz, seedBase);
        if (h0 > occEff) continue;
        const h2 = hash3a(ix, iy, iz, seedBase + 1);
        const h3 = hash3a(ix, iy, iz, seedBase + 2);
        const count = h0 > occEff ? 0 : h2 < 0.12 ? (h3 < 0.3 ? 3 : 2) : 1;
        for (let k = 0; k < count; k++) {
          // Size law skewed small-but-present: h^2.2 keeps many small + a
          // solid tail of large in every neighborhood (a pure Pareto
          // 0.03/sqrt(1-h) was tried and REJECTED — it starves the surface:
          // E[rad²] dropped ~6x and most samples fell in no crater at all,
          // leaving the terrain smooth/bare). Regional size factor: dense
          // blocks skew smaller (secondary fields), sparse blocks bigger.
          // The 0.40 cap is applied AFTER the factor so the seam bound
          // (support 1.8·radC ≤ 0.72 cell < 1.0 cell) still holds.
          const h2k = hash3a(ix, iy, iz, seedBase + 10 * k + 1);
          const reg = occEff - occupy;
          const radC = Math.min(0.40, (0.03 + 0.37 * Math.pow(h2k, 2.2)) * (1 - 1.2 * reg));
          // center anywhere inside the cell (full-cell jitter): a ring-2
          // center can then approach no closer than 1.0 cell while the
          // profile support tops out at 1.8·radC = 0.72 cells — no
          // discontinuity at the ±1 loop boundary.
          const rad = radC * cellM;
          if (rad <= minRad) continue;
          const jx = hash3a(ix, iy, iz, seedBase + 10 * k + 3);
          const jy = hash3a(ix, iy, iz, seedBase + 10 * k + 4);
          const jz = hash3a(ix, iy, iz, seedBase + 10 * k + 5);
          const bx = (ix + jx) * cellM;
          const by = (iy + jy) * cellM;
          const bz = (iz + jz) * cellM;
          // everything inlined: this loop is the single hottest function in
          // the sim (4 scales × 27 cells × 4225 verts per tile) — a
          // helper-call version measured ~2 ms/tile of pure call overhead.
          // bbox reject first (covers ~90% of cells) before any exp/sqrt.
          // M11w14 critical fix: the centers used to be SPHERE-PROJECTED
          // (bx·invLen) before the distance test, which shoved a center up
          // to ~8 km tangentially (invLen−1 ≈ 0.46% at this radius) —
          // far enough to leave its cell, so craters popped in/out at
          // window shifts as hard 100+ m steps in the height field. The
          // direct chord distance keeps every center inside its own cell,
          // making the window math exact (and saves a sqrt per crater).
          const rr = 1.8 * rad;
          const dx1 = px - bx, dy1 = py - by, dz1 = pz - bz;
          if (dx1 < rr && dx1 > -rr && dy1 < rr && dy1 > -rr && dz1 < rr && dz1 > -rr) {
            const s = Math.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
            const w = spacing > 0 ? smoothstep(3 * spacing, 8 * spacing, rad) : 1;
            if (w > 0) {
              const depth = rad * (0.04 + 0.14 * hash3a(ix, iy, iz, seedBase + 10 * k + 2)) * dScale;
              sum += craterProfile(s, rad, depth) * w;
            }
          }
        }
      }
    }
  }
  return sum;
}

/**
 * Moon surface elevation above the sphere, meters.
 * @param x,y,z unit direction (double)
 * @param spacing mesh vertex spacing for LOD fading (0 = full detail)
 */
export function moonHeight(x: number, y: number, z: number, spacing = 0): number {
  // --- maria: large dark basins (low-frequency mask) ---
  const mare = fbm3(x * 2.3, y * 2.3, z * 2.3, MOON_SEED, 3);
  const mareMask = smoothstep(0.18, 0.42, mare);

  // --- highlands relief ---
  const hF = moonOctaveFade(60, spacing);
  let h = (ridged3(x * 60, y * 60, z * 60, MOON_SEED + 71, 4) - 0.45) *
          2400 * hF * (1 - mareMask * 0.75);
  // medium roughness (~5 km wavelength) between the craters
  const rF = moonOctaveFade(340, spacing);
  h += fbm3(x * 340, y * 340, z * 340, MOON_SEED + 77, 3) *
       260 * rF * (1 - mareMask * 0.6);

  const px = x * R_M, py = y * R_M, pz = z * R_M;

  // crater scales: giant basins (not flattened by maria — they predate it)
  h += craterScale(px, py, pz, 140000, 0.45, 0.95, spacing, MOON_SEED + 400) *
       (1 - mareMask * 0.3);
  // regolith-scale craters — sparse base rates (the regional block mask +
  // multi-crater cells supply density variation), partially drowned in maria
  const craters =
    craterScale(px, py, pz, 26000, 1.0, 0.42, spacing, MOON_SEED + 500) +
    craterScale(px, py, pz, 5200, 0.55, 0.40, spacing, MOON_SEED + 600) +
    craterScale(px, py, pz, 1000, 0.30, 0.38, spacing, MOON_SEED + 700);
  h += craters * (1 - mareMask * 0.8);

  return h;
}

/**
 * Lunar surface color: grayscale regolith, darkened inside maria and on
 * steep slopes. Same signature as Earth's terrainColor.
 */
export function moonColor(
  x: number, y: number, z: number, h: number, slope: number,
): [number, number, number] {
  const mare = fbm3(x * 2.3, y * 2.3, z * 2.3, MOON_SEED, 3);
  const mareMask = smoothstep(0.18, 0.42, mare);
  // regolith brightness: slightly brighter on highlands, darker in lows
  let b = 0.50 + 0.10 * Math.min(Math.max(h / 4000, -1), 1);
  // slope darkening (steep crater walls expose fresher, darker rock)
  b -= Math.min(slope * 0.35, 0.18);
  // maria: darker with a subtle cool tint
  const r = b * (1 - mareMask * 0.40);
  const g = b * (1 - mareMask * 0.36);
  const bl = b * (1 - mareMask * 0.28);
  return [r, g, bl];
}

/**
 * M11n9r: global fallback sphere for the moon — the black-band safety net.
 *
 * The quadtree's build/evict budget cannot always fill the grazing horizon
 * ring (from 15 km up the ring sits ~230 km out and wants deep levels for
 * every compass bearing), so the band rendered as bare black background.
 * This coarse displaced sphere — same moonHeight/moonColor functions, ~109
 * km quads at res 32/face — renders BENEATH the tiles as a guaranteed
 * surface: the band reads as distant terrain instead of void, and the far
 * side of the moon always exists for orbit views.
 *
 * A 3 km inset keeps the fallback strictly below the tile surfaces: 24-bit
 * depth precision at the band's 200+ km distance is ~1 km, so without the
 * inset the coincident surfaces would z-fight (the fallback winning would
 * smear low-res terrain over near tiles). Where a crater bowl dips deeper
 * than the inset the bowl's own tile geometry is between the camera and
 * the fallback, so nothing shows through.
 */
export function buildMoonFallbackGeometry(resPerFace = 32, inset = 3000): THREE.BufferGeometry {
  // cube face axes: [normal, u, v] — right-handed per face
  const n = new THREE.Vector3();
  const u = new THREE.Vector3();
  const v = new THREE.Vector3();
  const faces: [number[], number[], number[]][] = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
  ];
  const R = R_MOON;
  const res = resPerFace;
  const vertsPerFace = (res + 1) * (res + 1);
  const total = vertsPerFace * 6;
  const pos = new Float32Array(total * 3);
  const nrm = new Float32Array(total * 3);
  const col = new Float32Array(total * 3);
  const grid = new Float32Array(total * 3);
  const ctr = new Float32Array(total * 3);
  const idx = new Uint32Array(res * res * 6 * 6);
  const dir = new THREE.Vector3();
  let vi = 0;
  for (let f = 0; f < 6; f++) {
    n.fromArray(faces[f][0]); u.fromArray(faces[f][1]); v.fromArray(faces[f][2]);
    const base = f * vertsPerFace;
    for (let j = 0; j <= res; j++) {
      for (let i = 0; i <= res; i++) {
        const su = -1 + (2 * i) / res;
        const sv = -1 + (2 * j) / res;
        dir.copy(n).addScaledVector(u, su).addScaledVector(v, sv).normalize();
        const dx = dir.x, dy = dir.y, dz = dir.z;
        const h = moonHeight(dx, dy, dz, 0);
        const r = R + h - inset;
        pos[vi * 3] = dx * r; pos[vi * 3 + 1] = dy * r; pos[vi * 3 + 2] = dz * r;
        const [cr, cg, cb] = moonColor(dx, dy, dz, h, 0);
        col[vi * 3] = cr; col[vi * 3 + 1] = cg; col[vi * 3 + 2] = cb;
        grid[vi * 3] = 0; grid[vi * 3 + 1] = 0; grid[vi * 3 + 2] = 0;
        ctr[vi * 3] = dx * R; ctr[vi * 3 + 1] = dy * R; ctr[vi * 3 + 2] = dz * R;
        vi++;
      }
    }
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const a = base + j * (res + 1) + i;
        const b = a + 1;
        const c = a + res + 1;
        const d = c + 1;
        let k = (f * res * res + j * res + i) * 6;
        idx[k++] = a; idx[k++] = c; idx[k++] = b;
        idx[k++] = b; idx[k++] = c; idx[k++] = d;
      }
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('aGrid', new THREE.BufferAttribute(grid, 3));
  geo.setAttribute('center', new THREE.BufferAttribute(ctr, 3));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  geo.computeVertexNormals();
  // weld normals across the 12 cube-face seams: computeVertexNormals leaves
  // duplicated border vertices with one-sided normals — average the
  // duplicates so no shading seams stripe the fallback sphere
  const weld = new Map<string, number[]>();
  const key = (x: number, y: number, z: number) =>
    `${Math.round(x / 500)},${Math.round(y / 500)},${Math.round(z / 500)}`;
  for (let i = 0; i < total; i++) {
    const k = key(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    let e = weld.get(k);
    if (!e) { e = [0, 0, 0, i]; weld.set(k, e); }
    e[0] += nrm[i * 3]; e[1] += nrm[i * 3 + 1]; e[2] += nrm[i * 3 + 2];
  }
  weld.forEach((e) => {
    const len = Math.hypot(e[0], e[1], e[2]) || 1;
    for (let c = 0; c < 3; c++) nrm[e[3] * 3 + c] = e[c] / len;
  });
  geo.getAttribute('normal').needsUpdate = true;
  return geo;
}
