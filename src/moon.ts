import * as THREE from 'three';
import { fbm3, ridged3, hash3i } from './noise';

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
  // central peak for complex craters (rad > 6 km)
  if (radius > 6000) v += depth * 0.22 * Math.exp(-Math.pow(r / 0.16, 2));
  return v;
}

/**
 * Craters from one lattice scale. Returns the summed elevation contribution.
 * cellM: lattice cell size (m). dScale: depth multiplier for this scale.
 * occupancy: P(cell hosts a crater) = 1 - occupancy threshold.
 */
function craterScale(
  px: number, py: number, pz: number,
  cellM: number, dScale: number, occupy: number,
  spacing: number, seedBase: number,
): number {
  const gx = Math.floor(px / cellM);
  const gy = Math.floor(py / cellM);
  const gz = Math.floor(pz / cellM);
  let sum = 0;
  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const ix = gx + dx, iy = gy + dy, iz = gz + dz;
        if (hash3i(ix, iy, iz, seedBase) > occupy) continue;
        const h2 = hash3i(ix, iy, iz, seedBase + 1);
        const h3 = hash3i(ix, iy, iz, seedBase + 2);
        // power-law-ish radius (squaring biases small), tied to the cell
        const rad = cellM * 0.04 + h2 * h2 * cellM * 0.38;
        // LOD: fade craters the mesh cannot resolve (~8 samples across)
        const w = spacing > 0 ? smoothstep(3 * spacing, 8 * spacing, rad) : 1;
        if (w <= 0) continue;
        // jittered center (±0.15 cell), projected onto the sphere
        const jx = (hash3i(ix, iy, iz, seedBase + 3) - 0.5) * 0.3 * cellM;
        const jy = (hash3i(ix, iy, iz, seedBase + 4) - 0.5) * 0.3 * cellM;
        const jz = (hash3i(ix, iy, iz, seedBase + 5) - 0.5) * 0.3 * cellM;
        const cx = (ix + 0.5) * cellM + jx;
        const cy = (iy + 0.5) * cellM + jy;
        const cz = (iz + 0.5) * cellM + jz;
        const invLen = R_M / Math.sqrt(cx * cx + cy * cy + cz * cz);
        // chord distance on the sphere between sample dir and crater dir
        const sx = px - cx * invLen;
        const sy = py - cy * invLen;
        const sz = pz - cz * invLen;
        const s = Math.sqrt(sx * sx + sy * sy + sz * sz);
        const depth = rad * (0.06 + 0.10 * h3) * dScale;
        sum += craterProfile(s, rad, depth) * w;
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
  // regolith-scale craters, partially drowned inside maria
  const craters =
    craterScale(px, py, pz, 26000, 1.0, 0.55, spacing, MOON_SEED + 500) +
    craterScale(px, py, pz, 5200, 0.55, 0.50, spacing, MOON_SEED + 600) +
    craterScale(px, py, pz, 1000, 0.30, 0.45, spacing, MOON_SEED + 700);
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
