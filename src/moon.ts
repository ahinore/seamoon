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
