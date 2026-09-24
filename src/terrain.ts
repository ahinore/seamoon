import { fbm3, ridged3 } from './noise';

/**
 * Terrain height function — the single source of truth for the planet surface
 * (Phase 3). CPU-only for now; the tile generator samples it directly, and
 * Phase 6 collision will use the very same function, guaranteeing
 * visual/physics agreement.
 *
 * Input: unit direction on the sphere (double).
 * Output: elevation above the sphere in meters.
 *
 * Frequency design (value-noise lattice on the unit sphere, ground wavelength
 * ~= R / freq): real mountain ranges carry 2-4 km of relief over 10-50 km
 * wavelengths, so the visible-slope energy must live around freq 130-1300.
 * Continents stay low-frequency (mask only); the octaves that dominate what
 * the eye sees from 1-100 km altitude are the ridge + hill layers.
 *
 * Precision note: at freq 1600*2^3 the hash lattice coords reach ~13k — well
 * inside exact int32 hashing. The finest detail octave (~600 m) sits far
 * above the L20 vertex spacing (0.15 m), so LOD sampling never starves.
 */
export const SEED = 1337;

/** Max plausible elevation, used for color mapping and bounds. */
export const MAX_ELEV = 5200;

export function terrainHeight(x: number, y: number, z: number): number {
  // continent mask: very low frequency, sharp-ish land/ocean split
  const c = fbm3(x * 1.2, y * 1.2, z * 1.2, SEED, 4);
  const land = smoothstep(-0.08, 0.12, c);

  // mountain ranges: ridged noise, 320 km -> 10 km wavelengths
  const m = ridged3(x * 20, y * 20, z * 20, SEED + 77, 6);
  const mountainMask = land * smoothstep(0.06, 0.38, c);

  // rolling hills: 40 km -> 2.5 km wavelengths, everywhere on land
  const d = fbm3(x * 160, y * 160, z * 160, SEED + 191, 5);

  // fine detail: 4 km -> 500 m wavelengths (matters below ~30 km altitude)
  const f = fbm3(x * 1600, y * 1600, z * 1600, SEED + 313, 4);

  // ocean floor: gentle negative relief
  const ocean = -900 - 2200 * smoothstep(0.0, -0.6, c);

  const landElev =
    60 * land +                        // coastal plains baseline
    3800 * mountainMask * m +          // mountain ranges
    500 * d * land +                   // hills
    160 * f * land;                    // fine roughness

  return land * landElev + (1 - land) * ocean;
}

const smoothstep = (e0: number, e1: number, x: number): number => {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
};

/**
 * Biome color for elevation + a moisture-ish noise.
 * Returns linear RGB in [0,1]. Deep ocean -> beach -> plains -> forest ->
 * rock -> snow, with the waterline exactly at h=0 (Phase 7 will replace
 * the ocean floor rendering with a real sea surface).
 */
export function terrainColor(x: number, y: number, z: number, h: number): [number, number, number] {
  const m = fbm3(x * 8, y * 8, z * 8, SEED + 555, 3); // moisture-ish [-1,1]

  if (h < 0) {
    // ocean: shallow -> deep
    const t = Math.min(-h / 3000, 1);
    return lerp3([0.12, 0.32, 0.42], [0.015, 0.06, 0.15], t);
  }
  if (h < 12) return [0.76, 0.7, 0.5]; // beach
  if (h > 3200 + 600 * m) return [0.93, 0.94, 0.96]; // snow
  if (h > 1800 + 400 * m) return [0.45, 0.42, 0.4]; // rock
  if (h > 600) {
    // highland: green-brown blend by moisture
    return lerp3([0.3, 0.42, 0.2], [0.48, 0.45, 0.24], m * 0.5 + 0.5);
  }
  // lowland: desert -> grass -> forest by moisture
  if (m < -0.25) return lerp3([0.72, 0.62, 0.36], [0.55, 0.55, 0.3], m + 0.5);
  return lerp3([0.28, 0.5, 0.22], [0.16, 0.36, 0.14], m * 0.5 + 0.5);
}

const lerp3 = (a: [number, number, number], b: [number, number, number], t: number): [number, number, number] => {
  const k = Math.min(Math.max(t, 0), 1);
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
};
