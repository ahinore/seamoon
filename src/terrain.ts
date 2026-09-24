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
export const MAX_ELEV = 9200;

export function terrainHeight(x: number, y: number, z: number): number {
  // continent mask: very low frequency, sharp-ish land/ocean split
  const c = fbm3(x * 1.2, y * 1.2, z * 1.2, SEED, 4);
  const land = smoothstep(-0.08, 0.12, c);

  // Domain warping: bend the input of the ridge/hill layers by a mid-frequency
  // noise field. Breaks up the radial symmetry of plain fBm and gives ranges
  // their curved, tectonic-looking sweep. Warp magnitude ~ 0.02 rad keeps
  // features inside their continent without smearing the coast mask itself.
  const wx = fbm3(x * 6 + 11.3, y * 6, z * 6, SEED + 901, 3);
  const wy = fbm3(x * 6, y * 6 + 7.7, z * 6, SEED + 902, 3);
  const wz = fbm3(x * 6, y * 6, z * 6 + 3.1, SEED + 903, 3);
  const W = 0.02;
  const px = x + wx * W;
  const py = y + wy * W;
  const pz = z + wz * W;

  // mountain ranges: ridged noise, 320 km -> 10 km wavelengths.
  // Ridged noise is already crest-sharpening; squaring the ridge term
  // narrows the crests further (Himalaya-like spine-and-valley profile)
  // so a taller amplitude doesn't just look like inflated hills.
  const mRaw = ridged3(px * 20, py * 20, pz * 20, SEED + 77, 6);
  const m = mRaw * mRaw;
  const mountainMask = land * smoothstep(0.06, 0.38, c);

  // rolling hills: 40 km -> 2.5 km wavelengths, everywhere on land
  const d = fbm3(px * 160, py * 160, pz * 160, SEED + 191, 5);

  // medium relief: 6 km -> 780 m wavelengths — the "foothill" band. Real
  // ranges are mostly foothills, not bare peaks; this octave carries most
  // of the visual texture seen from 1-20 km altitude.
  const g = fbm3(px * 1050, py * 1050, pz * 1050, SEED + 417, 4);

  // fine detail: 2.5 km -> 310 m wavelengths (matters below ~30 km altitude).
  // Slope/erosion damping: ridges are steep, and steep young relief carries
  // less fine sediment — attenuate the fine octave on high ridges. Cheap
  // stand-in for real erosion until Phase 10.
  const f = fbm3(px * 2560, py * 2560, pz * 2560, SEED + 313, 4);
  const erosion = 1 - 0.75 * mountainMask * m;

  // ocean floor: gentle negative relief, with hadal trenches in the deep
  // basins (ridged noise carves long narrow trenches ~10 km deep)
  const trench = ridged3(px * 14, py * 14, pz * 14, SEED + 611, 4);
  const ocean = -900 - 2600 * smoothstep(0.0, -0.6, c) - 6500 * smoothstep(0.55, 0.95, trench);

  const landElev =
    60 * land +                        // coastal plains baseline
    8200 * mountainMask * m +          // mountain ranges (Everest-class)
    700 * d * land +                   // hills
    420 * g * land +                   // foothills / medium relief
    160 * f * land * erosion;          // fine roughness, damped on ridges

  return land * landElev + (1 - land) * ocean;
}

const smoothstep = (e0: number, e1: number, x: number): number => {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
};

/**
 * Biome color: latitude (temperature) + elevation + slope + moisture noise,
 * per the strategy note's Phase 4. Returns linear RGB in [0,1].
 *
 *  - temperature falls with |lat| and with altitude (6.5 K/km lapse rate);
 *    the snowline therefore DROPS toward the poles (ice caps emerge naturally)
 *  - slope (0..1, tan of the terrain gradient vs. its wavelength) pushes
 *    vegetation to bare rock: steep = rock regardless of moisture
 *  - moisture noise drives desert <-> grass <-> forest; wet coasts green up
 *  - ocean: shallow -> deep gradient, waterline exactly at h=0 (Phase 7 will
 *    replace the ocean floor rendering with a real sea surface)
 *
 * `slope` arrives precomputed by the tile builder from the same height field
 * the mesh was displaced with, so shading and geometry can never disagree.
 */
export function terrainColor(
  x: number, y: number, z: number,
  h: number,
  slope = 0,
): [number, number, number] {
  const m = fbm3(x * 8, y * 8, z * 8, SEED + 555, 2); // moisture-ish [-1,1]

  if (h < 0) {
    // ocean: shallow -> deep
    const t = Math.min(-h / 3000, 1);
    return lerp3([0.12, 0.32, 0.42], [0.015, 0.06, 0.15], t);
  }

  // temperature: 0 at tropics/sea level, 1 at poles or high altitude.
  // Latitude dominates via a power curve; the lapse-rate term puts alpine
  // snowlines on tall ranges. At lat 72+ snowH <= 0 -> permanent ice caps;
  // high relief snows from ~2.9 km at mid latitudes, ~1 km at lat 60.
  const latRad = Math.asin(Math.min(Math.max(y, -1), 1));
  const temp =
    Math.pow(Math.abs(latRad) / (Math.PI / 2), 0.62) * 1.3 + // latitude term
    h / 6000 * 0.55 -                                         // lapse-rate term
    m * 0.06;                                                 // weather wobble
  const snowH = 2900 - temp * 3000;                           // snowline (m)

  if (h > snowH) return [0.93, 0.94, 0.96];     // snow / ice caps
  if (slope > 0.55 || h > snowH * 0.72) return [0.45, 0.42, 0.4]; // bare rock

  // tundra band just under the snowline
  if (h > snowH * 0.55) return lerp3([0.5, 0.48, 0.34], [0.38, 0.44, 0.3], m * 0.5 + 0.5);

  const dry = m < -0.15;
  if (h < 12) return dry ? [0.72, 0.62, 0.36] : [0.76, 0.7, 0.5]; // beach
  if (h > 600) {
    // highland: green-brown blend by moisture
    return lerp3([0.3, 0.42, 0.2], [0.48, 0.45, 0.24], m * 0.5 + 0.5);
  }
  // lowland: desert -> grass -> forest by moisture
  if (dry) return lerp3([0.72, 0.62, 0.36], [0.55, 0.55, 0.3], m + 0.5);
  return lerp3([0.28, 0.5, 0.22], [0.16, 0.36, 0.14], m * 0.5 + 0.5);
}

const lerp3 = (a: [number, number, number], b: [number, number, number], t: number): [number, number, number] => {
  const k = Math.min(Math.max(t, 0), 1);
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
};
