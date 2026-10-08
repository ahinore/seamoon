/**
 * Deterministic 3D value-noise on the unit sphere (Phase 3 foundation).
 *
 * Everything here is pure: same input dir => same output, on any machine,
 * in any thread (the eventual Worker shares this module). Integers drive the
 * hash so results are exact and free of float drift; double precision holds
 * down to the finest octaves we use (~1 m wavelength at Earth radius, where
 * vertex spacing bottoms out around 0.15 m — comfortable).
 */

const FNV_OFFSET = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

/** 32-bit FNV-1a over three integers + seed. */
function hash3(x: number, y: number, z: number, seed: number): number {
  let h = FNV_OFFSET ^ seed;
  // force unsigned 32-bit ops
  h = Math.imul(h ^ (x | 0), FNV_PRIME);
  h = Math.imul(h ^ (y | 0), FNV_PRIME);
  h = Math.imul(h ^ (z | 0), FNV_PRIME);
  return (h >>> 0) / 4294967295; // [0, 1)
}

/**
 * M11w14: avalanche variant used by the moon's crater lattice.
 *
 * FNV-1a with a single trailing multiply is an AFFINE map over consecutive
 * integer inputs — hash(i+1) − hash(i) is a near-constant 0.0039 — so
 * neighbouring lattice cells got nearly identical randoms: equal-size
 * craters lined up in bead-chain rows (the "too regular" look) and
 * near-constant jitter. This finalizer (murmur-style xor-shift rounds)
 * fully avalanches, breaking the neighbor correlation. Terrain noise keeps
 * plain hash3 so existing terrain/mare values (and the landing regression
 * baseline) stay byte-identical.
 */
export function hash3a(x: number, y: number, z: number, seed: number): number {
  let h = (0x811c9dc5 ^ seed) | 0;
  h = Math.imul(h ^ (x | 0), 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h ^ (y | 0), 0xc2b2ae35);
  h ^= h >>> 16;
  h = Math.imul(h ^ (z | 0), 0x27d4eb2f);
  h ^= h >>> 15;
  h = Math.imul(h, 0x9e3779b1);
  h ^= h >>> 13;
  return (h >>> 0) / 4294967296; // [0, 1)
}

const smooth = (t: number): number => t * t * (3 - 2 * t);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** Trilinear value noise in [0,1] at arbitrary 3D position, lattice ~1. */
export function valueNoise3(x: number, y: number, z: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const tx = smooth(x - xi);
  const ty = smooth(y - yi);
  const tz = smooth(z - zi);

  const n = (dx: number, dy: number, dz: number) =>
    hash3(xi + dx, yi + dy, zi + dz, seed);

  return lerp(
    lerp(
      lerp(n(0, 0, 0), n(1, 0, 0), tx),
      lerp(n(0, 1, 0), n(1, 1, 0), tx),
      ty,
    ),
    lerp(
      lerp(n(0, 0, 1), n(1, 0, 1), tx),
      lerp(n(0, 1, 1), n(1, 1, 1), tx),
      ty,
    ),
    tz,
  );
}

/**
 * Fractal Brownian motion (sum of octaves) in [-1, 1]-ish range.
 * `lacunarity` 2, `gain` 0.5 by default.
 */
export function fbm3(
  x: number,
  y: number,
  z: number,
  seed: number,
  octaves = 6,
  lacunarity = 2,
  gain = 0.5,
): number {
  let amp = 1;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += amp * valueNoise3(x * freq, y * freq, z * freq, seed + o * 1013);
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum / norm * 2 - 1; // center to [-1, 1]
}

/** Ridged multifractal (0 at valley floors, sharp 1 at crests), [0, 1]. */
export function ridged3(
  x: number,
  y: number,
  z: number,
  seed: number,
  octaves = 6,
  lacunarity = 2,
  gain = 0.5,
): number {
  let amp = 1;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    const n = 1 - Math.abs(valueNoise3(x * freq, y * freq, z * freq, seed + o * 1013) * 2 - 1);
    sum += amp * n * n;
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum / norm;
}

/** Cheap deterministic scalar hash on the unit sphere direction, [0,1]. */
export function dirHash(dir: { x: number; y: number; z: number }, seed: number): number {
  // quantize the direction finely (1e-7 rad ~ 0.6 m at Earth radius)
  return hash3(
    Math.round(dir.x * 1e7),
    Math.round(dir.y * 1e7),
    Math.round(dir.z * 1e7),
    seed,
  );
}

/**
 * Public integer-lattice hash in [0,1) — the raw cell hash used by
 * valueNoise3. Exposed for procedural scattering that addresses lattice
 * cells directly (moon craters, future vegetation/rock placement).
 */
export function hash3i(x: number, y: number, z: number, seed: number): number {
  return hash3(x, y, z, seed);
}
