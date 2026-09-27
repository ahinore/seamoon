import * as THREE from 'three';
import { terrainHeight } from './terrain';
import { moonHeight } from './moon';
import type { BodySurface } from './tileGeometry';

/**
 * Coordinate-frame hierarchy (Phase 9 M9.6 — strategy note "宇宙の描画と座標系").
 *
 * 太陽系座標 → 天体中心座標 → ローカル座標, with the invariant that switching
 * the reference body never costs precision:
 *
 * 1. HELIOCENTRIC-ROOT (absolute): ONE double-precision inertial frame for
 *    the whole simulation. Earth sits at (0,0,0); the moon's orbit is defined
 *    in this same frame (moonOrbit.ts). The Sun is directional only (a fixed
 *    sunDir — it never translates), so a separate solar-system origin would
 *    be dead structure; this absolute layer IS the solar-system frame.
 *
 * 2. BODY-CENTRIC (BodyFrame): per-body center (absolute double), radius, GM
 *    and surface height. Physics primaries (FlightModel primC/moonC), tile
 *    placement (PlanetView.bodyCenter) and body-reference selection
 *    (nearestFrame) all resolve through this registry, so a reference switch
 *    is a pointer change. Body-center magnitudes (~3.8e8 m for the moon) are
 *    harmless in doubles (ulp ~4e-8 m); the ONLY float32 quantization happens
 *    at the floating-origin handoff near the camera (world.ts), which is
 *    re-centered every frame — never relative to a body 384 Mm away.
 *
 * 3. LOCAL (spherical): one lat/lon/alt + east convention for BOTH bodies,
 *    shared by spawn placement and test instrumentation. Before this file the
 *    same spherical math lived duplicated in testAuto's earth and moon
 *    branches; one convention means ?lat/?lon/?pitch/?hdg mean the same thing
 *    on every body.
 */

export interface BodyFrame {
  readonly name: 'earth' | 'moon';
  /** Sphere radius (sea level for Earth), m. */
  readonly radius: number;
  /** GM, m^3/s^2. */
  readonly mu: number;
  /** Surface height above the sphere at a unit direction, m. */
  readonly height: BodySurface['height'];
  /**
   * Body center in ABSOLUTE coordinates. Earth: (0,0,0) forever. The moon's
   * is written every frame from moonOrbit — single writer is the main loop
   * (testAuto's constructor writes it before the loop starts); readers copy
   * or read directly, never re-derive.
   */
  readonly center: THREE.Vector3;
}

export const EARTH: BodyFrame = {
  name: 'earth',
  radius: 6_371_000,
  mu: 3.986004418e14,
  height: terrainHeight,
  center: new THREE.Vector3(0, 0, 0),
};

export const MOON: BodyFrame = {
  name: 'moon',
  radius: 1_737_000,
  mu: 4.9048e12,
  height: moonHeight,
  center: new THREE.Vector3(),
};

const DEG = Math.PI / 180;
const clamp = (x: number, a: number, b: number): number => Math.min(Math.max(x, a), b);

/**
 * THE nearest-body rule (was inline in main.ts): surface distance wins.
 * Hysteresis is not needed — the two surfaces are 384 Mm apart, so the
 * boundary region is a mathematical point, not a chatter band.
 */
export function nearestFrame(absPos: THREE.Vector3): 'earth' | 'moon' {
  const dEarth = absPos.distanceTo(EARTH.center) - EARTH.radius;
  const dMoon = absPos.distanceTo(MOON.center) - MOON.radius;
  return dMoon < dEarth ? 'moon' : 'earth';
}

export interface LocalPos {
  latDeg: number;
  lonDeg: number;
  /** Altitude above the SPHERE (not terrain), m. */
  alt: number;
}

const _r = new THREE.Vector3();

/**
 * Spherical → absolute position (alt above the sphere; callers that need
 * terrain-relative altitude add frame.height themselves, e.g. the lander
 * spawn). Same formula testAuto inlined per body, now shared.
 */
export function localToAbsolute(
  f: BodyFrame,
  latDeg: number,
  lonDeg: number,
  alt: number,
  out: THREE.Vector3,
): THREE.Vector3 {
  const lat = latDeg * DEG;
  const lon = lonDeg * DEG;
  return out
    .set(Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon))
    .multiplyScalar(f.radius + alt)
    .add(f.center);
}

/** Absolute → spherical (lat/lon deg, alt above the sphere). */
export function absoluteToLocal(f: BodyFrame, absP: THREE.Vector3, out: LocalPos): LocalPos {
  _r.copy(absP).sub(f.center);
  const n = _r.length();
  out.latDeg = Math.asin(clamp(_r.y / n, -1, 1)) / DEG;
  out.lonDeg = Math.atan2(_r.z, _r.x) / DEG;
  out.alt = n - f.radius;
  return out;
}

/**
 * Local east at (lat,lon) — the VIEW-convention vector the test poses were
 * calibrated against (testAuto earth/moon branches, M9.4 sun alignments).
 * It equals the true ∂pos/∂lon east only at the equator; kept EXACTLY as-is
 * for parity with the verified pose URLs. Normalized.
 */
export function eastAt(latDeg: number, lonDeg: number, out: THREE.Vector3): THREE.Vector3 {
  const lat = latDeg * DEG;
  const lon = lonDeg * DEG;
  return out.set(-Math.sin(lon), 0, Math.cos(lat) * Math.cos(lon)).normalize();
}
