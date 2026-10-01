import * as THREE from 'three';

/**
 * Moon orbit (Phase 9): deterministic circular orbit, Earth-centered.
 *
 * Radius and period are the real values (384.4 Mm, 27.32 days) so distances
 * and phases feel right; a future commit can swap this for real orbital
 * elements + date without touching callers (single evaluation point).
 *
 * The orbit plane is tilted 5.14 deg from the ecliptic like the real moon —
 * here approximated as a tilt around the X axis of a base circular path in
 * the XZ plane.
 */
export const MOON_ORBIT_R = 384.4e6; // m, semi-major axis (circular approx)
const MOON_PERIOD_S = 27.32 * 24 * 3600; // sidereal month
/** Orbit plane tilt from the XZ plane, rad (M11j: the moonshot ascent
 * programs its pitch program INTO this plane so the TLI geometry and the
 * dynamically-placed moon stay exactly coplanar). */
export const INCLINATION = 5.14 * (Math.PI / 180);

/** Angular rate of the circular orbit, rad/s. */
const ANG_RATE = (Math.PI * 2) / MOON_PERIOD_S;

/**
 * Moon position (absolute) at orbit angle `a` radians.
 * Phase 0 places the moon along +X at a=0.
 */
export function moonPositionAtAngle(a: number, out: THREE.Vector3): THREE.Vector3 {
  // base circle in XZ, inclined around X: y = sin(a)*sin(i)
  const ca = Math.cos(a), sa = Math.sin(a);
  const ci = Math.cos(INCLINATION);
  const si = Math.sin(INCLINATION);
  return out.set(MOON_ORBIT_R * ca, MOON_ORBIT_R * sa * si, MOON_ORBIT_R * sa * ci);
}

/** Moon position (absolute) at simulation time `t` seconds. */
export function moonPosition(t: number, out: THREE.Vector3): THREE.Vector3 {
  return moonPositionAtAngle(t * ANG_RATE, out);
}

const _mp = new THREE.Vector3();

/**
 * Moon center from the test params used elsewhere (?moonangle=<deg>):
 * defaults to the live clock position. The FlightModel's lunar lander needs
 * the same orbit position the renderer uses, resolved independently.
 */
export function moonCenterFromParams(q: URLSearchParams, tSimSec: number): THREE.Vector3 {
  if (q.has('moonangle')) {
    return moonPositionAtAngle(num2(q, 'moonangle', 0) * DEG2, _mp).clone();
  }
  return moonPosition(tSimSec, _mp).clone();
}

const DEG2 = Math.PI / 180;
const num2 = (q: URLSearchParams, k: string, d: number): number => {
  const v = q.get(k);
  if (v === null) return d;
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};
