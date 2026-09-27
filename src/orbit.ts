import * as THREE from 'three';

/**
 * Patched-conic orbit mechanics (Phase 9 M9.5).
 *
 * Coasting flight propagates on the Kepler rail via universal variables
 * (Stumpff functions): singularity-free across elliptic / parabolic /
 * hyperbolic orbits, and EXACT per call — one full orbit is a single
 * analytic evaluation rather than thousands of Euler steps, so time warp
 * costs nothing in accuracy. Thrust phases integrate numerically in the
 * caller (velocity Verlet), which is the standard hybrid from the strategy
 * note (rail when inertial, numeric when burning).
 *
 * All state is ABSOLUTE double positions/velocities about the current
 * primary body's center. Handing off between bodies (SOI switch) keeps the
 * inertial state untouched — with the moon frozen there is no rotating
 * frame to rebase, so the handoff is just "whose gravity do we evaluate".
 */

/** Moon's SOI radius about Earth: a·(mu_moon/mu_earth)^(2/5) ≈ 66.2 Mm. */
export const SOI_MOON =
  384.4e6 * Math.pow(4.9048e12 / 3.986004418e14, 0.4);

/** Stumpff c2(psi). Taylor branch covers the parabolic neighborhood. */
export function stumpC(psi: number): number {
  if (psi > 1e-6) {
    const s = Math.sqrt(psi);
    return (1 - Math.cos(s)) / psi;
  }
  if (psi < -1e-6) {
    const s = Math.sqrt(-psi);
    return (Math.cosh(s) - 1) / -psi;
  }
  return 0.5 - psi / 24 + (psi * psi) / 720;
}

/** Stumpff s(psi) (a.k.a. c3). */
export function stumpS(psi: number): number {
  if (psi > 1e-6) {
    const s = Math.sqrt(psi);
    return (s - Math.sin(s)) / (s * s * s);
  }
  if (psi < -1e-6) {
    const s = Math.sqrt(-psi);
    return (Math.sinh(s) - s) / (s * s * s);
  }
  return 1 / 6 - psi / 120 + (psi * psi) / 5040;
}

const clampStep = (step: number, lim: number): number =>
  Math.abs(step) > lim ? Math.sign(step) * lim : step;

/**
 * Kepler rail propagation (universal variables, Curtis Alg. 3.4): advance
 * state (r0, v0) about a body of GM mu by dt seconds, writing the new state
 * into outR/outV. Newton iteration on the universal anomaly chi, damped for
 * the first steps so wild initial guesses can not overshoot.
 */
export function propagateKepler(
  r0: THREE.Vector3,
  v0: THREE.Vector3,
  mu: number,
  dt: number,
  outR: THREE.Vector3,
  outV: THREE.Vector3,
): void {
  const r0m = r0.length();
  const rdotv = r0.dot(v0);
  const sqrtMu = Math.sqrt(mu);
  const alpha = 2 / r0m - v0.lengthSq() / mu; // 1/a (0 = parabola)

  // initial guess for the universal anomaly chi
  let chi: number;
  if (alpha > 1e-11) {
    chi = sqrtMu * dt * alpha; // elliptic
  } else if (alpha >= -1e-11) {
    chi = (sqrtMu * dt) / r0m; // near-parabolic
  } else {
    const a = 1 / alpha; // negative
    const arg =
      (-2 * mu * alpha * dt) /
      (rdotv + Math.sign(dt) * Math.sqrt(-mu * a) * (1 - r0m * alpha));
    chi = Math.sign(dt) * Math.sqrt(-a) * Math.log(arg > 1 ? arg : 1.0001);
  }

  let c2 = 0.5;
  let c3 = 1 / 6;
  let psi = 0;
  let r = r0m;
  for (let i = 0; i < 64; i++) {
    psi = chi * chi * alpha;
    c2 = stumpC(psi);
    c3 = stumpS(psi);
    r =
      chi * chi * c2 +
      (rdotv / sqrtMu) * chi * (1 - psi * c3) +
      r0m * (1 - psi * c2);
    // time-to-chi function F(chi) = 0 ; dF/dchi = r
    const F =
      (rdotv / sqrtMu) * chi * chi * c2 +
      (1 - alpha * r0m) * chi * chi * chi * c3 +
      r0m * chi -
      sqrtMu * dt;
    const step = F / r;
    chi -= clampStep(step, Math.abs(chi) * 0.5 + 1);
    if (Math.abs(step) < 1e-10 * (Math.abs(chi) + 1)) break;
  }

  // Lagrange f/g coefficients -> new state
  const f = 1 - (chi * chi * c2) / r0m;
  const g = dt - (chi * chi * chi * c3) / sqrtMu;
  const gdot = 1 - (chi * chi * c2) / r;
  const fdot = (sqrtMu / (r * r0m)) * chi * (psi * c3 - 1);
  outR.copy(r0).multiplyScalar(f).addScaledVector(v0, g);
  outV.copy(r0).multiplyScalar(fdot).addScaledVector(v0, gdot);
}

export type OrbitalElements = {
  /** semi-major axis, m (negative for hyperbolic) */
  a: number;
  e: number;
  /** period, s (0 if not elliptic) */
  period: number;
  /** periapsis radius, m (from body center) */
  rp: number;
  /** apoapsis radius, m (Infinity if not elliptic) */
  ra: number;
};

const _h = new THREE.Vector3();
const _evec = new THREE.Vector3();

/** Osculating elements from state vectors (Curtis Alg 4.1, reduced). */
export function elementsOf(
  r: THREE.Vector3,
  v: THREE.Vector3,
  mu: number,
  out?: OrbitalElements,
): OrbitalElements {
  const rm = r.length();
  _h.crossVectors(r, v);
  // e vector = (v x h)/mu - r_hat
  _evec.crossVectors(v, _h).multiplyScalar(1 / mu);
  const invRm = 1 / rm;
  _evec.addScaledVector(r, -invRm);
  const e = _evec.length();
  const a = 1 / (2 / rm - v.lengthSq() / mu);
  const rp = a * (1 - e);
  const ra = e < 1 && a > 0 ? a * (1 + e) : Infinity;
  const period = e < 1 && a > 0 ? 2 * Math.PI * Math.sqrt((a * a * a) / mu) : 0;
  const el = out ?? { a: 0, e: 0, period: 0, rp: 0, ra: 0 };
  el.a = a;
  el.e = e;
  el.period = period;
  el.rp = rp;
  el.ra = ra;
  return el;
}
