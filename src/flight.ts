import * as THREE from 'three';
import { terrainHeight } from './terrain';
import type { CameraRig } from './cameraRig';
import type { WorldOrigin } from './world';

const DEG = Math.PI / 180;
const R = 6_371_000; // planet radius (sea level), m
const MU = 3.986004418e14; // GM, m^3/s^2 — gravity = MU/r^2 (g0 = 9.82 at R)

// --- aircraft parameters (light single, arcade-tuned) ---
const MASS = 1200; // kg
const WING_S = 16; // m^2
const CL0 = 0.3; // camber lift at zero AoA
const CL_A = 5.0; // lift-curve slope per radian
const AR = 7.5; // aspect ratio
const OE = 0.8; // Oswald efficiency
const CD0 = 0.028; // parasite drag
const CY_B = 1.0; // sideslip side-force coefficient
const MAX_THRUST = 3200; // N, sea level static (T/W ~0.27, light single)
const GEAR_H = 1.7; // landing-gear height, m
const MAX_PITCH_RATE = 45 * DEG; // rad/s at full deflection, full effectiveness
const MAX_ROLL_RATE = 120 * DEG;
const MAX_YAW_RATE = 20 * DEG;
const Q_REF = 0.5 * 1.225 * 100 * 100; // dynamic pressure at 100 m/s SL

const clamp = (x: number, a: number, b: number) => Math.min(Math.max(x, a), b);

/**
 * US Standard Atmosphere 1976 density (kg/m^3), layers 0-32 km plus an
 * exponential halo above (density never quite reaches zero — high-altitude
 * flight keeps weak control authority instead of NaNs).
 */
export function isaDensity(alt: number): number {
  if (alt > 32000) {
    return isaDensity(32000) * Math.exp(-(alt - 32000) / 7000);
  }
  if (alt < 11000) {
    const T = 288.15 - 0.0065 * alt;
    const p = 101325 * Math.pow(T / 288.15, 5.25588);
    return p / (287.05287 * T);
  }
  if (alt < 20000) {
    const p = 22632.1 * Math.exp(-(alt - 11000) / 6341.62);
    return p / (287.05287 * 216.65);
  }
  const T = 216.65 + 0.001 * (alt - 20000);
  const p = 5474.89 * Math.pow(T / 216.65, -34.1632);
  return p / (287.05287 * T);
}

export type FlightStatus = {
  ias: number; // indicated airspeed m/s
  gs: number; // ground speed m/s
  agl: number; // above ground level m
  vs: number; // vertical speed m/s (radial)
  aoa: number; // angle of attack deg
  thr: number; // throttle 0..1
  phase: string; // TAKEOFF/CLIMB/... or MANUAL
  note: string; // LANDED/CRASHED/... or ''
};

/**
 * Phase 6: 6-DoF flight model inside the atmosphere.
 *
 * State lives in ABSOLUTE coordinates (double): position, velocity, and a
 * body-to-world quaternion. The camera rig provides control inputs (spring-
 * centered stick deflections from mouse/keys); the aircraft pose is written
 * back to the camera each step through WorldOrigin, so the floating origin,
 * LOD, and atmosphere all keep working unchanged.
 *
 * Ground truth for collision is terrainHeight() itself — the very function
 * the tile meshes displace with — so wheels can never sink into the visual
 * ground regardless of LOD level.
 *
 * Controls (flight mode): mouse/arrows = stick (up=pitch up, right=roll
 * right), W/S throttle, A/D rudder (also nose-wheel steering on the ground),
 * R resets to the spawn point, F exits back to the free camera.
 *
 * ?demo=fly runs a scripted mission proving the strategy note's completion
 * criteria: takeoff -> climb -> 90° turn -> approach -> flare -> landing.
 */
export class FlightModel {
  mode: 'manual' | 'fly' = 'manual';
  frozen = false;

  private readonly rig: CameraRig;
  private readonly world: WorldOrigin;

  // absolute state (double precision)
  private readonly pos = new THREE.Vector3();
  private readonly vel = new THREE.Vector3();
  private readonly q = new THREE.Quaternion();
  private thr = 0;

  // spawn point (absolute, unit direction + ground elevation + heading)
  private readonly spawnDir = new THREE.Vector3(0, 0, 1);
  private spawnH = 0;
  private spawnHdg = 90;

  // autopilot mission state
  private apPhase = 'off';
  private apT = 0;
  private cruiseT = 0;
  private h0 = 0;

  // cached per-step telemetry for the HUD
  private tIas = 0;
  private tGs = 0;
  private tAgl = 0;
  private tVs = 0;
  private tAoa = 0;
  private rawAoa = 0;
  private rawBeta = 0;
  private note = '';
  private lastNear = -1;

  private readonly _fwd = new THREE.Vector3();
  private readonly _right = new THREE.Vector3();
  private readonly _upB = new THREE.Vector3();
  private readonly _up = new THREE.Vector3();
  private readonly _east = new THREE.Vector3();
  private readonly _north = new THREE.Vector3();
  private readonly _tmp = new THREE.Vector3();
  private readonly _tmp2 = new THREE.Vector3();
  private readonly _vb = new THREE.Vector3();
  private readonly _qinv = new THREE.Quaternion();
  private readonly _dq = new THREE.Quaternion();
  private readonly _basis = new THREE.Matrix4();
  private readonly _la = { pitch: 0, bank: 0, heading: 0 };
  private readonly WORLD_Y = new THREE.Vector3(0, 1, 0);

  constructor(rig: CameraRig, world: WorldOrigin) {
    this.rig = rig;
    this.world = world;
    const q = new URLSearchParams(location.search);
    if (q.get('demo') === 'fly') {
      this.mode = 'fly';
      this.apPhase = 'takeoff';
    }
    this.spawnHdg = num(q, 'hdg', 90);
    const lat = num(q, 'lat', 5.5);
    const lon = num(q, 'lon', -104);
    this.findSpawn(lat, lon);
    this.reset();
  }

  /** Body-frame airflow angles for HUD debugging (AoA deg, sideslip deg). */
  airflowDebug(): { aoa: number; beta: number; ias: number } {
    this._qinv.copy(this.q).invert();
    const vb = this._vb.copy(this.vel).applyQuaternion(this._qinv);
    const up = this._up.copy(this.pos).normalize();
    const rho = isaDensity(Math.max(this.pos.length() - R, 0));
    let aoa = Math.atan2(-vb.y, -vb.z);
    let beta = Math.atan2(-vb.x, -vb.z);
    aoa = clamp(aoa, -0.5, 0.5);
    beta = clamp(beta, -0.6, 0.6);
    return { aoa: aoa / DEG, beta: beta / DEG, ias: this.vel.length() * Math.sqrt(rho / 1.225) };
  }

  /**
   * Grid-search the flattest land cell around (lat, lon) for the airfield:
   * terrainHeight is cheap (µs), so sampling ~900 cells + slope probes is
   * fine at spawn time. Requires h > 20 m (dry land, clear of the beach
   * band) and prefers low-elevation plains.
   */
  private findSpawn(latDeg: number, lonDeg: number): void {
    const lat0 = latDeg * DEG;
    const lon0 = lonDeg * DEG;
    let bestScore = Infinity;
    let bestH = 0;
    const dir = new THREE.Vector3();
    for (let i = -6; i <= 6; i++) {
      for (let j = -6; j <= 6; j++) {
        const lat = lat0 + i * 0.0015;
        const lon = lon0 + (j * 0.0015) / Math.max(Math.cos(lat0), 0.2);
        dir
          .set(Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon))
          .normalize();
        const h = terrainHeight(dir.x, dir.y, dir.z);
        if (h < 20) continue;
        // slope probes ~170 m east/north
        const d = 170 / R;
        const east = _sEast.set(-Math.sin(lon), 0, Math.cos(lat) * Math.cos(lon)).normalize();
        const north = _sNorth.crossVectors(dir, east).normalize();
        const he = terrainHeight(
          _sTmp.copy(dir).multiplyScalar(Math.cos(d)).addScaledVector(east, Math.sin(d)).x,
          _sTmp.y,
          _sTmp.z,
        );
        const hn = terrainHeight(
          _sTmp.copy(dir).multiplyScalar(Math.cos(d)).addScaledVector(north, Math.sin(d)).x,
          _sTmp.y,
          _sTmp.z,
        );
        const score = Math.abs(he - h) + Math.abs(hn - h) + Math.abs(h - 80) * 0.002;
        if (score < bestScore) {
          bestScore = score;
          bestH = h;
          this.spawnDir.copy(dir);
        }
      }
    }
    if (bestScore === Infinity) {
      // all water: fall back to the requested point at sea level (will ditch)
      this.spawnDir
        .set(Math.cos(lat0) * Math.cos(lon0), Math.sin(lat0), Math.cos(lat0) * Math.sin(lon0))
        .normalize();
      this.spawnH = 0;
    } else {
      this.spawnH = bestH;
    }
  }

  /** Respawn: gear on the ground, zero velocity, aligned with the horizon. */
  reset(): void {
    this.pos.copy(this.spawnDir).multiplyScalar(R + this.spawnH + GEAR_H);
    this.vel.set(0, 0, 0);
    this.thr = 0;
    this.frozen = false;
    this.note = '';
    this.apT = 0;
    this.cruiseT = 0;
    this.apPhase = this.mode === 'fly' ? 'takeoff' : 'off';
    const up = this._up.copy(this.spawnDir);
    const east = this._east.set(-Math.sin(0), 0, 1); // placeholder, replaced below
    // local east/north at the spawn point
    const lon = Math.atan2(this.spawnDir.z, this.spawnDir.x);
    const lat = Math.asin(clamp(this.spawnDir.y, -1, 1));
    east
      .set(-Math.sin(lon), 0, Math.cos(lat) * Math.cos(lon))
      .normalize();
    this._north.crossVectors(up, east).normalize().negate(); // up×east = -north
    const fwd = this._tmp
      .copy(this._north)
      .multiplyScalar(Math.cos(this.spawnHdg * DEG))
      .addScaledVector(east, Math.sin(this.spawnHdg * DEG))
      .normalize();
    const rightH = this._tmp2.crossVectors(fwd, up).normalize();
    const upH = _sUp.crossVectors(rightH, fwd).normalize();
    this._basis.makeBasis(rightH, upH, _sBack.copy(fwd).negate());
    this.q.setFromRotationMatrix(this._basis);
    this.writeCamera();
  }

  /** One physics step; writes the pose back to the rig camera. */
  step(dt: number): void {
    if (this.frozen) {
      this.writeCamera();
      return;
    }
    if (this.mode === 'fly') this.autopilot(dt);
    this.integrate(dt);
    this.writeCamera();
  }

  // ------------------------------------------------------------------ physics

  private integrate(dt: number): void {
    const up = this._up.copy(this.pos).normalize();
    const n = this.pos.length();
    const hGnd = terrainHeight(up.x, up.y, up.z);
    const agl = n - R - hGnd;
    const altASL = n - R;
    const rho = isaDensity(Math.max(altASL, 0));

    // body axes from attitude
    const q = this.q;
    const fwd = this._fwd.set(0, 0, -1).applyQuaternion(q);
    const right = this._right.set(1, 0, 0).applyQuaternion(q);
    const upB = this._upB.set(0, 1, 0).applyQuaternion(q);

    // controls: stick deflections from the rig, throttle from W/S
    const c = this.rig.ctl;
    const thrIn = (this.rig.isDown('KeyW') ? 0.6 : 0) - (this.rig.isDown('KeyS') ? 0.6 : 0);
    this.thr = clamp(this.thr + thrIn * dt, 0, 1);

    const V = this.vel.length();
    const qbar = 0.5 * rho * V * V;
    const eff = clamp(qbar / Q_REF, 0, 1.3); // control authority grows with q

    // airflow in body frame -> AoA / sideslip
    this._qinv.copy(q).invert();
    const vb = this._vb.copy(this.vel).applyQuaternion(this._qinv);
    let aoa = Math.atan2(-vb.y, -vb.z);
    let beta = Math.atan2(-vb.x, -vb.z);
    this.rawAoa = aoa / DEG;
    this.rawBeta = beta / DEG;
    aoa = clamp(aoa, -0.5, 0.5);
    beta = clamp(beta, -0.6, 0.6);

    // lift/drag coefficients with a simple stall break at ~15 deg
    const s = smoothstep(15 * DEG, 25 * DEG, Math.abs(aoa));
    const cl = (CL0 + CL_A * aoa) * (1 - s) + Math.sign(aoa) * 0.5 * s;
    const cd = CD0 + (cl * cl) / (Math.PI * AR * OE) + s * 0.8;

    // forces: gravity, thrust, lift, drag, side force
    const acc = this._tmp.set(0, 0, 0);
    acc.addScaledVector(up, -MU / (n * n)); // gravity
    acc.addScaledVector(fwd, (this.thr * MAX_THRUST * Math.pow(rho / 1.225, 0.8)) / MASS);
    const vhat = this._tmp2.copy(this.vel).multiplyScalar(1 / Math.max(V, 0.1));
    // lift direction: body-up component perpendicular to the velocity
    const liftDir = _sLift.copy(upB).addScaledVector(vhat, -upB.dot(vhat));
    if (liftDir.lengthSq() < 0.01) liftDir.copy(upB); // degenerate: straight up
    liftDir.normalize();
    acc.addScaledVector(liftDir, (qbar * WING_S * cl) / MASS);
    acc.addScaledVector(vhat, (-qbar * WING_S * cd) / MASS);
    acc.addScaledVector(right, (qbar * WING_S * CY_B * beta) / MASS);

    this.vel.addScaledVector(acc, dt);
    this.pos.addScaledVector(this.vel, dt);

    // rotation: rate command scaled by dynamic pressure
    let wx = c.pitch * MAX_PITCH_RATE * eff; // + = nose up
    let wy = -c.yaw * MAX_YAW_RATE * eff; // + about +Y = yaw LEFT, so negate
    let wz = -c.roll * MAX_ROLL_RATE * eff; // + about +Z = roll LEFT, so negate
    wy += beta * 2.0 * eff; // weathervane: nose chases the relative wind
    const bank = this.bankOf(q, up);
    const pitch = Math.asin(clamp(fwd.dot(up), -1, 1));

    // ground contact (gear on the terrain function itself).
    // Only true contact settles/brakes: while VS is positive (climbing away)
    // or VS is small-negative at speed (gear kissing the runway in rotation)
    // the ground reaction is limited so lift can pull the aircraft up.
    if (agl < GEAR_H) {
      const vs = this.vel.dot(up);
      if (vs < -12 || Math.abs(bank) > 25 * DEG || pitch < -30 * DEG || hGnd < 0) {
        this.frozen = true;
        this.note = hGnd < 0 ? 'DITCHED' : 'CRASHED';
        return;
      }
      if (vs <= 0.1) {
        // touchdown / rolling: snap to gear height, kill vertical velocity,
        // apply rolling friction + brakes
        this.pos.copy(up).multiplyScalar(R + Math.max(hGnd, 0) + GEAR_H);
        this.vel.addScaledVector(up, -vs);
        const vt = this._tmp2.copy(this.vel).addScaledVector(up, -this.vel.dot(up));
        const vtLen = vt.length();
        const brake = this.thr < 0.05 && this.rig.isDown('KeyS') ? 3.0 : 0.25;
        if (vtLen > 1e-3) {
          const dec = Math.min(vtLen, brake * dt);
          this.vel.addScaledVector(vt.normalize(), -dec);
        }
        if (vtLen < 0.4 && this.thr < 0.03) this.vel.set(0, 0, 0);
        // settle the attitude onto the gear (level, heading preserved)
        const fh = _sLift.copy(fwd).addScaledVector(up, -fwd.dot(up));
        if (fh.lengthSq() > 1e-4) {
          fh.normalize();
          const rightH = _sEast.crossVectors(fh, up).normalize();
          const upH = _sNorth.crossVectors(rightH, fh).normalize();
          this._basis.makeBasis(rightH, upH, _sBack.copy(fh).negate());
          _sLevelQ.setFromRotationMatrix(this._basis);
          q.slerp(_sLevelQ, Math.min(1, dt * 4));
        }
        // nose-wheel steering at low speed
        const steer = (1 - clamp(vtLen / 30, 0, 1)) * 30 * DEG;
        wy += -c.yaw * steer;
        // landing outcome for the scripted mission
        if (this.mode === 'fly' && (this.apPhase === 'approach' || this.apPhase === 'flare')) {
          this.apPhase = 'off';
          this.note = 'LANDED';
          this.thr = 0;
        }
      }
    }

    // integrate attitude (body-frame small-angle quaternion, post-multiplied)
    const h = dt;
    this._dq.set((wx * h) / 2, (wy * h) / 2, (wz * h) / 2, 1).normalize();
    q.multiply(this._dq).normalize();

    // telemetry for the HUD
    this.tGs = this.vel.length();
    this.tIas = this.tGs * Math.sqrt(rho / 1.225);
    this.tAgl = Math.max(agl, 0);
    this.tVs = this.vel.dot(up);
    this.tAoa = aoa / DEG;
  }

  // ---------------------------------------------------------------- autopilot

  /**
   * Scripted mission (?demo=fly): takeoff roll, rotate, climb to 1800 m AGL,
   * banked 90° turn, short cruise, throttled approach, flare, landing.
   * Writes stick deflections + throttle; physics stays the same code path.
   */
  private autopilot(dt: number): void {
    this.apT += dt;
    const c = this.rig.ctl;
    const la = this.lookAngles();
    const vs = this.tVs;
    const agl = this.tAgl;
    const ias = this.tIas;
    const aoa = this.rawAoa; // unclamped — the real angle of attack
    // AoA protection: past ~11° push the nose down instead of chasing VS,
    // otherwise the VS feedback loop drives the aircraft into a stall.
    const stallGuard = aoa > 11 ? -clamp((aoa - 11) * 0.25, 0, 0.5) : 0;

    switch (this.apPhase) {
      case 'takeoff':
        this.thr = 1;
        c.pitch = 0;
        c.roll = 0;
        c.yaw = 0;
        if (ias > 75) this.apPhase = 'rotate';
        break;
      case 'rotate': {
        // hold ~10° body pitch (aircraft frame) and wait for lift
        this.thr = 1;
        c.pitch = clamp((10 - la.pitch) * 0.1, -0.2, 0.5) + stallGuard;
        c.roll = 0;
        c.yaw = 0;
        if (agl > 60) this.apPhase = 'climb';
        break;
      }
      case 'climb':
        // hold ~12° pitch until cruise altitude
        this.thr = 1;
        c.pitch = clamp((12 - la.pitch) * 0.1, -0.4, 0.5) + stallGuard;
        c.roll = clamp((0 - la.bank) * 0.06, -1, 1);
        c.yaw = 0;
        if (agl > 1800) {
          this.apPhase = 'turn';
          this.h0 = la.heading;
        }
        break;
      case 'turn': {
        this.thr = 0.9;
        c.pitch = clamp((10 - la.pitch) * 0.1, -0.4, 0.5) + stallGuard;
        c.roll = clamp((25 - la.bank) * 0.12, -1, 1);
        c.yaw = 0;
        const dh = ((la.heading - this.h0 + 540) % 360) - 180;
        if (Math.abs(dh) >= 88) {
          this.apPhase = 'cruise';
          this.cruiseT = this.apT;
        }
        break;
      }
      case 'cruise':
        // level off: hold the horizon
        this.thr = 0.6;
        c.pitch = clamp((0 - la.pitch) * 0.08, -0.4, 0.4) + stallGuard;
        c.roll = clamp((0 - la.bank) * 0.12, -1, 1);
        c.yaw = 0;
        if (this.apT - this.cruiseT > 6) this.apPhase = 'approach';
        break;
      case 'approach':
        // ~-6° descent pitch, slow
        this.thr = 0.12;
        c.pitch = clamp((-6 - la.pitch) * 0.08, -0.4, 0.35) + stallGuard;
        c.roll = clamp((0 - la.bank) * 0.12, -1, 1);
        c.yaw = 0;
        if (agl < 60) this.apPhase = 'flare';
        break;
      case 'flare':
        this.thr = 0;
        c.pitch = clamp((-1 - la.pitch) * 0.06, -0.4, 0.5) + stallGuard;
        c.roll = clamp((0 - la.bank) * 0.12, -1, 1);
        c.yaw = 0;
        break;
      default:
        // mission over (LANDED/CRASHED): hands off, brakes on
        this.thr = 0;
        c.pitch = 0;
        c.roll = 0;
        c.yaw = 0;
        break;
    }
  }

  // ------------------------------------------------------------------ helpers

  /** Camera pose + near plane from the absolute state (frame-relative write). */
  private writeCamera(): void {
    this.world.rel(this.pos, this._tmp);
    this.rig.camera.position.copy(this._tmp);
    this.rig.camera.quaternion.copy(this.q);
    const near = clamp(this.tAgl * 0.25 + 0.3, 0.3, 5e4);
    if (Math.abs(near - this.lastNear) / near > 0.3 || this.lastNear < 0) {
      this.lastNear = near;
      this.rig.camera.near = near;
      this.rig.camera.updateProjectionMatrix();
    }
  }

  /** pitch/bank/heading (deg) of the body attitude vs. the local horizon. */
  private lookAngles(): { pitch: number; bank: number; heading: number } {
    const up = this._up.copy(this.pos).normalize();
    const fwd = this._fwd.set(0, 0, -1).applyQuaternion(this.q);
    this._la.pitch = Math.asin(clamp(fwd.dot(up), -1, 1)) / DEG;
    // heading
    this._east.crossVectors(this.WORLD_Y, up);
    if (this._east.lengthSq() > 1e-8) {
      this._east.normalize();
      this._north.crossVectors(up, this._east);
      let hdg =
        Math.atan2(fwd.dot(this._east), fwd.dot(this._north)) / DEG;
      if (hdg < 0) hdg += 360;
      this._la.heading = hdg;
    }
    this._la.bank = this.bankOf(this.q, up) / DEG;
    return this._la;
  }

  /** Roll vs. the leveled horizon (same math as CameraRig.getLookAngles). */
  private bankOf(q: THREE.Quaternion, up: THREE.Vector3): number {
    const fwd = this._fwd.set(0, 0, -1).applyQuaternion(q);
    const camUp = _sCamUp.set(0, 1, 0).applyQuaternion(q);
    const rightH = _sEast.crossVectors(fwd, up);
    if (rightH.lengthSq() < 1e-8) return 0;
    rightH.normalize();
    const upH = _sNorth.crossVectors(rightH, fwd).normalize();
    const rollAxis = _sRoll.crossVectors(upH, camUp);
    return Math.atan2(rollAxis.dot(fwd), upH.dot(camUp));
  }

  getStatus(): FlightStatus {
    const la = this.mode === 'fly' ? this.apPhase.toUpperCase() : 'MANUAL';
    return {
      ias: this.tIas,
      gs: this.tGs,
      agl: this.tAgl,
      vs: this.tVs,
      aoa: this.tAoa,
      thr: this.thr,
      phase: la,
      note: this.frozen ? this.note : this.note,
    };
  }

  statusLine(): string {
    const s = this.getStatus();
    // stationary: airflow angles are degenerate (atan2 of ~0) — show dashes
    const a = s.gs > 2 ? this.airflowDebug() : null;
    return (
      `FLY IAS ${s.ias.toFixed(0)} m/s  GS ${s.gs.toFixed(0)}  AGL ${fmtM(s.agl)}  VS ${s.vs >= 0 ? '+' : ''}${s.vs.toFixed(1)}  ` +
      (a
        ? `AoA ${a.aoa.toFixed(1)}° β${a.beta.toFixed(1)}  `
        : `AoA --  `) +
      `THR ${(s.thr * 100).toFixed(0)}%  ` +
      `[${s.phase}${s.note ? ' ' + s.note : ''}]`
    );
  }
}

// scratch shared with nothing else (module-level temps)
const _sEast = new THREE.Vector3();
const _sNorth = new THREE.Vector3();
const _sTmp = new THREE.Vector3();
const _sUp = new THREE.Vector3();
const _sBack = new THREE.Vector3();
const _sLift = new THREE.Vector3();
const _sCamUp = new THREE.Vector3();
const _sRoll = new THREE.Vector3();
const _sLevelQ = new THREE.Quaternion();

const smoothstep = (e0: number, e1: number, x: number): number => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
};

const num = (q: URLSearchParams, k: string, d: number): number => {
  const v = q.get(k);
  if (v === null) return d;
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};

const fmtM = (m: number): string =>
  m >= 10000 ? (m / 1000).toFixed(1) + 'km' : m.toFixed(0) + 'm';
