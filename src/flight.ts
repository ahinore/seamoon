import * as THREE from 'three';
import { terrainHeight } from './terrain';
import { ATMOSPHERE_TOP as ATMOS_TOP } from './atmosphere';
import { moonHeight } from './moon';
import { moonCenterFromParams, MOON_ORBIT_R } from './moonOrbit';
import { propagateKepler, elementsOf, SOI_MOON, type OrbitalElements } from './orbit';
import { EARTH, MOON, nearestFrame, localToAbsolute, eastAt } from './frames';
import type { CameraRig } from './cameraRig';
import type { WorldOrigin } from './world';

const DEG = Math.PI / 180;
const R = EARTH.radius;
const MU = EARTH.mu;
const R_MOON = MOON.radius;
const MU_MOON = MOON.mu;

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
  mode: 'manual' | 'fly' | 'lunar' | 'orbital' = 'manual';
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

  // --- lunar lander state (mode 'lunar') ---
  /** Moon center (absolute) at mission start; the moon is frozen during the mission. */
  private readonly moonC = new THREE.Vector3();
  /** Lander dry+prop mass, kg. */
  private mLand = 4200;
  /** Descent engine max thrust, N (DSE-class). */
  private readonly THRUST_LANDER = 45000;
  /** Propellant burn rate at full throttle, kg/s. */
  private readonly BURN_RATE = 8;
  /** Propellant remaining, kg. */
  private prop = 2300;
  /** True when wheels/feet are on the surface. */
  private landed = false;

  // --- orbital mode state (mode 'orbital', M9.5) ---
  /** Primary body center (absolute): earth until the moon SOI handoff. */
  private readonly primC = new THREE.Vector3();
  /** Primary GM. */
  private primMu = MU;
  /** True after the SOI handoff to the moon. */
  private inMoonSoi = false;
  /** Osculating elements for the HUD. */
  private el: OrbitalElements = { a: 0, e: 0, period: 0, rp: 0, ra: 0 };
  /** Mission clock, s. */
  private orbT = 0;
  /** Coast time-warp factor (rail is analytic — warp is exact). */
  private coastWarp = 6000;
  /** Target apoapsis radius for the TLI burn, m. */
  private tliRa = 344e6;
  /** ?notli=1: suppress the demo TLI burn (entry-testing orbits). */
  private noTli = false;
  /** ?lob=1: suborbital hop (pad launch → ballistic reentry). */
  private lob = false;
  /** ?lob=1 boost-phase clock, s (MECO at 60). */
  private lobT = 0;
  /** ?lob=1 booster thrust, N (TWR ~9 at ignition). */
  private readonly LOB_THRUST = 900_000;
  /** M10.8c: main parachute staged (earth entry, <9 km and subsonic). */
  private paraOpen = false;
  /** M10.8e: entry rumble 0..1 (peak q this step; decays each frame). */
  private shake = 0;
  /** M10.9: last slice's air density (kg/m^3) for the audio wind layer. */
  airDensity = 0;
  /** M10.9: true while the lob booster is firing (audio layer gate). */
  get boostPhase(): boolean {
    return this.lob && this.apPhase === 'boost';
  }

  // --- M10.8 entry telemetry (orbital mode, earth atmosphere) ---
  /** Normalized plasma/heat glow 0..1 (drives the viewport entry effect). */
  heat = 0;
  /** Stagnation heat flux, W/m^2 (Sutton-Graves). */
  heatFlux = 0;
  /** Deceleration load in g. */
  gLoad = 0;
  /** ?pe= test hook: elliptical spawn with this periapsis (null = circular). */
  private peOverride: number | null = null;

  // autopilot mission state
  private apPhase = 'off';
  private apT = 0;
  private cruiseT = 0;
  private h0 = 0;

  // cached per-step telemetry for the HUD
  private tIas = 0;
  tGs = 0; // public for the audio engine (M10.9); rest stay private
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
    if (q.get('demo') === 'lunar') {
      this.mode = 'lunar';
      this.apPhase = 'descent';
      // Moon frozen at ?moonangle (default 0): the lander needs a static
      // gravity well for the scripted descent. The frozen center ALSO seeds
      // the MOON frame registry (M9.6) so every consumer agrees.
      this.moonC.copy(moonCenterFromParams(q, performance.now() / 1000));
      MOON.center.copy(this.moonC);
      // spawn: 15 km above the surface, small horizontal drift, engine down
      this.spawnDir.set(1, 0, 0);
      this.spawnH = 15000;
      this.spawnHdg = 90;
    }
    if (q.get('demo') === 'orbital') {
      this.mode = 'orbital';
      this.apPhase = 'coast';
      // Circular LEO at ?alt (default 200 km); spawn state (pos/vel) is
      // derived in spawnStateInit's orbital branch. Spawn on the -X side:
      // after one lap the TLI burn happens there, so the raised apoapsis
      // points at the moon (frozen at +X with ?moonangle=0) and the transfer
      // actually enters its SOI. Depot propellant for the TLI burn.
      this.spawnDir.set(-1, 0, 0);
      this.spawnH = num(q, 'alt', 200_000);
      this.prop = 6000;
      this.coastWarp = clamp(num(q, 'warp', 6000), 1, 50_000);
      // ?pe=<m>: spawn directly on an elliptical orbit with this periapsis
      // (M10.8 entry testing — a low pe like 20000 dives the orbit into the
      // atmosphere without waiting for the TLI/moon loop). The spawn stays
      // AT apoapsis (radius R+alt) with circular speed for that radius, so
      // the craft falls toward the pe on the far side.
      this.peOverride = q.has('pe') ? num(q, 'pe', 200_000) : null;
      this.noTli = q.has('notli');
      // ?lob=1 (M10.8c): suborbital hop instead of an orbit — spawn on the
      // pad and lob up (v=2300 m/s at 45°, apogee ~180 km). The reentry
      // speed is ~2.3 km/s instead of orbital 7.8 km/s, so the shield +
      // chute + landing burn can actually bring the craft down intact.
      this.lob = q.has('lob');
      // Transfer apoapsis exactly at the moon's orbital radius: with the
      // moon frozen at +X the transfer ellipse ends right on the moon
      // center, so the craft plunges deep into the SOI — the boundary
      // capture below converts it to a lunar ellipse long before lunar
      // approach matters.
      this.tliRa = MOON_ORBIT_R;
      this.moonC.copy(moonCenterFromParams(q, performance.now() / 1000));
      MOON.center.copy(this.moonC);
      this.primC.set(0, 0, 0);
      this.primMu = MU;
    }
    this.spawnHdg = num(q, 'hdg', 90);
    const lat = num(q, 'lat', 5.5);
    const lon = num(q, 'lon', -104);
    // terrain spawn search is meaningless off-earth (orbital spawns on the
    // rail, lunar on the moon's surface — both set their own state above)
    if (this.mode === 'manual' || this.mode === 'fly') this.findSpawn(lat, lon);
    // Compute the spawn state but do NOT write it to the camera: the free
    // camera / AutoPilot owns the pose until flight mode is actually entered
    // (main.ts calls reset() explicitly at that point).
    this.resetSilent();
  }

  /** reset() without touching the camera pose. */
  private resetSilent(): void {
    this.spawnStateInit();
  }

  /** Position/attitude state at the spawn point (no camera write). */
  private spawnStateInit(): void {
    if (this.mode === 'orbital') {
      // M10.8c ?lob=1: suborbital hop — spawn on the pad, lob at 45°.
      // Reentry at ~2.3 km/s is survivable with the shield+chute stack.
      if (this.lob) {
        // Day-side pad (lon 0, subsolar): the night-side -X spawn would
        // make the whole hop a black screen. Moon-facing -X is only
        // needed for the TLI geometry, not a suborbital hop.
        this.spawnDir.set(1, 0, 0);
        const up = _oUp.copy(this.spawnDir);
        const tan = _oRail.copy(this.spawnDir).cross(this.WORLD_Y).normalize();
        // pad sits on the local terrain so the camera doesn't start
        // inside a mountain
        const padTh = terrainHeight(up.x, up.y, up.z);
        this.pos.copy(up).multiplyScalar(R + Math.max(padTh, 0) + 50);
        this.vel.copy(up).multiplyScalar(2300 * Math.SQRT1_2)
          .addScaledVector(tan, 2300 * Math.SQRT1_2);
        this.q.identity();
        this.thr = 0;
        this.frozen = false;
        this.note = 'LOB';
        this.orbT = 0;
        this.prop = 6000;
        this.inMoonSoi = false;
        this.paraOpen = false;
        this.heat = 0;
        this.heatFlux = 0;
        this.gLoad = 0;
        this.primC.set(0, 0, 0);
        this.primMu = MU;
        this.apPhase = 'boost'; // pad launch: booster burns first
        this.lobT = 0;
        this.vel.set(0, 0, 0); // start at rest on the pad
        elementsOf(_oR.copy(this.pos), this.vel, MU, this.el);
        return;
      }
      // circular LEO at spawnH above the start direction (?pe= makes it an
      // ellipse with that periapsis, apoapsis at the spawn radius)
      this.pos.copy(this.spawnDir).multiplyScalar(R + this.spawnH);
      const vc = Math.sqrt(MU / (R + this.spawnH));
      // prograde tangent: spawnDir x worldY (verified against elementsOf:
      // yields e<1e-6 circular)
      this.vel.copy(this.spawnDir).cross(this.WORLD_Y).normalize()
        .multiplyScalar(this.peOverride !== null
          // pe is ALTITUDE above the surface: rp = R + pe. Vis-viva at
          // apoapsis r=ra: v² = mu(2/ra - 1/a), a = (ra+rp)/2 →
          // v² = mu(2/ra - 2/(ra+rp)) (M10.8: the old code used pe as a
          // center-radius, putting perigee INSIDE the planet).
          ? Math.sqrt(Math.max(MU * (2 / (R + this.spawnH) - 2 / (2 * R + this.spawnH + this.peOverride)), 1))
          : vc);
      this.q.identity();
      this.thr = 0;
      this.frozen = false;
      this.note = '';
      this.orbT = 0;
      this.prop = 6000;
      this.inMoonSoi = false;
      this.paraOpen = false;
      this.heat = 0;
      this.heatFlux = 0;
      this.gLoad = 0;
      this.primC.set(0, 0, 0);
      this.primMu = MU;
      this.apPhase = 'coast';
      // Seed the osculating elements so the coast-warp clamp (TLI exactly on
      // the spawn node) is valid from the very first frame — at high warp a
      // single unclamped step could overshoot the whole parking period.
      elementsOf(_oR.copy(this.pos), this.vel, MU, this.el);
      return;
    }
    if (this.mode === 'lunar') {
      // lander: engine-down upright, feet 15 km above the TERRAIN. Spawn
      // through the shared local-frame helper (M9.6): ?lat/?lon (default the
      // near-side point) on the MOON frame + terrain height on top.
      const q2 = new URLSearchParams(location.search);
      const latL = num(q2, 'lat', 0);
      const lonL = num(q2, 'lon', 0);
      const hSurf = moonHeight(
        Math.cos(latL * DEG) * Math.cos(lonL * DEG),
        Math.sin(latL * DEG),
        Math.cos(latL * DEG) * Math.sin(lonL * DEG),
      );
      this.pos.copy(localToAbsolute(MOON, latL, lonL, hSurf + this.spawnH, this.pos));
      // slight HORIZONTAL drift to null out (east at spawn (1,0,0) is -Z:
      // east = worldY x up = (0,1,0)x(1,0,0) = (0,0,-1))
      this.vel.set(0, 0, -6);
      this.q.identity();
      this.thr = 0;
      this.frozen = false;
      this.note = '';
      this.apT = 0;
      this.prop = 2300;
      this.landed = false;
      this.apPhase = this.mode === 'lunar' ? 'descent' : 'off';
      return;
    }
    this.pos.copy(this.spawnDir).multiplyScalar(R + this.spawnH + GEAR_H);
    this.vel.set(0, 0, 0);
    this.thr = 0;
    this.frozen = false;
    this.note = '';
    this.apT = 0;
    this.cruiseT = 0;
    this.apPhase = this.mode === 'fly' ? 'takeoff' : 'off';
    const up = this._up.copy(this.spawnDir);
    const lon = Math.atan2(this.spawnDir.z, this.spawnDir.x);
    const lat = Math.asin(clamp(this.spawnDir.y, -1, 1));
    const east = eastAt(lat / DEG, lon / DEG, this._east);
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
   * band) and prefers low-elevation plains. (Lunar/orbital spawns skip this.)
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
    this.spawnStateInit();
    this.writeCamera();
  }

  /** One physics step; writes the pose back to the rig camera. */
  step(dt: number): void {
    if (this.frozen) {
      this.writeCamera();
      return;
    }
    if (this.mode === 'lunar') {
      this.lunarStep(dt);
      this.writeCameraLunar();
      return;
    }
    if (this.mode === 'orbital') {
      this.orbitalStep(dt);
      this.writeCameraOrbital();
      return;
    }
    if (this.mode === 'fly') this.autopilot(dt);
    this.integrate(dt);
    this.writeCamera();
  }

  // ------------------------------------------------------------- lunar lander

  /**
   * Lunar lander physics (Phase 9 M9.3): vacuum point-mass descent inside
   * the moon's gravity well. No lift, no drag, no atmosphere; the engine
   * gimbals to null horizontal drift, and the autopilot flies a fuel-aware
   * descent-rate profile to a soft touchdown (<2.5 m/s vertical).
   *
   * Every scratch vector here comes from the dedicated _l* set: sharing the
   * earth-flight temps made the autopilot silently overwrite the gravity
   * accumulator once (VS froze) — never again.
   */
  private lunarStep(dt: number): void {
    // Landed: pin to the surface forever (gravity would otherwise pull the
    // lander back into the ground every step — post-touchdown sink bug).
    if (this.landed) {
      const rL = _lR.copy(this.pos).sub(this.moonC);
      const upL = _lUp.copy(rL).normalize();
      this.pos.copy(upL).multiplyScalar(R_MOON + moonHeight(upL.x, upL.y, upL.z)).add(this.moonC);
      this.vel.set(0, 0, 0);
      this.tAgl = 0;
      this.tVs = 0;
      this.tGs = 0;
      return;
    }
    const r = _lR.copy(this.pos).sub(this.moonC);
    const n = Math.max(r.length(), 1);
    const up = _lUp.copy(r).multiplyScalar(1 / n);
    const hSurf = moonHeight(up.x, up.y, up.z);
    const agl = n - R_MOON - hSurf;

    // gravity (moon GM) — dedicated accumulator, never aliased
    const acc = _lAcc.copy(up).multiplyScalar(-MU_MOON / (n * n));

    // controls: W/S throttle up/down
    const c = this.rig.ctl;
    const thrIn = (this.rig.isDown('KeyW') ? 0.5 : 0) - (this.rig.isDown('KeyS') ? 0.5 : 0);
    this.thr = clamp(this.thr + thrIn * dt, 0, 1);

    // autopilot: scripted descent when active (writes stick + throttle)
    if (this.apPhase === 'descent') this.lunarAutopilot(agl, up);

    // thrust: engine down (-up), gimbaled by the stick (capped 20 deg)
    const tilt = clamp(Math.hypot(c.pitch, c.roll) * 20 * DEG, 0, 20 * DEG);
    if (tilt > 1e-4 && this.thr > 0.01 && this.prop > 0) {
      const east = _lE.crossVectors(this.WORLD_Y, up).normalize();
      if (east.lengthSq() < 0.1) east.set(0, 0, 1);
      const north = _lN.crossVectors(up, east).normalize();
      // stick pitch -> north tilt, roll -> east tilt (simple gimbal mapping)
      // The engine exhausts DOWNWARD, so the force on the craft is UP (+up),
      // tilted by the gimbal. (Thrusting -up is a brake-less dive — v1 bug.)
      const tUp = _lTUp
        .copy(up)
        .addScaledVector(north, -c.pitch * Math.sin(tilt))
        .addScaledVector(east, -c.roll * Math.sin(tilt))
        .normalize();
      // thrust with propellant check
      const mdot = this.BURN_RATE * this.thr;
      const burn = Math.min(this.prop, mdot * dt);
      const frac = mdot > 0 ? burn / (mdot * dt) : 0;
      this.prop -= burn;
      acc.addScaledVector(tUp, (this.thr * frac * this.THRUST_LANDER) / (this.mLand + this.prop));
    }

    this.vel.addScaledVector(acc, dt);
    this.pos.addScaledVector(this.vel, dt);

    // upright attitude: slerp the engine axis (body +Y) toward the local sky
    const upB = _lB1.set(0, 1, 0).applyQuaternion(this.q);
    const axis = _lB2.crossVectors(upB, up);
    const sinA = axis.length();
    if (sinA > 1e-6) {
      axis.multiplyScalar(1 / sinA);
      const ang = Math.asin(clamp(sinA, -1, 1)) * Math.min(1, dt * 1.5);
      this._dq.setFromAxisAngle(axis, ang);
      this.q.premultiply(this._dq).normalize();
    }

    // ground contact — re-evaluated AFTER integration so fast descents can
    // not tunnel through the surface between steps
    const r2 = _lR.copy(this.pos).sub(this.moonC);
    const n2 = r2.length();
    const up2 = _lUp.copy(r2).multiplyScalar(1 / n2);
    const hS2 = moonHeight(up2.x, up2.y, up2.z);
    const agl2 = n2 - R_MOON - hS2;
    if (agl2 < 0 && !this.landed) {
      const vs = this.vel.dot(up2);
      this.pos.copy(up2).multiplyScalar(R_MOON + hS2).add(this.moonC);
      this.vel.set(0, 0, 0);
      if (vs < -2.5) {
        this.frozen = true;
        this.note = 'CRASHED';
        this.apPhase = 'off';
      } else {
        this.landed = true;
        this.thr = 0;
        if (this.apPhase === 'descent') {
          this.apPhase = 'off';
          this.note = 'LANDED';
        }
      }
      this.tAgl = 0;
      this.tVs = 0;
      this.tGs = 0;
      return;
    }

    // telemetry
    this.tAgl = Math.max(agl2, 0);
    this.tVs = this.vel.dot(up2);
    this.tGs = this.vel.length();
    this.tIas = 0; // no atmosphere
  }

  /**
   * Scripted lunar descent: hold a descent-rate schedule (120 m/s free-fall
   * high up, 50 m/s under 6 km, 8 m/s under 300 m, 1.5 m/s under 30 m),
   * nulling horizontal drift with gimbal tilt. Touchdown <2.5 m/s.
   * Writes throttle + stick; scratch vectors from the _l* set only.
   */
  private lunarAutopilot(agl: number, up: THREE.Vector3): void {
    const m = this.mLand + this.prop;
    const gLocal = MU_MOON / Math.pow(this.pos.distanceTo(this.moonC), 2);
    // target vertical speed schedule
    const vsTgt =
      agl > 6000 ? -Math.min(120, Math.sqrt(2 * gLocal * Math.max(agl - 3000, 0)))
      : agl > 300 ? -50
      : agl > 30 ? -8
      : -1.5;
    // thrust to close the VS gap: a = g + (vsTgt - vs)/tau
    const vs = this.vel.dot(up);
    const tau = agl > 300 ? 3 : 1.2;
    const aNeed = gLocal + (vsTgt - vs) / tau;
    const aMax = this.prop > 0 ? this.THRUST_LANDER / m : 0;
    this.thr = clamp(aNeed / aMax, 0, 1);
    // horizontal drift nulling via stick (proportional). Thrust direction is
    // -up + north*(-pitch*s) + east*(-roll*s), so opposing an eastward drift
    // needs roll > 0 and a northward drift needs pitch > 0.
    const east = _lE.crossVectors(this.WORLD_Y, up).normalize();
    const north = _lN.crossVectors(up, east).normalize();
    const vh = _lVh.copy(this.vel).addScaledVector(up, -vs);
    const driftE = vh.dot(east);
    const driftN = vh.dot(north);
    const c = this.rig.ctl;
    c.roll = clamp(driftE * 0.05, -1, 1);
    c.pitch = clamp(driftN * 0.05, -1, 1);
  }

  /** Camera write for the lander (near plane from AGL). */
  private writeCameraLunar(): void {
    // Landing camera: rides a few meters above the feet, looking straight
    // down at the surface (the body attitude animates the lander, the view
    // here is the descent camera).
    const up = _lUp.copy(this.pos).sub(this.moonC).normalize();
    const camAbs = _lB1.copy(this.pos).addScaledVector(up, 2.5);
    this.world.rel(camAbs, this._tmp);
    this.rig.camera.position.copy(this._tmp);
    const belowAbs = _lB2.copy(this.pos).addScaledVector(up, -1000);
    this.world.rel(belowAbs, _lTUp);
    this.rig.camera.up.copy(this.WORLD_Y);
    this.rig.camera.lookAt(_lTUp);
    const near = clamp(this.tAgl * 0.25 + 0.3, 0.3, 5e4);
    if (Math.abs(near - this.lastNear) / near > 0.3 || this.lastNear < 0) {
      this.lastNear = near;
      this.rig.camera.near = near;
      this.rig.camera.updateProjectionMatrix();
    }
  }

  // ------------------------------------------------------------------ physics

  /**
   * Orbital mechanics step (Phase 9 M9.5, patched conics).
   *
   * Coasting: propagate on the Kepler rail (analytic, exact at any warp).
   * Thrusting (TLI): numeric kick because the rail is only valid inertially.
   * SOI handoff: when the moon's SOI sphere (66.2 Mm) is entered the primary
   * switches to the moon — state stays untouched (absolute frame, moon
   * frozen), only the gravity well changes.
   */
  private orbitalStep(dt: number): void {
    // M10.8e: rumble envelope decays between steps (re-armed each slice
    // where drag is significant)
    this.shake *= Math.exp(-dt * 3);
    // Coast time-warp: the rail is analytic, so propagation is exact at any
    // dt; warp scales the mission clock too so phase timers (TLI after one
    // parking-orbit period) fire in demo-realistic wall time.
    let wdt = dt * this.coastWarp;
    // clamp the final coast step so TLI ignites exactly on the node (the
    // spawn point: the raise ellipse's apoapsis then faces the frozen moon)
    // (M10.8: only when a TLI is armed — with ?notli=1 the clamp would
    // freeze wdt at 0 every lap and the craft would never move)
    if (this.apPhase === 'coast' && !this.noTli && this.el.period > 0) {
      wdt = Math.min(wdt, Math.max(this.el.period - this.orbT, 0));
    }
    this.orbT += wdt;

    // --- M10.8c lob boost phase (pad launch, first 60 s) -----------------
    // A real rocket ascent instead of teleporting to 2300 m/s: 900 kN
    // booster (TWR ~9 earth) for 60 s, pitch program vertical → 45° over
    // the first 40 s, then the ballistic coast takes over. Numeric
    // integration like the atmosphere block (the rail would ignore thrust).
    if (this.lob && this.apPhase === 'boost') {
      this.lobT += wdt;
      const upB = _oUp.copy(this.pos).normalize();
      const tanB = _oRail.copy(upB).cross(this.WORLD_Y).normalize();
      // pitch: 90° (vertical) at t=0 → 45° by t=40 s, hold 45°
      const pitch = clamp(90 - (this.lobT / 40) * 45, 45, 90);
      const rad = pitch * Math.PI / 180;
      const thrustDir = _oV.copy(upB).multiplyScalar(Math.sin(rad))
        .addScaledVector(tanB, Math.cos(rad)).normalize();
      const slices = Math.min(Math.ceil(wdt / 0.25), 96);
      const step = wdt / slices;
      for (let i = 0; i < slices; i++) {
        const m = this.mLand + this.prop;
        this.vel.addScaledVector(thrustDir, this.LOB_THRUST * step / m);
        this.vel.addScaledVector(upB, -this.primMu / (this.pos.lengthSq()) * step);
        this.pos.addScaledVector(this.vel, step);
        this.prop = Math.max(0, this.prop - this.BURN_RATE * step);
      }
      if (this.lobT >= 60 || this.prop <= 10) {
        this.apPhase = 'off'; // ballistic from here (atm block takes over)
        this.note = 'MECO';
      }
      this.orbT -= wdt; // boost time is not coast time
    } else if (this.apPhase !== 'tli') {
      const rail = _oRail.copy(this.pos).sub(this.primC);
      const railV = _oRailV.copy(this.vel);
      propagateKepler(rail, railV, this.primMu, wdt, _oNew, _oNewV);
      this.pos.copy(this.primC).add(_oNew);
      this.vel.copy(_oNewV);
    }

    // --- mission phases -------------------------------------------------
    if (this.apPhase === 'coast') {
      // First coast: verify the rail (one lap) then ignite TLI at apoapsis
      // of the raise ellipse. For the demo we ignite after one full period.
      // ?notli=1 (M10.8 entry testing): stay in coast forever so a low-pe
      // orbit can dip into the atmosphere and reenter without the TLI burn
      // hijacking the trajectory mid-test.
      if (!this.noTli && this.orbT >= this.el.period && this.el.period > 0) {
        this.apPhase = 'tli';
        this.orbT = 0;
      }
    }

    const r = _oR.copy(this.pos).sub(this.primC);
    const rm = r.length();

    // --- SOI handoff (earth -> moon) ------------------------------------
    if (!this.inMoonSoi) {
      const dMoon = this.pos.distanceTo(this.moonC);
      if (dMoon < SOI_MOON) {
        this.inMoonSoi = true;
        this.primC.copy(this.moonC);
        this.primMu = MU_MOON;
        // Patched-conic capture at the SOI boundary: replace the incoming
        // hyperbolic velocity (v∞ ≈ 560 m/s, mostly RADIAL — the transfer
        // ellipse ends on the moon center) with the apolune speed of the
        // target ellipse (rp 500 km altitude, ra = entry radius), directed
        // prograde-tangential. A retrograde-only burn along the incoming
        // velocity would preserve the radial component and drive the new
        // orbit's periapsis inside the moon — the tangential replacement is
        // the clean impulsive capture. The moon is frozen in this demo
        // (?moonangle), so absolute velocity is already moon-relative.
        const rRel = _oR.copy(this.pos).sub(this.moonC);
        const rr = rRel.length();
        const aT = (R_MOON + 500_000 + rr) / 2;
        const vTgt = Math.sqrt(MU_MOON * (2 / rr - 1 / aT));
        // prograde tangential unit vector: ĥ × r̂ with h = r × v
        const hV = _oV.copy(rRel).cross(this.vel);
        const tDir = _oUp.copy(hV).normalize().cross(rRel).normalize();
        const dv = _oB1.copy(tDir).multiplyScalar(vTgt).sub(this.vel).length();
        if (this.prop > 10) {
          this.vel.copy(tDir).multiplyScalar(vTgt);
          const ve = this.THRUST_LANDER / this.BURN_RATE;
          this.prop = Math.max(0, this.prop - (this.mLand + this.prop) * (1 - Math.exp(-dv / ve)));
          this.apPhase = 'lunar-orbit';
          this.note = 'CAPTURED';
        } else {
          this.apPhase = 'soi-moon';
          this.note = 'SOI MOON';
        }
      }
    }

    // --- thrust (impulse only — the coast above already propagated) ------
    if (this.apPhase === 'tli') {
      // Patched-conic TLI: impulsive prograde burn from the circular parking
      // speed to the transfer-ellipse speed at this radius (vis-viva). KSP-
      // style instant maneuver — a finite burn with the lander engine would
      // smear over ~50° of arc and miss the moon entirely. Propellant via
      // Tsiolkovsky with the lander engine's effective exhaust velocity
      // (ve = F/mdot). Ignition is exactly on the spawn node (mission clock
      // = one parking period, enforced by the wdt clamp), so the raised
      // apoapsis faces the frozen moon.
      const vDir = _oV.copy(this.vel).normalize();
      const vTgt = Math.sqrt(this.primMu * (2 / rm - 1 / ((this.tliRa + rm) / 2)));
      const dv = vTgt - this.vel.length();
      if (dv <= 0 || this.prop <= 10) {
        this.apPhase = 'trans-lunar';
        this.note = this.prop <= 10 ? 'NO PROP' : 'TLI DONE';
      } else {
        this.vel.addScaledVector(vDir, dv);
        const ve = this.THRUST_LANDER / this.BURN_RATE;
        this.prop = Math.max(0, this.prop - (this.mLand + this.prop) * (1 - Math.exp(-dv / ve)));
        this.apPhase = 'trans-lunar';
        this.note = 'TLI DONE';
      }
    }

    // --- post-step: elements, altitude, ground guard --------------------
    const r2 = _oR.copy(this.pos).sub(this.primC);
    elementsOf(r2, this.vel, this.primMu, this.el);
    const up2 = _oUp.copy(r2).multiplyScalar(1 / r2.length());
    const alt = r2.length() - (this.primMu === MU ? R : R_MOON);
    this.tAgl = Math.max(alt, 0);
    this.tVs = this.vel.dot(up2);
    this.tGs = this.vel.length();
    this.tIas = 0;

    // --- M10.8: atmospheric entry (earth atmosphere only) ----------------
    // Below the atmosphere top the Kepler rail is no longer the whole story:
    // drag bleeds energy, the craft decelerates, heats, and eventually falls
    // ballistically. Sub-stepped numeric integration replaces the analytic
    // rail inside the atmosphere (small slices keep the v^3 heating and the
    // drag honest at 100x warp); outside it the rail stays exact.
    this.heat = 0;
    // M10.8c parachute: a 10.2 t lander on a 28 m² shield alone still hits
    // at ~60 m/s. Below 9 km and subsonic (<340 m/s), stage a main chute —
    // 650 m² / Cd 1.4 gives a survivable ~14 m/s splashdown. Staged INSIDE
    // the slice loop (fresh altitude each 0.25 s piece) so a fast descent
    // can't skip past the gate between frames.
    const paraA = this.paraOpen ? 650.0 : 0.0;
    if (this.primMu === MU && alt < ATMOS_TOP) {
      // Integrate the SAME warp-scaled step the rail used (wdt), sliced
      // into <=0.25 s pieces. M10.8c fix: this block previously advanced
      // by raw dt while the rail advanced dt*warp — at warp 10 the craft
      // coasted 10x faster than the atmosphere dragged it, so entry never
      // decelerated and every warp>1 landing lithobraked.
      // Bounded work: at extreme warp cap the slice count (coarser slices,
      // drag still bleeds the energy — precision matters less than staying
      // interactive during 20000x coast frames).
      const v0 = this.vel.length();
      const slices = Math.min(Math.ceil(wdt / 0.25), 96);
      const h = wdt / slices;
      let remaining = wdt;
      while (remaining > 1e-6) {
        const step = Math.min(h, remaining);
        remaining -= step;
        const rr = _oR.copy(this.pos).sub(this.primC);
        const upS = _oUp.copy(rr).multiplyScalar(1 / rr.length());
        const altS = rr.length() - R;
        if (altS >= ATMOS_TOP) break; // skipped back out (skip-up trajectory)
        const rho = isaDensity(Math.max(altS, 0));
        this.airDensity = rho;
        const vS = this.vel.length();
        // descending = retrograde motion along up (the burn/chute/shield
        // staging only apply on the way DOWN, not on the lob ascent)
        const vsS = this.vel.dot(upS);
        const descending = vsS < 0;
        // parachute gate (fresh altitude per slice)
        if (!this.paraOpen && descending && altS < 9000 && vS < 340) {
          this.paraOpen = true;
          this.note = 'PARACHUTE';
        }
        // M10.8: once the plasma is hot the ablative heat shield deploys:
        // the bare lander hull has a small 5 m² attached area (ok for a
        // propulsive moon landing) but orbital entry needs a blunt shield —
        // scale to a Dragon-class 28 m² / Cd 1.5 when heating is significant.
        const shieldA = this.heat > 0.05 && descending ? 28.0 : 5.0;
        const shieldCd = this.heat > 0.05 && descending ? 1.5 : 1.2;
        // deceleration: drag on the shield + staged main chute (Cd*A sums)
        const dragA = (shieldCd * shieldA + (this.paraOpen && descending ? 1.4 * paraA : 0))
          * 0.5 * rho * vS * vS / (this.mLand + this.prop);
        // stagnation heat flux (Sutton-Graves, k=1.7e-4, W/m^2) -> telemetry
        this.heatFlux = descending ? 1.7e-4 * Math.sqrt(rho) * vS * vS * vS : 0;
        this.gLoad = dragA / 9.80665;
        // M10.8c landing burn: the lander's engine (45 kN, TWR 0.45 earth)
        // can't hover but adds ~4.9 km/s of dv over the tank — enough to
        // finish what the shield started. Retro-thrust below 40 km, past
        // the plasma peak (heat<0.5: the flux itself keeps RISING through
        // the thick-air phase, so gating on flux would never open).
        if (this.prop > 0 && descending && altS < 40000 && this.heat < 0.5 && vS > 60) {
          const acc = this.THRUST_LANDER / (this.mLand + this.prop);
          this.vel.addScaledVector(this.vel, -Math.min(acc * step / Math.max(vS, 1e-6), 0.9));
          this.prop = Math.max(0, this.prop - this.BURN_RATE * step);
        }
        this.vel.addScaledVector(this.vel, -dragA * step / Math.max(vS, 1e-6));
        // gravity during the slice (rail no longer carries it)
        this.vel.addScaledVector(upS, -this.primMu / (rr.length() * rr.length()) * step);
        this.pos.addScaledVector(this.vel, step);
        // peak heat drives the glow: normalized 0..1 over ~1 MW/m^2 with
        // a slow cool-down so the plasma persists through the peak region
        this.heat = Math.max(this.heat, clamp(this.heatFlux / 1e6, 0, 1));
        // M10.8e entry shake: peak dynamic pressure this step drives the
        // camera rumble (writeCameraOrbital adds the offset). g/10 capped
        // — 5 g+ reads as violent vibration.
        this.shake = Math.max(this.shake, clamp(this.gLoad / 10, 0, 1));
      }
      this.heat = Math.max(this.heat, this.heat * Math.exp(-dt * 0.35));
      // stale telemetry guard: leaving the atmosphere must zero the
      // per-slice readouts (they were frozen at the last slice's values)
      if (this.heat < 0.01) {
        this.heatFlux = 0;
        this.gLoad = 0;
      }
      // ENTRY note: set once heat is significant, keep it until landing/
      // impact (any phase — the lob runs apPhase 'off' after MECO)
      if (this.heat > 0.03) {
        this.note = 'ENTRY';
      }
      this.tGs = this.vel.length();
    }

    // ground/crash guard (never expected on a clean orbit, but a bad burn
    // can drop periapsis into the planet). M10.8c: r2 was captured BEFORE
    // the atmospheric block moved the craft — recompute so a soft chute
    // touchdown is detected the same frame instead of oscillating around
    // the surface (rail + drag fighting each frame).
    const r2b = _oR.copy(this.pos).sub(this.primC);
    const up2b = _oUp.copy(r2b).multiplyScalar(1 / r2b.length());
    const surfR = (this.primMu === MU ? R : R_MOON) + (this.primMu === MU
      ? terrainHeight(up2b.x, up2b.y, up2b.z)
      : moonHeight(up2b.x, up2b.y, up2b.z));
    if (r2b.length() < surfR) {
      this.pos.copy(up2b).multiplyScalar(surfR).add(this.primC);
      const vsTouch = this.vel.dot(up2b);
      this.vel.set(0, 0, 0);
      this.paraOpen = false;
      this.frozen = true;
      // touchdown classification: chute terminal ~20 m/s lands intact,
      // anything faster is a crash
      this.note = vsTouch > -30 ? 'LANDED' : 'CRASHED';
      this.apPhase = 'off';
      // M10.8: clear entry telemetry — otherwise the HUD shows a stale
      // HEAT/g readout forever after touchdown.
      this.heatFlux = 0;
      this.gLoad = 0;
      this.heat = 0;
      this.tAgl = 0;
      this.tVs = 0;
      this.tGs = 0;
    }
  }

  /** Camera write for the orbital view: ride slightly behind/above, look
   * down the velocity vector tilted toward the surface so the planet fills
   * the frame; near plane tracks altitude like the lander cam. */
  private writeCameraOrbital(): void {
    const up = _oUp.copy(this.pos).sub(this.primC).normalize();
    // look direction: surface point below the craft (nadir). The camera
    // itself sits 30 m "above" the craft along up so the HUD-style probe at
    // the frame center samples the planet, not the vehicle.
    const camAbs = _oB1.copy(this.pos).addScaledVector(up, 30);
    // M10.8e entry rumble: offset the camera along up/right by the shake
    // envelope (high-q vibration), plus a slow g-induced sway
    if (this.shake > 0.003) {
      const t = performance.now() / 1000;
      const right = _oRail.copy(up).cross(this.WORLD_Y).normalize();
      camAbs.addScaledVector(up, Math.sin(t * 61) * 0.6 * this.shake)
        .addScaledVector(right, Math.sin(t * 47 + 1.3) * 0.5 * this.shake);
    }
    this.world.rel(camAbs, this._tmp);
    this.rig.camera.position.copy(this._tmp);
    const belowAbs = _oB2.copy(this.pos).addScaledVector(up, -this.tAgl);
    this.world.rel(belowAbs, _oRail);
    this.rig.camera.up.copy(this.WORLD_Y);
    this.rig.camera.lookAt(_oRail);
    // near plane: altitude-tracking like the lander (0.25·AGL + floor),
    // capped so the whole planet still fits inside far=2e9.
    const near = clamp(this.tAgl * 0.25 + 0.3, 0.3, 1e5);
    if (Math.abs(near - this.lastNear) / near > 0.3 || this.lastNear < 0) {
      this.lastNear = near;
      this.rig.camera.near = near;
      this.rig.camera.updateProjectionMatrix();
    }
  }

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
    const la =
      this.mode === 'lunar' ? (this.landed ? 'TOUCHDOWN' : this.apPhase.toUpperCase()) :
      this.mode === 'fly' || this.mode === 'orbital' ? this.apPhase.toUpperCase() : 'MANUAL';
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
    if (this.mode === 'lunar') {
      return (
        `LANDER GS ${s.gs.toFixed(1)}  AGL ${fmtM(s.agl)}  VS ${s.vs >= 0 ? '+' : ''}${s.vs.toFixed(1)}  ` +
        `THR ${(s.thr * 100).toFixed(0)}%  PROP ${this.prop.toFixed(0)} kg  ` +
        `[${s.phase}${s.note ? ' ' + s.note : ''}]`
      );
    }
    if (this.mode === 'orbital') {
      const body = this.primMu === MU ? 'E' : 'M';
      const el = this.el;
      const apo = el.ra === Infinity ? '∞' : fmtM(el.ra - (body === 'E' ? R : R_MOON));
      const entry = this.heatFlux > 1e4
        ? `  HEAT ${(this.heatFlux / 1e6).toFixed(2)}MW/m2  ${this.gLoad.toFixed(1)}g`
        : '';
      return (
        `ORBIT(${body}) GS ${s.gs.toFixed(0)} m/s  ALT ${fmtM(s.agl)}  ` +
        `a ${fmtM(el.a)}  e ${el.e.toFixed(4)}  ` +
        `Pe ${fmtM(el.rp - (body === 'E' ? R : R_MOON))}  Ap ${apo}  ` +
        `T ${el.period > 0 ? (el.period / 60).toFixed(1) + 'min' : '--'}  ` +
        `PROP ${this.prop.toFixed(0)}kg  [${s.phase}${s.note ? ' ' + s.note : ''}]` +
        entry
      );
    }
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
// dedicated lunar lander temps — the autopilot and the physics step run in
// the same call stack, so sharing ANY of these between the two (or with the
// earth-flight temps) silently corrupts the other user. Never alias.
const _lR = new THREE.Vector3();
const _lUp = new THREE.Vector3();
const _lAcc = new THREE.Vector3();
const _lE = new THREE.Vector3();
const _lN = new THREE.Vector3();
const _lTUp = new THREE.Vector3();
const _lVh = new THREE.Vector3();
const _lB1 = new THREE.Vector3();
const _lB2 = new THREE.Vector3();
// dedicated orbital-mode temps (same no-aliasing rule as _l*).
const _oR = new THREE.Vector3();
const _oUp = new THREE.Vector3();
const _oV = new THREE.Vector3();
const _oRail = new THREE.Vector3();
const _oRailV = new THREE.Vector3();
const _oNew = new THREE.Vector3();
const _oNewV = new THREE.Vector3();
const _oB1 = new THREE.Vector3();
const _oB2 = new THREE.Vector3();

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
