import * as THREE from 'three';
import { terrainHeight } from './terrain';
import { ATMOSPHERE_TOP as ATMOS_TOP } from './atmosphere';
import { moonHeight } from './moon';
import { moonCenterFromParams, moonPositionAtAngle, MOON_ORBIT_R, INCLINATION } from './moonOrbit';
import { propagateKepler, elementsOf, SOI_MOON, type OrbitalElements } from './orbit';
import { EARTH, MOON, nearestFrame, localToAbsolute, eastAt } from './frames';
import type { CameraRig } from './cameraRig';
import type { WorldOrigin } from './world';

const DEG = Math.PI / 180;
const R = EARTH.radius;
const MU = EARTH.mu;
const R_MOON = MOON.radius;
// M11w10 moonshot cinematic timing (sim seconds; the ascent runs at warp 3-3.5)
/** Pre-launch pad hold — the camera frames the moon over the horizon. */
const MS_HOLD_S = 24;
/** Stage-2 push length before the cut to the parking orbit (195 sim s at
 *  warp 8 ≈ 24 wall s — the arc ends near 270 km / 6 km/s, so the cut to
 *  the 250 km parking orbit is nearly continuous). */
const MS_S2_CLIMB_S = 195;
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
  /** ?full=1: TLI → 15km-perilune capture → powered descent → landing. */
  private fullMission = false;
  /** M11j ?moonshot=1: pad → ascent → orbit → TLI → descent, the whole
   * outbound half of a moon landing in one scripted demo (~5 min).
   * Implies the fullMission capture/descent path. */
  private moonshot = false;
  /** M11j moonshot ascent stage: 1 = booster (LOX/Kero), 2 = upper stage. */
  private msStage = 0;
  /** M11j moonshot ascent clock (stage-local), s. */
  private msT = 0;
  /** M11w10 moonshot night-side deorbit done flag. */
  private msDeorbit = false;
  /** M11j moonshot staging-event counter for the audio crack. */
  msStageEvents = 0;
  /** M11j: true while a moonshot demo owns the moon's placement (main.ts
   * must not overwrite MOON.center with the live clock position). */
  get moonshotActive(): boolean {
    return this.moonshot;
  }
  /** M11j: read-only access to the demo's frozen moon center. */
  get moonCenter(): THREE.Vector3 {
    return this.moonC;
  }
  /** M11j probe: distance to the demo's frozen moon center (m). */
  get probeMoonDist(): number {
    return this.pos.distanceTo(this.moonC);
  }
  /** M11 descent autopilot state: r-trend around the moon. */
  private lastR = 0;
  private rTrendUp = false;
  /** ?lob=1 boost-phase clock, s (MECO at 60). */
  private lobT = 0;
  /** ?lob=1 booster thrust, N (TWR ~9 at ignition). */
  private readonly LOB_THRUST = 900_000;
  /** M10.8c: main parachute staged (earth entry, <9 km and subsonic). */
  private paraOpen = false;
  /** M11i: drogue chute staged first (fast/high, small canopy). */
  private drogueOpen = false;
  /** M11i: canopy inflation progress 0..1 — chutes fill over ~1-2 s in
   * reality; ramping the area spreads the opening shock (a full 650 m²
   * main snapping open in one slice spiked the accelerometer to ~435 g). */
  private drogueT = 0;
  private paraT = 0;
  /** M11i: staging-event counter (audio crack hook; public one-shot). */
  chuteEvents = 0;
  /** M10.8e: entry rumble 0..1 (peak q this step; decays each frame). */
  private shake = 0;
  /** M10.9: last slice's air density (kg/m^3) for the audio wind layer. */
  airDensity = 0;
  /** M10.9: true while the lob booster is firing (audio layer gate). */
  get boostPhase(): boolean {
    return this.lob && this.apPhase === 'boost';
  }
  /** M11 probe: moon-relative horizontal speed during descent. */
  get probeHoriz(): number {
    if (this.primMu !== MU_MOON) return 0;
    const r = _oR.copy(this.pos).sub(this.moonC);
    const up = _oUp.copy(r).normalize();
    return _oB1.copy(this.vel).addScaledVector(up, -this.vel.dot(up)).length();
  }

  // --- M10.8 entry telemetry (orbital mode, earth atmosphere) ---
  /** Normalized plasma/heat glow 0..1 (drives the viewport entry effect). */
  heat = 0;
  /** Stagnation heat flux, W/m^2 (Sutton-Graves). */
  heatFlux = 0;
  /** Deceleration load in g. */
  gLoad = 0;
  // M11f: mission summary bookkeeping (peak values, reset on spawn)
  peakG = 0;
  peakHeat = 0;
  missionT = 0;
  touchdownVs = 0;
  // M11g: lunar-return mission flag + one-shot TEI guard
  returnMission = false;
  returnedFromMoon = false;
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
      // M11j: the moonshot demo warps hard on the trans-lunar coast so
      // the whole outbound leg fits in ~5 wall minutes (the 3-day coast
      // at 20000x ≈ 13 s), but the DEFAULT is capped so the orbital
      // phasing stays visible.
      if (this.moonshot) this.coastWarp = clamp(num(q, 'warp', 20000), 1, 50_000);
      // ?pe=<m>: spawn directly on an elliptical orbit with this periapsis
      // (M10.8 entry testing — a low pe like 20000 dives the orbit into the
      // atmosphere without waiting for the TLI/moon loop). The spawn stays
      // AT apoapsis (radius R+alt) with circular speed for that radius, so
      // the craft falls toward the pe on the far side.
      this.peOverride = q.has('pe') ? num(q, 'pe', 200_000) : null;
      this.noTli = q.has('notli');
      // ?full=1 (M11): the complete TLI → capture → powered-descent →
      // landing scenario. Capture targets a 15 km perilune ellipse
      // directly and the descent autopilot takes over at perilune.
      this.fullMission = q.has('full');
      // M11j ?moonshot=1: launch from the pad on the moon's orbit plane,
      // ascend to orbit, TLI, capture, and land — a ~5-minute scripted
      // round trip with warp where nothing interesting happens. The
      // moon is PLACED opposite the orbit-insertion point so the TLI
      // apoapsis faces it exactly (see spawnStateInit).
      this.moonshot = q.has('moonshot');
      if (this.moonshot) this.fullMission = true;
      // ?return=1 (M11g): parked in a 200 km lunar orbit — the TEI autopilot
      // burns at the anti-earth point and the earthward leg ends in a
      // 25 km-pe reentry with the chute path from M10.8.
      this.returnMission = q.has('return');
      // The landing burn alone needs ~1.4 t (1730 m/s at ve 5625 from the
      // 15 km-perilune ellipse); TLI + capture eat 7.3 t of tank, so the
      // full mission spawns with a tanker-tender budget.
      if (this.fullMission) this.prop = 14000;
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
      // M11w: the return mission owns the moon's phase too. With the default
      // live-clock placement (+X, the day side) the TEI hands the craft a
      // tangential retrograde velocity, so the return ellipse's 25 km
      // perigee — the splashdown point — sits at the moon's ANTIPODE, and
      // every return mission splashed down on the night side (in-plane
      // perigee rotation is unaffordable: 10 deg costs ~830 m/s at this
      // ra/rp ratio, 15 deg goes hyperbolic). Park the moon on the
      // anti-solar side instead: its antipode then enjoys a ~38 deg sun
      // elevation at splashdown, and the entry crosses the terminator into
      // daylight (same trick as the moonshot's Math.PI placement below).
      // ?moonangle= keeps its test-hook meaning and overrides this.
      if (this.returnMission && !q.has('moonangle')) {
        moonPositionAtAngle(235 * DEG, this.moonC);
      }
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
      // M11j ?moonshot=1: pad launch into the MOON'S ORBIT PLANE. The pad
      // sits at the -Z side of the inclined plane, the moon gets placed
      // at the antipode (+Z side, 60° ahead of the insertion point) so
      // the post-TLI apoapsis faces it exactly — no moonangle needed,
      // the demo owns the geometry. Physics = the lob boost ladder with
      // a plane-aligned pitch program; stages 1→2→circularize→TLI.
      if (this.moonshot) {
        // plane basis: the moon's orbit plane spans ê1=+X and ê2=(0,si,ci)
        const ci = Math.cos(INCLINATION), si = Math.sin(INCLINATION);
        // M11w10: the pad sits `?mspad=`° around the moon's orbit plane from
        // +X (default 112). The sun is fixed near +X, so this puts the pad
        // just past sunset (sun ~0.5° below the horizon): the pre-launch
        // shot holds the twilight ocean horizon with stars emerging and the
        // full moon ~22° up on the right (112 was picked over deeper-dusk
        // angles 116-122 by screenshot A/B — deeper dusk darkens the horizon
        // to black and kills the composition; the moon is at (877,180) via
        // the ?loddbg=1 moonNdc line, clear of the debug panel). The
        // insertion state is hard-overwritten at the orbit cut, so the pad's
        // position is free.
        const padAng = (Number(new URLSearchParams(location.search).get('mspad')) || 112) * DEG;
        const padDir = _oUp.set(Math.cos(padAng), Math.sin(padAng) * si, Math.sin(padAng) * ci).normalize();
        // in-plane downrange tangent t(a) = sin(a)·ê1 − cos(a)·ê2 points
        // toward decreasing a (the +X insertion / sunset side from this pad)
        const tanDir = _oRail.set(Math.sin(padAng), -Math.cos(padAng) * si, -Math.cos(padAng) * ci).normalize();
        const up = _oUp.copy(padDir);
        const padTh = terrainHeight(up.x, up.y, up.z);
        this.pos.copy(up).multiplyScalar(R + Math.max(padTh, 0) + 50);
        this.vel.set(0, 0, 0); // rest on the pad
        this.q.identity();
        this.thr = 0;
        this.frozen = false;
        this.note = 'MOONSHOT';
        this.orbT = 0;
        // tanker budget: 9 t stage-1 + ~42 t stage-2 + ~3.5 t TLI +
        // descent reserve — the stage-2 dv (12.1 km/s) must cover the
        // gravity/drag losses AND the TLI impulse with margin.
        this.prop = 55000;
        this.inMoonSoi = false;
        this.paraOpen = false;
        this.drogueOpen = false;
        this.drogueT = 0;
        this.paraT = 0;
        this.chuteEvents = 0;
        this.heat = 0;
        this.heatFlux = 0;
        this.gLoad = 0;
        this.primC.set(0, 0, 0);
        this.primMu = MU;
        this.apPhase = 'boost';
        this.lobT = 0;
        this.msStage = 1;
        this.msT = 0;
        this.msStageEvents = 0;
        // Moon: antipode of the insertion point (insertion = +X side of the
        // plane) — TLI at insertion kicks prograde toward this spot.
        this.moonC.set(0, 0, 0);
        moonPositionAtAngle(Math.PI, this.moonC);
        MOON.center.copy(this.moonC);
        elementsOf(_oR.copy(this.pos), this.vel, MU, this.el);
        return;
      }
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
        this.drogueOpen = false; // M11i
        this.drogueT = 0;
        this.paraT = 0;
        this.chuteEvents = 0;
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
      this.prop = this.fullMission ? 14000 : 6000;
      this.inMoonSoi = false;
      this.paraOpen = false;
      this.drogueOpen = false; // M11i
      this.drogueT = 0;
      this.paraT = 0;
      this.chuteEvents = 0;
      this.heat = 0;
      this.heatFlux = 0;
      this.gLoad = 0;
      // M11f: reset the mission summary peaks
      this.peakG = 0;
      this.peakHeat = 0;
      this.missionT = 0;
      this.touchdownVs = 0;
      this.primC.set(0, 0, 0);
      this.primMu = MU;
      this.apPhase = 'coast';
      // M11g ?return=1: override the LEO spawn with a 200 km circular
      // LUNAR orbit. The moon sits frozen at +X (MOON.center above); the
      // TEI autopilot (apPhase 'tei') handles the escape burn.
      if (this.returnMission) {
        this.primC.copy(this.moonC);
        this.primMu = MU_MOON;
        this.inMoonSoi = true;
        this.returnedFromMoon = false;
        this.prop = 4000;
        // 200 km circular orbit, prograde-tangential around the moon.
        // Orbit plane: worldY × moon-up so the burn at the anti-earth
        // point kicks the craft along -X (earthward).
        const rRel = _oR.copy(this.pos).sub(this.moonC);
        rRel.setLength(R_MOON + 200_000);
        this.pos.copy(this.moonC).add(rRel);
        const upM = _oUp.copy(rRel).multiplyScalar(1 / rRel.length());
        this.vel.copy(this.WORLD_Y).cross(upM).normalize()
          .multiplyScalar(Math.sqrt(MU_MOON / rRel.length()));
        this.apPhase = 'tei';
        this.orbT = 0;
      }
      // Seed the osculating elements so the coast-warp clamp (TLI exactly on
      // the spawn node) is valid from the very first frame — at high warp a
      // single unclamped step could overshoot the whole parking period.
      elementsOf(_oR.copy(this.pos), this.vel, this.primMu, this.el);
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
      // M11c: the lander's primary body is the moon from the first frame —
      // leave primC at the earth origin and every primC-relative consumer
      // (the prograde-horizon camera, altitude) aims 383 Mm off-target.
      this.primC.copy(this.moonC);
      this.primMu = MU_MOON;
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
    // M11w10: the moonshot drives a full cinematic warp schedule — the
    // ascent, the LEO lap and the transfer are all watched at deliberate
    // speed instead of the old 20000x blink.
    if (this.moonshot) {
      if (this.apPhase === 'boost') {
        wdt = dt * (this.msStage === 1 ? 3 : 8);
      } else if (this.apPhase === 'coast') {
        // one LEO lap ≈ 27 wall s. 200x (not 300x): the terrain tile
        // builder starves above ~100 km/frame of sub-point motion and the
        // coast renders black — 7.76 km/s × 200 × dt stays under it.
        wdt = dt * 200;
      } else if (this.apPhase === 'trans-lunar') {
        // slow while the earth still fills the moonward view (the moon
        // emerges from behind its limb in this window), then ramp hard for
        // the cruise once the earth has left the frame
        const rE = _oR.copy(this.pos).length() - R;
        const w = rE < 8e6 ? 150
          : rE < 6e7 ? 150 + (12000 - 150) * ((rE - 8e6) / 5.2e7)
          : 12000;
        wdt = dt * w;
      } else if (this.apPhase === 'lunar-orbit' && this.fullMission) {
        // the capture ellipse falls from ~66 Mm: cruise down fast, easing
        // off as the surface nears (the descent cap below takes the last
        // 200 km)
        const rAg = _oR.copy(this.pos).sub(this.moonC).length() - R_MOON;
        wdt = dt * (rAg > 5e6 ? 12000
          : 30 + (12000 - 30) * clamp((rAg - 2e5) / 4.8e6, 0, 1));
      }
    }
    // M11: auto-drop warp for the powered descent — 20000x would hand
    // the autopilot 3.5 s slices (limit-cycle bounce off the moon);
    // landing control needs <=0.5 s slices, i.e. warp <= 30 at 60 fps.
    if (this.apPhase === 'lunar-orbit' && this.fullMission) {
      const rAg = _oR.copy(this.pos).sub(this.moonC).length() - R_MOON;
      if (rAg < 200_000) wdt = Math.min(wdt, dt * 30);
    }
    // M11g: auto-drop warp on the earthward leg — a 20000x Kepler step
    // through perigee explodes (the analytic rail is exact per step, but
    // the SOLVER's time step at perigee speed 11 km/s x 20000x hops past
    // the planet). Cap the step when the earth is close; the atmosphere
    // block slices its own integration from there.
    if (this.apPhase === 'trans-earth') {
      const rE = _oR.copy(this.pos).length();
      if (rE < 2e7) wdt = Math.min(wdt, dt * 60);
      else if (rE < 1e8) wdt = Math.min(wdt, dt * 2000);
    }
    // M11j: same solver blowup hit the moonshot's trans-lunar leg (the
    // transfer's perigee is the parking radius; a 20000x step across it
    // produced 1e5 m/s ghosts). Clamp when close to the earth on the way
    // out as well — the coast is where the warp should be big, not here.
    // (M11w10: the moonshot's cinematic schedule already starts the transfer
    // at 150x and ramps past the perigee region, so it opts out.)
    if (this.apPhase === 'trans-lunar' && !this.moonshot) {
      const rE = _oR.copy(this.pos).length();
      if (rE < 2e7) wdt = Math.min(wdt, dt * 60);
      else if (rE < 6e7) wdt = Math.min(wdt, dt * 2000);
    }
    // clamp the final coast step so TLI ignites exactly on the node (the
    // spawn point: the raise ellipse's apoapsis then faces the frozen moon)
    // (M10.8: only when a TLI is armed — with ?notli=1 the clamp would
    // freeze wdt at 0 every lap and the craft would never move)
    if (this.apPhase === 'coast' && !this.noTli && this.el.period > 0) {
      wdt = Math.min(wdt, Math.max(this.el.period - this.orbT, 0));
    }
    this.orbT += wdt;
    // M11f: mission wall-clock — REAL seconds, not warp-scaled (a
    // warp-scaled clock reads 8616 minutes after a 20x-warp cruise and
    // means nothing to the player)
    this.missionT += dt;

    // --- M11j moonshot ascent (pad → orbit, two stages) -------------------
    // Same numeric-integration ladder as the lob boost, but the pitch
    // program flies IN THE MOON'S ORBIT PLANE (pad at -Z of the plane,
    // down-range toward +X where the TLI insertion happens):
    //   stage 1 (900 kN booster, 0-64 s warp 3): vertical → 55° by 60 s,
    //     burnout ~12 km / 267 m/s, staging event (shell jettison)
    //   stage 2 (450 kN vacuum, warp 12): 55° → 6°, throttle fades as
    //     the tangential speed nears circular; impulsive circularization
    //     above 140 km triggers apPhase 'coast' and the standard
    //     coast→TLI node logic takes over one period later.
    // NOTE: the boost branch integrates the full step itself (thrust +
    // drag + gravity + displacement) — the Kepler rail and the atmosphere
    // block are BOTH bypassed this step (else-if chain + the atm gate).
    if (this.moonshot && this.apPhase === 'boost') {
      const upB = _oUp.copy(this.pos).normalize();
      // in-plane tangent: ĥ = ŷ×r̂ is the moon-orbit-plane normal cross up…
      // simpler: the plane's normal is n̂ = ŷ×ẑrot — compute the in-plane
      // tangent as (n̂ × up) with n̂ the plane normal from INCLINATION.
      const nHat = _oB1.set(0, Math.sin(INCLINATION), Math.cos(INCLINATION)).normalize();
      // M11w10: proper in-plane tangent in the moon's orbit plane
      // (ê1=+X, ê2=nHat). The old n̂×up formula degenerated for pads away
      // from +X (it returned the plane normal, not the tangent). Sense:
      // toward the +X insertion hemisphere from the pad.
      const aPos = Math.atan2(this.pos.dot(nHat), this.pos.x);
      const tanB = _oRail.set(Math.sin(aPos), -Math.cos(aPos) * Math.sin(INCLINATION), -Math.cos(aPos) * Math.cos(INCLINATION)).normalize();
      this.msT += wdt;
      // M11w10 pre-launch hold: the first MS_HOLD_S seconds sit on the pad
      // with engines off (the camera holds the moon-over-horizon shot),
      // then ignition. The burn clock excludes the hold.
      const msTBurn = Math.max(this.msT - MS_HOLD_S, 0);
      const rNow = this.pos.length();
      const vCirc = Math.sqrt(MU / rNow);
      let thrustN: number;
      let burnRate: number;
      if (this.msStage === 1) {
        thrustN = this.msT < MS_HOLD_S ? 0 : 900_000;
        burnRate = 140; // ~64 s of burn on 9 t of stage-1 prop
        // M11j: MECO — hand to the stage-2 cinematic push (the orbit cut
        // comes after it). Stage 1 carried the pad/horizon footage.
        if (this.prop <= 9000 || msTBurn >= 64) {
          this.msStage = 2;
          this.msT = 0;
          this.msStageEvents++;
          this.note = 'MECO';
        }
      } else {
        // M11w10 stage-2 push: 1.4 MN vacuum stage (TWR ~2.8 on the stack)
        // so the climb keeps altitude through the whole tilt-down beat;
        // the prop is discarded at the cut anyway.
        thrustN = 1_400_000;
        burnRate = 105;
      }
      // M11j stage-2 (post-cut): never reached — the MECO handoff above
      // returns straight into the coast phase. Kept only for the type
      // of thrustDir/throttle below.
      let thrustDir: THREE.Vector3;
      // M11w10: no prop burn during the pre-launch hold
      const throttle = this.msT < MS_HOLD_S && this.msStage === 1 ? 0 : 1;
      if (this.msStage === 1) {
        // stage 1: classic pitch program (90° → 55° over the burn minute)
        const pitchDeg = clamp(90 - (msTBurn / 60) * 35, 55, 90);
        const rad = pitchDeg * Math.PI / 180;
        thrustDir = _oV.copy(upB).multiplyScalar(Math.sin(rad))
          .addScaledVector(tanB, Math.cos(rad)).normalize();
      } else {
        // M11w10 stage-2 push: the gravity turn continues 55° → 18°; the
        // stack arcs over slightly at the very end — the cut below rescues
        // the state, and the scripted camera never follows the nose anyway.
        const pitchDeg = clamp(55 - (this.msT / MS_S2_CLIMB_S) * 37, 18, 55);
        const rad = pitchDeg * Math.PI / 180;
        thrustDir = _oV.copy(upB).multiplyScalar(Math.sin(rad))
          .addScaledVector(tanB, Math.cos(rad)).normalize();
      }
      // M11j: reserve guard — if the remaining prop can no longer reach
      // the insertion gate, cut to the reserve and circularize NOW on
      // the patched conic with whatever altitude the arc has reached
      // (better a 120 km ellipse than a suborbital lob into the pad).
      if (this.msStage === 2 && this.prop <= 3000 && this.apPhase === 'boost') {
        const upI = _oUp.copy(this.pos).normalize();
        const tDirI = _oB3.copy(nHat).cross(upI).normalize();
        if (tDirI.dot(this.vel) < 0) tDirI.negate();
        this.vel.copy(tDirI).multiplyScalar(Math.sqrt(MU / this.pos.length()));
        this.apPhase = 'coast';
        this.orbT = 0;
        this.note = 'ORBIT*';
        elementsOf(_oR.copy(this.pos).sub(this.primC), this.vel, this.primMu, this.el);
      }
      // M11w10: pinned to the pad until ignition — the boost integrator
      // applies gravity, and with engines off the stack would fall the
      // 50 m to the ground during the hold and trip the crash guard.
      if (this.msStage === 1 && this.msT < MS_HOLD_S) {
        this.vel.set(0, 0, 0);
        this.orbT -= wdt; // boost time is not coast time
        return;
      }
      const slices = Math.min(Math.ceil(wdt / 0.25), 96);
      const step = wdt / slices;
      for (let i = 0; i < slices; i++) {
        const m = this.mLand + this.prop;
        this.vel.addScaledVector(thrustDir, thrustN * throttle * step / m);
        // atmosphere drag on the ascending stack (small, but honest)
        const rrS = _oB2.copy(this.pos);
        const altS = rrS.length() - R;
        const rhoS = isaDensity(Math.max(altS, 0));
        const vS = this.vel.length();
        if (rhoS > 1e-9 && vS > 1) {
          const dragA = 1.2 * 20.0 * 0.5 * rhoS * vS * vS / m;
          this.vel.addScaledVector(this.vel, -dragA * step / Math.max(vS, 1e-6));
        }
        this.vel.addScaledVector(upB, -this.primMu / (this.pos.lengthSq()) * step);
        this.pos.addScaledVector(this.vel, step);
        this.prop = Math.max(0, this.prop - burnRate * throttle * step);
        // g-load bookkeeping for the summary (thrust accel in g)
        this.peakG = Math.max(this.peakG, thrustN * throttle / m / 9.80665);
      }
      // orbit insertion: tangential speed within 0.5% of circular above
      // 140 km → impulsive circularization (apPhase 'coast'), then the
      // standard coast→TLI node logic takes over one period later.
      const rIns = this.pos.length();
      const vT = this.vel.dot(_oB3.copy(nHat).cross(_oUp.copy(this.pos).normalize()).normalize());
      const vRad2 = this.vel.lengthSq() - vT * vT;
      const vCircIns = Math.sqrt(MU / rIns);
      if (rIns - R > 140_000 && vT >= vCircIns * 0.995 && vRad2 < (300 * 300)) {
        // patched-conic circularization: set exactly circular speed
        const upI = _oUp.copy(this.pos).normalize();
        const tDirI = _oB3.copy(nHat).cross(upI).normalize();
        if (tDirI.dot(this.vel) < 0) tDirI.negate();
        this.vel.copy(tDirI).multiplyScalar(Math.sqrt(MU / rIns));
        // M11j: the demo OWNS the moon — reposition it to the ANTIPODE
        // of the actual insertion point so the TLI apogee (which lands
        // exactly opposite a prograde burn) hits the moon regardless of
        // where the ascent ended up. The camera watches the earth during
        // the LEO coast, so the one-time reposition is never on screen.
        this.moonC.copy(upI).multiplyScalar(-MOON_ORBIT_R);
        MOON.center.copy(this.moonC);
        this.apPhase = 'coast';
        this.orbT = 0;
        this.note = 'ORBIT';
        elementsOf(_oR.copy(this.pos).sub(this.primC), this.vel, this.primMu, this.el);
      }
      // insertion safety: if stage 2 ran out of prop below orbital speed,
      // fall back to ballistic (the coast rail keeps whatever orbit the
      // ascent achieved; TLI still fires on the node if the period > 0)
      if (this.prop <= 50 && this.apPhase === 'boost') {
        this.apPhase = 'coast';
        this.orbT = 0;
        this.note = 'BURNOUT';
        elementsOf(_oR.copy(this.pos).sub(this.primC), this.vel, this.primMu, this.el);
      }
      // M11w10: the cinematic ascent ends here — cut to the 250 km parking
      // orbit and hand the state to the PROVEN full-mission pipeline
      // (coast → TLI → SOI capture → powered descent, all verified by
      // M11/M11g). The camera is already tilted down at the planet, so the
      // cut reads as the planet receding rather than a snap.
      if (this.msStage === 2 && this.msT >= MS_S2_CLIMB_S && this.apPhase === 'boost') {
        this.mLand = Math.max(this.mLand - 4800, 4200);
        // parking orbit state: 250 km circular, in the moon's orbit
        // plane, at the +X node (antipode of the demo's moon)
        const si = Math.sin(INCLINATION), ci = Math.cos(INCLINATION);
        const rPark = R + 250_000;
        this.pos.set(rPark, 0, 0);
        // prograde tangent in-plane: ê2×ê1 (toward decreasing a — the TLI
        // node returns here one lap later)
        const nHatIns = _oB1.set(0, si, ci).normalize();
        const upIns = _oUp.set(1, 0, 0);
        const tanIns = _oRail.copy(nHatIns).cross(upIns).normalize();
        this.vel.copy(tanIns).multiplyScalar(Math.sqrt(MU / rPark));
        this.primC.set(0, 0, 0);
        this.primMu = MU;
        this.apPhase = 'coast';
        this.orbT = 0;
        this.note = 'ORBIT';
        // the ascent stack is GONE with the cut (classic stage disposal):
        // what remains is the lander + its budget. M11w10: 15 t — the
        // night-side deorbit (~1.1 t) sits on top of the proven descent
        // (~3.4-4.6 t) + TLI (~8.4 t), with margin to spare.
        this.prop = 15_000;
        elementsOf(_oR.copy(this.pos).sub(this.primC), this.vel, this.primMu, this.el);
        return;
      }
      this.orbT -= wdt; // boost time is not coast time
    } else if (this.lob && this.apPhase === 'boost') {
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
    } else {
      // M11 ?full=1 powered descent: engage above the horizon — braking
      // 2.3 km/s of orbital speed takes ~270 s at 8.6 m/s², so starting
      // at 20 km AGL impacts first. 120 km AGL gives ~350 s of fall time
      // (g_moon 1.62) — just enough. Everything higher coasts on the rail.
      const rAg = _oR.copy(this.pos).sub(this.primC);
      const aglNow = rAg.length() - (this.primMu === MU_MOON ? R_MOON : 0);
      const descendingNow = this.apPhase === 'lunar-orbit' && this.fullMission
        && this.primMu === MU_MOON && aglNow < 120_000;
      if (descendingNow) {
      // --- M11 powered descent --------------------------------------------
      // Captured into the 15 km-perilune ellipse: numeric thrust replaces
      // the rail from perilune approach until touchdown. Sliced like the
      // atmosphere block. Autopilot:
      //   1. above 20 km AGL: coast on the rail (handled below the else-if)
      //   2. below 20 km: null horizontal velocity first (retro-thrust on
      //      the horizontal component), then hold VS ≈ -8 m/s to touchdown.
      // The moon has no atmosphere, so this is pure rocket braking.
      const rRel = _oR.copy(this.pos).sub(this.moonC);
      const r = rRel.length();
      const upB = _oUp.copy(rRel).multiplyScalar(1 / r);
      const agl = r - R_MOON;
      const vRel = this.vel; // moon frozen: absolute = moon-relative
      const vsB = vRel.dot(upB);
      const vHoriz = _oB1.copy(vRel).addScaledVector(upB, -vsB);
      const gh = vHoriz.length();
      const slices = Math.min(Math.ceil(wdt / 0.25), 96);
      const step = wdt / slices;
      for (let i = 0; i < slices; i++) {
        const m = this.mLand + this.prop;
        const rrS = _oR.copy(this.pos).sub(this.moonC);
        const rS = rrS.length();
        const upS = _oUp.copy(rrS).multiplyScalar(1 / rS);
        const hSite = moonHeight(upS.x, upS.y, upS.z);
        // M11e fix: aglS is SPHERE-relative; the touchdown guard tests
        // TERRAIN-relative depth (surfR = R_MOON + hSite). On crater/
        // basin floors (hSite < 0) the old 'aglS < 0.5 → break' left the
        // craft hovering at terrain level forever: the rail re-engaged
        // each frame, no thrust, no touchdown ever detected. Land on the
        // terrain directly here instead.
        const aglTerrain = rS - (R_MOON + hSite);
        if (aglTerrain < 0.5) {
          this.pos.copy(upS).multiplyScalar(R_MOON + hSite).add(this.moonC);
          const vsTouch = this.vel.dot(upS);
          this.vel.set(0, 0, 0);
          this.frozen = true;
          this.note = vsTouch > -30 ? 'LANDED' : 'CRASHED';
          this.apPhase = 'off';
          this.thr = 0;
          // M11f: freeze the summary numbers
          this.touchdownVs = vsTouch;
          return;
        }
        const aglS = rS - R_MOON; // sphere-relative AGL (vs profile keying)
        const vsS = this.vel.dot(upS);
        const vTot = this.vel.length();
        const vhS = _oB1.copy(this.vel).addScaledVector(upS, -vsS);
        const ghS = vhS.length();
        const aMax = this.THRUST_LANDER / m;
        // suicide-burn check: stopping dv needed vs the height available
        //   dv_h = gh, dv_v = |vs| + sqrt(2*g*agl) budget... simple ladder:
        //   while high: burn horizontal only; below 3km or when gh small:
        //   tilt toward vertical braking.
        const thrust = _oV.set(0, 0, 0);
        const gHere = this.primMu / (rS * rS);
        if (aglS > 2000 && ghS > 20) {
          // braking phase: priority is killing the HORIZONTAL speed
          // (2.3 km/s needs ~270 s at full 8.6 m/s²; the fall from
          // 120 km takes ~344 s). Vertical needs only gravity support:
          // share up-thrust to cap sink rate, horizontal gets the rest.
          // up share: 0 at vs>0, grows as vs sinks past -60, capped 0.5
          const upShare = clamp((-vsS - 60) / 240, 0, 0.5);
          const hShare = Math.sqrt(1 - upShare * upShare);
          thrust.addScaledVector(vhS, -hShare * aMax / Math.max(ghS, 1e-6));
          // vertical PD: target a gentle controlled sink that grows with
          // altitude (from -12 m/s near the ground to free-fall high up)
          // sink profile: free-fall high up (-150 m/s), braking to a
          // gentle -8 only in the last ~2 km — holding a slow sink from
          // orbit burns the tank just to fight gravity for minutes
          const vsTarget = -clamp(aglS / 300, 8, 150);
          const vsErr = vsS - vsTarget;
          const aUp = clamp(gHere - vsErr * 0.08, 0, aMax * upShare * 1.4);
          thrust.addScaledVector(upS, aUp);
          thrust.setLength(Math.min(thrust.length(), aMax));
        } else {
          // terminal: null horizontal, control vertical toward -6 m/s
          const wH = clamp(ghS / 30, 0, 1);
          if (ghS > 0.5) thrust.addScaledVector(vhS, -wH * aMax / Math.max(ghS, 1e-6));
          // same sink profile as the braking branch: free-fall high,
          // -8 m/s only in the last ~2 km (prop is finite!)
          const vsErr = vsS + clamp(aglS / 300, 8, 150);
          // only UP-thrust in the terminal branch: vsErr<0 (sinking
          // faster than target) fires the engine; above target, gravity
          // pulls the craft back to the sink profile on its own
          const aV = (1 - wH) * aMax * clamp(-vsErr / 10, 0, 1);
          if (aV > 0) thrust.addScaledVector(upS, aV);
        }
        const tMag = thrust.length();
        // M11f: descent burn load (the moon has no drag — thrust IS the g load)
        this.peakG = Math.max(this.peakG, tMag / 9.80665);
        // empty tank = no engine (the burn must not run on fumes)
        if (tMag > 1e-3 && this.prop > 0) {
          const dm = Math.min(this.BURN_RATE * step, this.prop);
          // thrust components are ACCELERATIONS (built from aMax = T/m),
          // so dv = |a| * dt — the old tMag/m divided by mass twice and
          // the descent autopilot's thrust was ~8000x too weak
          this.vel.addScaledVector(thrust.normalize(), tMag * step);
          this.prop = Math.max(0, this.prop - dm * (tMag / aMax));
        }
        this.vel.addScaledVector(upS, -this.primMu / (rS * rS) * step);
        this.pos.addScaledVector(this.vel, step);
      }
      this.note = agl < 500 ? 'LANDING' : 'DESCENT';
      } else {
        const rail = _oRail.copy(this.pos).sub(this.primC);
        const railV = _oRailV.copy(this.vel);
        propagateKepler(rail, railV, this.primMu, wdt, _oNew, _oNewV);
        this.pos.copy(this.primC).add(_oNew);
        this.vel.copy(_oNewV);
      }
    }

    // --- mission phases -------------------------------------------------
    if (this.apPhase === 'coast') {
      // First coast: verify the rail (one lap) then ignite TLI at apoapsis
      // of the raise ellipse. For the demo we ignite after one full period.
      // ?notli=1 (M10.8 entry testing): stay in coast forever so a low-pe
      // orbit can dip into the atmosphere and reenter without the TLI burn
      // hijacking the trajectory mid-test.
      // M11j: the moonshot parks in a 150 km CIRCULAR orbit. A PROGRADE
      // burn makes the burn point the PERIGEE of the transfer and the
      // apoapsis lands on the OPPOSITE side — so TLI must fire at the
      // insertion point (+X), whose antipode (−X) is exactly where the
      // demo placed the moon. A short 2% arc keeps the ORBIT note
      // readable before ignition.
      // M11w10: the moonshot now watches the FULL parking lap at cinematic
      // warp (the look eases from the horizon down to the globe across it,
      // and the TLI node look continues seamlessly) — the old 2% arc made
      // the LEO coast a 0.3 s blink. The node clamp lands the burn exactly
      // on the insertion node either way.
      if (!this.noTli && this.orbT >= this.el.period && this.el.period > 0) {
        this.apPhase = 'tli';
        this.orbT = 0;
      }
    }

    const r = _oR.copy(this.pos).sub(this.primC);
    const rm = r.length();

    // --- SOI handoff (earth -> moon) ------------------------------------
    if (!this.inMoonSoi && !this.returnedFromMoon) {
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
        // M11 ?full=1: capture straight into the descent ellipse (rp 15 km
        // altitude, ra = entry radius) — the perilune pass then becomes the
        // landing burn instead of a separate circularization + descent.
        // M11w10 moonshot: capture into the 500 km-perilune ellipse instead
        // (the descent happens later via the night-side deorbit below — the
        // capture geometry alone always lands on the moon's NIGHT side,
        // because the perilune is antipodal to the sunward SOI entry point).
        const rpT = this.moonshot ? R_MOON + 500_000
          : this.fullMission ? R_MOON + 15_000 : R_MOON + 500_000;
        const aT = (rpT + rr) / 2;
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
          this.note = this.fullMission && !this.moonshot ? 'CAPTURED-DESCENT' : 'CAPTURED';
          if (this.fullMission) {
            this.lastR = rr;
            this.rTrendUp = false;
          }
        } else {
          this.apPhase = 'soi-moon';
          this.note = 'SOI MOON';
        }
      }
    }
    // M11w10 moonshot: night-side deorbit. The capture ellipse's perilune
    // (500 km) sits on the anti-sun side (antipodal to the sunward SOI
    // entry). A retro burn near that perilune keeps the burn point as the
    // new apoapsis and drops the OPPOSITE side to a 15 km perilune — 180°
    // around the orbit, on the sunlit hemisphere — so the dive and the
    // landing sweep around into daylight.
    if (this.moonshot && this.apPhase === 'lunar-orbit' && this.primMu === MU_MOON && !this.msDeorbit) {
      const rAgD = _oR.copy(this.pos).sub(this.moonC).length() - R_MOON;
      if (rAgD < 560_000) {
        const rRelD = _oR.copy(this.pos).sub(this.moonC);
        const rrD = rRelD.length();
        const aD = (rrD + R_MOON + 15_000) / 2;
        const vTgtD = Math.sqrt(MU_MOON * (2 / rrD - 1 / aD));
        const dvD = this.vel.length() - vTgtD;
        if (dvD > 0 && this.prop > 10) {
          this.vel.addScaledVector(_oV.copy(this.vel).normalize(), -dvD);
          const ve = this.THRUST_LANDER / this.BURN_RATE;
          this.prop = Math.max(0, this.prop - (this.mLand + this.prop) * (1 - Math.exp(-dvD / ve)));
          this.msDeorbit = true;
          this.note = 'DEORBIT';
        }
      }
    }

    // --- thrust (impulse only — the coast above already propagated) ------
    if (this.apPhase === 'tei') {
      // M11g TEI: wait until near the anti-earth point (the departure node
      // for a minimum-dv earthward transfer), then escape the moon SOI with
      // a small residual and re-anchor the orbit on the EARTH. The moon is
      // frozen (no orbital velocity), so the patched-conic exit velocity is
      // simply the apoapsis speed of the return ellipse: a tangential
      // 5.8 m/s at the moon's radius gives pe ≈ 25 km — the chute path
      // takes it from there.
      const rRelT = _oR.copy(this.pos).sub(this.moonC);
      const rT = rRelT.length();
      const antiEarth = -rRelT.x / rT; // +1 when on the far side from earth
      if (antiEarth > 0.95) {
        // apoapsis speed of the earth-return ellipse (ra = moon distance,
        // rp = R + 25 km)
        const ra = MOON_ORBIT_R;
        const rpE = R + 25_000;
        const aE = (ra + rpE) / 2;
        const vApo = Math.sqrt(MU * (2 / ra - 1 / aE));
        // moon-frame hyperbolic excess: matching the tangential vApo at
        // exit means v_inf ≈ vApo (the moon is static in this frame)
        const vLeave = Math.sqrt(vApo * vApo + 2 * MU_MOON / rT);
        const hV2 = _oV.copy(rRelT).cross(this.vel);
        const tDir2 = _oUp.copy(hV2).normalize().cross(rRelT).normalize();
        const dv = vLeave - this.vel.length();
        if (dv > 0 && this.prop > 10) {
          this.vel.addScaledVector(tDir2, dv);
          const ve = this.THRUST_LANDER / this.BURN_RATE;
          this.prop = Math.max(0, this.prop - (this.mLand + this.prop) * (1 - Math.exp(-dv / ve)));
        }
        // exit the SOI: hand the orbit to the EARTH with the exact patched-
        // conic solution (tangential vApo at the moon's radius)
        this.inMoonSoi = false;
        this.returnedFromMoon = true;
        this.primC.set(0, 0, 0);
        this.primMu = MU;
        const tangential = _oB1.copy(this.WORLD_Y).cross(_oUp.copy(this.pos).normalize()).normalize();
        this.vel.copy(tangential).multiplyScalar(-vApo); // retrograde: dives to the 25 km pe
        this.pos.copy(this.moonC).add(_oR.copy(this.pos).sub(this.moonC)); // unchanged, clarity
        elementsOf(_oR.copy(this.pos), this.vel, MU, this.el);
        this.apPhase = 'trans-earth';
        this.note = 'TEI DONE';
      }
    }
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
    // M10.8c parachute → M11i staged chutes: a 10.2 t lander on a 28 m²
    // shield alone still hits at ~60 m/s. Drogue first (small 35 m² canopy,
    // works at transonic speeds), then the 650 m² main below 7 km and
    // ~200 m/s — survivable ~14 m/s splashdown. Staged INSIDE the slice
    // loop (fresh altitude each 0.25 s piece) so a fast descent can't skip
    // past the gate between frames. Canopies inflate over ~1-2 s (ramped
    // area): a full main snapping open in one slice spiked peakG to ~435 g.
    const paraA = this.paraOpen ? 650.0 : 0.0;
    // M11j: while the moonshot stack is BOOSTING, the ascent ladder above
    // already integrates drag+gravity+position for this step — the
    // atmosphere block would integrate the SAME step a second time
    // (double gravity + double displacement = the craft slams back into
    // the pad). Skip entirely until the boost phase ends.
    if (this.moonshot && this.apPhase === 'boost') {
      // keep the telemetry fields consistent; no integration here
      this.heatFlux = 0;
      this.gLoad = 0;
      this.airDensity = isaDensity(Math.max(alt, 0));
    } else if (this.primMu === MU && alt < ATMOS_TOP) {
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
        // M11i staged chute gates (fresh altitude per slice):
        // drogue below 14 km / 700 m/s, main below 7 km / 200 m/s
        // M11j: never while the moonshot stack is still climbing
        const chutesArmed = !(this.moonshot && this.apPhase === 'boost');
        if (chutesArmed && !this.drogueOpen && descending && altS < 14000 && vS < 700) {
          this.drogueOpen = true;
          this.drogueT = 0;
          this.chuteEvents++;
          this.note = 'DROGUE';
        }
        if (chutesArmed && this.drogueOpen && !this.paraOpen && descending && altS < 7000 && vS < 200) {
          this.paraOpen = true;
          this.paraT = 0;
          this.chuteEvents++;
          this.note = 'PARACHUTE';
        }
        // canopy inflation ramps (drogue 1.2 s, main 1.8 s)
        if (this.drogueOpen && this.drogueT < 1) this.drogueT = Math.min(1, this.drogueT + step / 1.2);
        if (this.paraOpen && this.paraT < 1) this.paraT = Math.min(1, this.paraT + step / 1.8);
        // M10.8: once the plasma is hot the ablative heat shield deploys:
        // the bare lander hull has a small 5 m² attached area (ok for a
        // propulsive moon landing) but orbital entry needs a blunt shield —
        // scale to a Dragon-class 28 m² / Cd 1.5 when heating is significant.
        const shieldA = this.heat > 0.05 && descending ? 28.0 : 5.0;
        const shieldCd = this.heat > 0.05 && descending ? 1.5 : 1.2;
        // deceleration: drag on the shield + staged chutes (Cd*A sums);
        // chute areas scale with their inflation ramps
        const drogueA = this.drogueOpen ? 35.0 * this.drogueT : 0.0;
        const chuteA = paraA * this.paraT;
        const dragA = (shieldCd * shieldA + (descending ? 1.4 * (drogueA + chuteA) : 0))
          * 0.5 * rho * vS * vS / (this.mLand + this.prop);
        // stagnation heat flux (Sutton-Graves, k=1.7e-4, W/m^2) -> telemetry
        this.heatFlux = descending ? 1.7e-4 * Math.sqrt(rho) * vS * vS * vS : 0;
        this.gLoad = dragA / 9.80665;
        // M10.8c landing burn: the lander's engine (45 kN, TWR 0.45 earth)
        // can't hover but adds ~4.9 km/s of dv over the tank — enough to
        // finish what the shield started. Retro-thrust below 40 km, past
        // the plasma peak (heat<0.5: the flux itself keeps RISING through
        // the thick-air phase, so gating on flux would never open).
        // M11j: the moonshot ascent flies THROUGH the atmosphere — the
        // landing-burn retro thrust and the chute gates must not fire
        // while the boost phase is climbing.
        if (this.moonshot && this.apPhase === 'boost') {
          // ascending stack: no chutes, no retro; drag only (applied below)
        } else {
        if (this.prop > 0 && descending && altS < 40000 && this.heat < 0.5 && vS > 60) {
          const acc = this.THRUST_LANDER / (this.mLand + this.prop);
          this.vel.addScaledVector(this.vel, -Math.min(acc * step / Math.max(vS, 1e-6), 0.9));
          this.prop = Math.max(0, this.prop - this.BURN_RATE * step);
        }
        }
        this.vel.addScaledVector(this.vel, -dragA * step / Math.max(vS, 1e-6));
        // gravity during the slice (rail no longer carries it)
        this.vel.addScaledVector(upS, -this.primMu / (rr.length() * rr.length()) * step);
        this.pos.addScaledVector(this.vel, step);
        // peak heat drives the glow: normalized 0..1 over ~1 MW/m^2 with
        // a slow cool-down so the plasma persists through the peak region
        this.heat = Math.max(this.heat, clamp(this.heatFlux / 1e6, 0, 1));
        // M11f: mission summary peaks
        this.peakG = Math.max(this.peakG, this.gLoad);
        this.peakHeat = Math.max(this.peakHeat, this.heat);
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
      this.drogueOpen = false; // M11i
      this.frozen = true;
      // touchdown classification: chute terminal ~20 m/s lands intact,
      // anything faster is a crash
      this.note = vsTouch > -30 ? 'LANDED' : 'CRASHED';
      this.apPhase = 'off';
      // M11f: freeze the summary numbers
      this.touchdownVs = vsTouch;
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

  /**
   * M11f: post-flight summary lines for the HUD (shown while frozen after
   * a lob / full mission / reentry). Empty until a mission has ended.
   */
  missionSummary(): string[] {
    if (!this.frozen) return [];
    const mins = Math.floor(this.missionT / 60);
    const secs = Math.round(this.missionT % 60);
    const outcome = this.note === 'LANDED' ? 'MISSION COMPLETE' : 'MISSION FAILED';
    return [
      `-- ${outcome} (${this.note}) --`,
      `mission time ${mins}m ${secs.toString().padStart(2, '0')}s  prop remaining ${this.prop.toFixed(0)} kg`,
      `touchdown ${Math.abs(this.touchdownVs).toFixed(1)} m/s  peak ${this.peakG.toFixed(1)} g  peak heat ${(this.peakHeat * 100).toFixed(0)}%`,
    ];
  }

  /** Camera write for the orbital view: ride slightly behind/above, look
   * down the velocity vector tilted toward the surface so the planet fills
   * the frame; near plane tracks altitude like the lander cam. */
  private writeCameraOrbital(): void {
    const up = _oUp.copy(this.pos).sub(this.primC).normalize();
    // M11j: the moonshot demo owns the camera choreography — a scripted
    // sequence keyed on the mission phase (see moonshotLook()).
    if (this.moonshot) {
      if (this.moonshotLook()) return;
    }
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
    // Look target: nadir by default, but tilt toward the prograde horizon
    // when moving fast over the surface — during orbital flight / reentry /
    // descent braking the view shows where the craft is going (the plasma
    // core, the approaching terrain) instead of a static patch below.
    const belowAbs = _oB2.copy(this.pos).addScaledVector(up, -this.tAgl);
    const spd = this.vel.length();
    // M11e: blend the look elevation across gs 150-450 so the descent's
    // horizon-to-nadir transition eases instead of snapping at the 300 m/s
    // gate while the retro thrusters are still firing.
    const horizonBlend = clamp((spd - 150) / 300, 0, 1);
    const vh = _oV.copy(this.vel).addScaledVector(up, -this.vel.dot(up));
    if (horizonBlend > 0 && vh.lengthSq() > 1) {
      vh.normalize();
      // Aim at the SURFACE 60% of the way to the horizon (the horizon
      // lies sqrt(2 R h) away). Projecting the point onto the sphere
      // matters: a straight 'pos - up*k + vh*d' point floats above the
      // curved limb and the frame fills with space instead of terrain.
      const hd = Math.sqrt(2 * this.primC.distanceTo(this.pos) * Math.max(this.tAgl, 1));
      const lookDist = clamp(hd * 0.6, 2e3, 3e6);
      const aheadAbs = _oB2.copy(this.pos).addScaledVector(vh, lookDist);
      const surfaceR = this.primC.distanceTo(this.pos) - this.tAgl;
      aheadAbs.sub(this.primC).setLength(surfaceR).add(this.primC);
      // blend: at low blend the target slides from the ahead-point back
      // toward nadir along the same look ray (lerp the DIRECTION, not the
      // points, so the elevation angle interpolates cleanly)
      const aheadRel = _oB2.copy(aheadAbs).sub(this.pos);
      const nadirRel = _oB3.copy(belowAbs).sub(this.pos);
      aheadRel.lerp(nadirRel, 1 - horizonBlend);
      aheadAbs.copy(this.pos).add(aheadRel);
      this.world.rel(aheadAbs, _oRail);
    } else {
      this.world.rel(belowAbs, _oRail);
    }
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

  /**
   * M11j moonshot camera choreography — rewritten in M11w10 after the user
   * found the phase-keyed hard cuts jarring. Returns true when it wrote the
   * camera (the caller skips the standard orbital camera). The camera
   * POSITION always rides the craft (pos + 30 m up); only the LOOK target
   * is scripted, and every phase boundary is a continuation, not a cut:
   *   pre-launch hold + ascent — the horizon under the moon (the pad is on
   *                    the evening terminator with the frozen moon 20° up);
   *                    past the cloud deck the look eases toward the planet
   *                    (halfway to nadir by the stage-2 burnout)
   *   coast (LEO)    — the same ease continues to nadir across the lap; at
   *                    the TLI node nadir IS the moon bearing (the moon
   *                    hides behind the earth), so the transfer handoff is
   *                    seamless
   *   trans-lunar    — locked on the moon: the full earth disk slides out
   *                    of frame while the moon emerges from behind its limb
   *   lunar-orbit    — moon-locked until its angular radius passes 20°,
   *                    then eased toward the projected prograde horizon;
   *                    past 45° the standard descent camera takes over
   */
  private moonshotLook(): boolean {
    // ride position: craft + 30 m along up (same as the standard cam)
    const up = _oUp.copy(this.pos).sub(this.primC).normalize();
    const camAbs = _oB1.copy(this.pos).addScaledVector(up, 30);
    if (this.shake > 0.003) {
      const t = performance.now() / 1000;
      const right = _oRail.copy(up).cross(this.WORLD_Y).normalize();
      camAbs.addScaledVector(up, Math.sin(t * 61) * 0.6 * this.shake)
        .addScaledVector(right, Math.sin(t * 47 + 1.3) * 0.5 * this.shake);
    }
    this.world.rel(camAbs, this._tmp);
    this.rig.camera.position.copy(this._tmp);

    // -- look direction per phase (unit vector from the craft) ------------
    let lookDir: THREE.Vector3 | null = null;
    if (this.apPhase === 'boost' || this.apPhase === 'off') {
      // ASCENT: the horizon under the moon. The bearing is scripted (the
      // moon's azimuth) and fixed through the climb, so the moon stays
      // framed while the ground falls away; past the deck the look eases
      // down toward the planet.
      const si = Math.sin(INCLINATION), ci = Math.cos(INCLINATION);
      const mH = _oV.copy(this.moonC).sub(this.pos);
      mH.addScaledVector(up, -mH.dot(up));
      if (mH.lengthSq() > 1) {
        mH.normalize();
        // bias the bearing ~30° around the local up (a true in-plane
        // rotation) so the moon sits toward the upper RIGHT of the frame —
        // the debug HUD panel covers the upper left, and an on-axis moon
        // hides behind it (verified with the level camera: −side lands the
        // moon at x≈407 behind the panel; +side puts it at x≈873)
        {
          const bias = 30 * DEG;
          const side = _oB2.copy(up).cross(mH).normalize();
          mH.multiplyScalar(Math.cos(bias)).addScaledVector(side, Math.sin(bias)).normalize();
        }
        const dC = this.primC.distanceTo(this.pos);
        const hd = Math.sqrt(2 * dC * Math.max(this.tAgl, 1));
        const aheadAbs = _oB2.copy(this.pos).addScaledVector(mH, clamp(hd * 0.75, 2e3, 3e6));
        const surfaceR = dC - this.tAgl;
        aheadAbs.sub(this.primC).setLength(surfaceR).add(this.primC);
        lookDir = _oB3.copy(aheadAbs).sub(this.pos).normalize();
        // raise the look ~30% toward the moon itself: the pad's moon rides
        // 32° up, so the frame keeps both the moon (upper third) and the
        // horizon (lower half)
        const moonDir3 = _oV.copy(this.moonC).sub(this.pos).normalize();
        lookDir.lerp(moonDir3, 0.3).normalize();
        // tilt-down after the clouds: ~42% of the way to nadir by ~70 km —
        // kept under 50% so the planet's limb stays in frame at the orbit
        // cut (the user's "地球全体が見える" beat)
        const tilt = 0.42 * smoothstep(12, 70, this.tAgl / 1000);
        if (tilt > 0) {
          lookDir.lerp(_oV.copy(this.primC).sub(this.pos).normalize(), tilt).normalize();
        }
      }
    } else if (this.apPhase === 'coast') {
      // LEO: continue the same ease across the lap (k 0.5 → 1 by p 0.55).
      // The bearing stays the moon's azimuth for continuity with the
      // ascent (the velocity bearing would swing ~90° at the cut); the
      // end look (nadir) is exactly the moon bearing at the TLI node.
      const p = clamp(this.orbT / Math.max(this.el.period, 1), 0, 1);
      // k starts at 0.42 matching the ascent-end tilt (continuity), and eases
      // to full nadir by ~half the lap, well before the TLI node
      const k = 0.42 + 0.58 * smoothstep(0, 0.55, p);
      const mH = _oV.copy(this.moonC).sub(this.pos);
      mH.addScaledVector(up, -mH.dot(up));
      if (mH.lengthSq() > 1) {
        mH.normalize();
        const dC = this.primC.distanceTo(this.pos);
        const hd = Math.sqrt(2 * dC * Math.max(this.tAgl, 1));
        const aheadAbs = _oB2.copy(this.pos).addScaledVector(mH, clamp(hd * 0.75, 2e3, 3e6));
        aheadAbs.sub(this.primC).setLength(dC - this.tAgl).add(this.primC);
        lookDir = _oB3.copy(aheadAbs).sub(this.pos).normalize();
        lookDir.lerp(_oV.copy(this.primC).sub(this.pos).normalize(), k).normalize();
      }
    } else if (this.apPhase === 'trans-lunar' || (this.inMoonSoi && this.apPhase !== 'lunar-orbit')) {
      // TRANSLUNAR: locked on the moon — the earth starts dead-center in
      // this look (the moon hides behind it at the TLI node), slides out,
      // and the moon emerges from behind the limb as the craft rises.
      lookDir = _oV.copy(this.moonC).sub(this.pos).normalize();
    } else if (this.apPhase === 'lunar-orbit' && this.primMu === MU_MOON) {
      // LUNAR APPROACH: keep the moon-locked look while the moon is small,
      // then ease toward the surface horizon as it fills the frame. Past
      // 45° angular radius the standard camera's horizon/descent blend
      // takes over (return false = skip writing the camera here).
      const mDist = _oV.copy(this.moonC).sub(this.pos).length();
      const angDeg = Math.asin(Math.min(1, R_MOON / Math.max(mDist, R_MOON + 1))) / DEG;
      if (angDeg < 45) {
        const moonDir = _oV.copy(this.moonC).sub(this.pos).normalize();
        if (angDeg < 20) {
          lookDir = moonDir;
        } else {
          const vh = _oB2.copy(this.vel).addScaledVector(up, -this.vel.dot(up));
          if (vh.lengthSq() > 1) {
            vh.normalize();
            const dC = mDist;
            const hd = Math.sqrt(2 * dC * Math.max(this.tAgl, 1));
            const aheadAbs = _oB3.copy(this.pos).addScaledVector(vh, clamp(hd * 0.6, 2e3, 3e6));
            const surfaceR = dC - this.tAgl;
            aheadAbs.sub(this.moonC).setLength(surfaceR).add(this.moonC);
            lookDir = moonDir.lerp(_oB2.copy(aheadAbs).sub(this.pos).normalize(),
              smoothstep(20, 45, angDeg)).normalize();
          } else {
            lookDir = moonDir;
          }
        }
      }
    }
    if (!lookDir) return false; // lunar-orbit late/descent/tei: standard camera
    // M11w10 fix: _oB2 here — lookDir itself aliases _oB3 in the ascent and
    // coast branches, so building the target into _oB3 zeroed the direction
    // and the camera ended up looking at pos·1e6 ≈ straight up (the pad and
    // LEO views were sky-only and every ground tile frustum-culled).
    const lookAbs = _oB2.copy(this.pos).addScaledVector(lookDir, 1e6);
    this.world.rel(lookAbs, _oRail);
    // M11w10 fix: the roll reference must be the LOCAL radial up, not the
    // world Y axis — near the equator Y_world lies ALONG the ground (it is
    // the polar axis), so WORLD_Y rolled the camera ~90° (the HUD bank
    // read 80°+ on the pad and the moon sat sideways in frame).
    this.rig.camera.up.copy(up);
    this.rig.camera.lookAt(_oRail);
    const near = clamp(this.tAgl * 0.25 + 0.3, 0.3, 1e5);
    if (Math.abs(near - this.lastNear) / near > 0.3 || this.lastNear < 0) {
      this.lastNear = near;
      this.rig.camera.near = near;
      this.rig.camera.updateProjectionMatrix();
    }
    return true;
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
      // M11i: chute state readout (canopy inflation % when staged)
      const chutes = this.drogueOpen
        ? `  CH (D${Math.round(this.drogueT * 100)}%${this.paraOpen ? ` M${Math.round(this.paraT * 100)}%` : ''})`
        : '';
      return (
        `ORBIT(${body}) GS ${s.gs.toFixed(0)} m/s  ALT ${fmtM(s.agl)}  ` +
        `a ${fmtM(el.a)}  e ${el.e.toFixed(4)}  ` +
        `Pe ${fmtM(el.rp - (body === 'E' ? R : R_MOON))}  Ap ${apo}  ` +
        `T ${el.period > 0 ? (el.period / 60).toFixed(1) + 'min' : '--'}  ` +
        `PROP ${this.prop.toFixed(0)}kg  [${s.phase}${s.note ? ' ' + s.note : ''}]` +
        chutes + entry
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
const _oB3 = new THREE.Vector3();

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
