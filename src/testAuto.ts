import * as THREE from 'three';
import type { CameraRig } from './cameraRig';
import type { WorldOrigin } from './world';
import { moonPositionAtAngle } from './moonOrbit';
import { EARTH, MOON, localToAbsolute, eastAt } from './frames';
import { terrainHeight } from './terrain';

const PLANET_R = EARTH.radius;
const R_MOON = MOON.radius;
const DEG = Math.PI / 180;

/** One stop of the M11w20 tour (?demo=tour). Orientation uses the SAME URL
 * conventions as the hover spawn (?pitch: -90 = level horizon, ?hdg: spin
 * around the local zenith) so every keyframe is a verifiable hover pose. */
interface TourKey {
  body: 'earth' | 'moon';
  latDeg: number;
  lonDeg: number;
  /** Altitude above the sphere, m (the moon adds the terrain lift like the
   * hover spawn does; Earth only lifts when agl is set). */
  altM: number;
  agl: boolean;
  pitchDeg: number;
  hdgDeg: number;
  /** Frozen moon orbit angle (deg) while this keyframe is active. */
  moonDeg: number;
  /** Level-camera construction (views above the horizon): pitchDeg is the
   * view elevation above the horizon and hdgDeg spins it from the up x east
   * reference, with a level up vector (the tilt+spin path gains bank once
   * the tilt passes 90 deg). */
  levelView?: boolean;
  /** Seconds to hold at arrival. */
  holdS: number;
  /** Seconds of travel from the previous keyframe. */
  travelS: number;
}

/** The user's reference screenshots (M11w20 request): start on the beach
 * (image 1), pass the Earth-from-space pose (image 2), a close moon flyby
 * (image 3: alt 2.76 Mm, moon disc centered and fully in frame with the
 * Earth hidden behind/off-screen so its nav arrow clamps to the LEFT edge —
 * the screenshot's exact camera state is not reproducible from a hover URL
 * because it was framed after manual movement, so K1.5 matches the
 * COMPOSITION: nadir view over the dayside point (0,52) puts the disc
 * dead-center, fully lit, and the Earth ~52 deg off-axis — off-screen left,
 * so its nav arrow clamps to the left edge like the screenshot), finish hovering the moon surface
 * (image 4). Numbers are read off the screenshots' HUD lines. K0 is a
 * terrain-scan beach (low sand, trees left, water ahead-right — the default
 * spawn sits 900 m inland so it cannot reproduce image 1); its HUD heading
 * differs from the screenshot's 188 deg because the user's exact beach spot
 * is unknown — the COMPOSITION (sand, trees, ocean, clouds) is what is
 * matched. The K1 moon angle 230 deg puts the moon just above the Earth's
 * limb, left of up-screen, matching image 2; the moon angle then tweens
 * monotonically 230 -> 360 (K1.5, = 0 deg: hover geometry identical to the
 * screenshot's ?moonangle=0) -> 540 (= 180 deg, K2); K2 sits where the Earth
 * shows ~25 deg up at rig-heading 118 with the moon frozen at 180 deg
 * (scan: lat 44, lon 52). */
const TOUR: TourKey[] = [
  { body: 'earth', latDeg: 5.45, lonDeg: 20.75, altM: 45.9, agl: false,
    pitchDeg: 0.7, hdgDeg: 255, moonDeg: 0, holdS: 18, travelS: 0,
    levelView: true },
  { body: 'earth', latDeg: 15.8, lonDeg: 19.3, altM: 11_690_000, agl: false,
    pitchDeg: -82, hdgDeg: 80, moonDeg: 230, holdS: 8, travelS: 26 },
  { body: 'moon', latDeg: 0, lonDeg: 52, altM: 2_760_000, agl: false,
    pitchDeg: -89, hdgDeg: 0, moonDeg: 360, holdS: 8, travelS: 20 },
  { body: 'moon', latDeg: 44, lonDeg: 52, altM: 2770.8, agl: false,
    pitchDeg: 8.1, hdgDeg: -68, moonDeg: 540, holdS: Infinity, travelS: 30,
    levelView: true },
];

/**
 * Test instrumentation for verifying milestones without manual input.
 * All positions are handled in ABSOLUTE coordinates; the autopilot writes
 * frame-relative camera positions through WorldOrigin, so tests stay correct
 * while a floating origin is active (rebases happen mid-flight).
 *
 * URL params:
 *   ?demo=drop&alt0=25000000&alt1=100   radial descent from alt0 to alt1 (m)
 *   ?demo=hover&alt=200000              static view from altitude
 *   ?demo=bounce&alt0=25000000&alt1=100 repeated descend<->ascend cycles
 *                                       (exercises merge, LRU cache reuse,
 *                                       and split/merge chatter resistance)
 *   ?demo=look&alt=20000                scripted view-relative rotations from
 *                                       the nadir-looking pose: pitch up to
 *                                       horizon, yaw 90°, pitch through +100°
 *                                       (proves gimbal-free quaternion look)
 *   ?demo=orbit&alt=200000&span=deg     lateral arc flight at fixed altitude
 *                                       (floating-origin stress: the camera
 *                                       crosses rebase thresholds sideways)
 *   ?demo=tour                          M11w20 guided flythrough of the user's
 *                                       four reference screenshots: beach start
 *                                       -> Earth from 11.69 Mm (moon above the
 *                                       limb) -> moon flyby 2.76 Mm (Earth
 *                                       behind the disc) -> moon surface
 *                                       2770.8 m (Earth in the sky). Smoothstep
 *                                       position lines + slerped orientation;
 *                                       the frozen moon angle tweens between
 *                                       keyframe values. Overrides per key i:
 *                                       tourp<i> tourh<i> (URL-convention
 *                                       pitch/hdg) and tourm<i> (moon angle);
 *                                       touro=N starts at keyframe N, skipping
 *                                       earlier travel.
 * The autopilot only moves the camera; LOD behavior stays production code.
 */
export class AutoPilot {
  readonly mode: string;
  /** Flight mode suspends the test driver (the aircraft owns the pose). */
  private suspended = false;
  suspend(): void {
    this.suspended = true;
  }
  resume(): void {
    this.suspended = false;
  }
  private readonly alt0: number;
  private readonly alt1: number;
  private phase: 'descend' | 'ascend' | 'done' = 'descend';
  private cycles = 0;
  private lookPhase: 'A' | 'B' | 'C' | 'D' | 'E' = 'A';
  private h0 = 0;
  private orbitAngle = 0;
  private readonly world: WorldOrigin;
  /** Frozen moon orbit angle (rad) from ?moonangle, or null = live orbit.
   * The tour OWNS this value too: it tweens between keyframe angles. */
  moonAngle: number | null;
  // M11w20 tour state (keyframes with URL overrides resolved).
  private tourK: TourKey[] = [];
  private readonly tourPos: THREE.Vector3[] = [];
  private readonly tourQ: THREE.Quaternion[] = [];
  private tourIdx = 0;
  private tourHold = 0;
  private tourTraveling = false;
  private tourT = 0;
  private tourDur = 1;
  private tourFromMoon = 0;
  private tourToMoon = 0;
  private readonly tourFromPos = new THREE.Vector3();
  private readonly tourFromQ = new THREE.Quaternion();
  // Session spawn pose (absolute), captured after the constructor places the
  // camera — H ("home") restores exactly this spot and nadir orientation,
  // through any number of floating-origin rebases (stored absolute).
  private readonly homePos = new THREE.Vector3();
  private readonly homeQ = new THREE.Quaternion();

  constructor(rig: CameraRig, world: WorldOrigin) {
    const q = new URLSearchParams(location.search);
    // M11k: ?ground=1 — the EARTH surface with the horizon in view: the
    // free camera stands on the terrain (AGL ~1.7 m gear height) looking
    // at the horizon. Implies demo=hover (static pose at ground level),
    // pitch=-60 (horizon framed at the upper third), agl=1, alt=2 m. Set
    // BEFORE the mode/alt reads below.
    // M11k: a BARE url (no demo/alt/pitch/lat/lon/hdg) now defaults to the
    // same ground start — the old 25,000 km sphere-rail spawn showed the
    // earth as a small disc (technically correct, useless as a first
    // impression). Explicit params keep the historical behavior.
    const bare = !q.get('demo') && !q.has('alt') && !q.has('pitch')
      && !q.has('lat') && !q.has('lon') && !q.has('hdg');
    if ((q.get('ground') === '1' || bare) && q.get('body') !== 'moon') {
      if (!q.get('demo')) q.set('demo', 'hover');
      // pitch: the URL value wins when explicitly given (ground=1 only
      // sets the default); agl=1 + alt=2 m put the camera ON the terrain.
      if (!q.has('pitch')) q.set('pitch', '-60');
      // day-side default pad (the subsolar point) — the historical default
      // (5.5, -104) spawned on the night side: "opening the app shows
      // nothing" reports.
      if (!q.has('lat')) q.set('lat', '15.8');
      if (!q.has('lon')) q.set('lon', '19.3');
      q.set('agl', '1');
      if (!q.has('alt')) q.set('alt', '2');
    }
    this.mode = q.get('demo') ?? '';
    this.alt0 = num(q, 'alt0', 25_000_000);
    this.alt1 = num(q, 'alt1', 100);
    if (this.mode === 'hover' || this.mode === 'look' || this.mode === 'orbit') {
      const dflt = this.mode === 'hover' ? 200_000 : this.mode === 'look' ? 20_000 : 2_000_000;
      const alt = num(q, 'alt', dflt);
      this.alt0 = alt;
      this.alt1 = alt;
    }
    // M11n3b review fix: ?alt is authoritative in EVERY spawn mode. The old
    // demo-gated reassignment above silently ignored ?alt otherwise —
    // ?body=moon&alt=3000 spawned at the historical 25 Mm sphere-rail
    // altitude (the moon view rendered an all-black frame and the black-band
    // review could not even reproduce the horizon). Only alt0 is overridden:
    // drop/bounce keep their explicit/default alt1 (drop target) untouched.
    if (q.has('alt') && !q.has('alt0')) {
      this.alt0 = num(q, 'alt', this.alt0);
    }
    this.world = world;
    // ?moonangle=<deg> freezes the moon at an orbit angle (testing): the
    // main loop skips its time-based update when this is present.
    this.moonAngle = q.has('moonangle') ? num(q, 'moonangle', 0) * DEG : null;

    // M11w20: ?demo=tour — the guided flythrough of the user's reference
    // screenshots (beach -> Earth from 11.69 Mm -> moon flyby 2.76 Mm ->
    // moon surface). Keyframe pose/heading/moon-angle overrides come from
    // the URL so each stop can be tuned against its screenshot without
    // touching the source; touro=N starts the tour at keyframe N (skipping
    // earlier travel).
    if (this.mode === 'tour') {
      const tk = TOUR.map(k => ({ ...k }));
      const ov = (key: keyof TourKey, param: string): void => {
        const v = q.get(param);
        if (v !== null && Number.isFinite(Number(v))) {
          (tk[Number(param.slice(-1))] as unknown as Record<string, number>)[key] =
            Number(v);
        }
      };
      ov('pitchDeg', 'tourp0'); ov('hdgDeg', 'tourh0'); ov('moonDeg', 'tourm0');
      ov('pitchDeg', 'tourp1'); ov('hdgDeg', 'tourh1'); ov('moonDeg', 'tourm1');
      ov('pitchDeg', 'tourp2'); ov('hdgDeg', 'tourh2'); ov('moonDeg', 'tourm2');
      ov('pitchDeg', 'tourp3'); ov('hdgDeg', 'tourh3'); ov('moonDeg', 'tourm3');
      this.tourK = tk;
      const skip = Math.min(Math.max(num(q, 'touro', 0), 0), tk.length - 1);
      for (let i = 0; i <= skip; i++) {
        this.moonAngle = tk[i].moonDeg * DEG;
        this.tourPlace(rig, i);
      }
      this.tourIdx = skip;
      this.tourHold = tk[skip].holdS;
      return;
    }

    // ?body=moon relocates the spawn to lunar orbit (M9.2 test hook):
    // hover over the moon's surface at ?alt, nadir view, moon frozen at
    // ?moonangle. The moon's absolute center comes from the MOON frame
    // (M9.6) — the same registry entry the renderer and physics use.
    if (q.get('body') === 'moon') {
      const ang = this.moonAngle ?? 0;
      moonPositionAtAngle(ang, MOON.center);
      const lat2 = num(q, 'lat', 0);
      const lon2 = num(q, 'lon', 0);
      // radial position above the moon's near-side point via the SHARED
      // local-frame helper (M9.6: one lat/lon convention for both bodies)
      _pos.copy(localToAbsolute(MOON, lat2, lon2, this.alt0, _pos));
      // M11n9h: spawn ABOVE THE TERRAIN — alt0 is sphere-relative, and at
      // low ?alt the camera could end up inside a mountain (black frame:
      // the near plane clips the surrounding backfaces). Lift by the
      // terrain height at the spawn direction (+2 m clearance).
      {
        const dir = _pos.clone().sub(MOON.center).normalize();
        _pos.addScaledVector(dir, MOON.height(dir.x, dir.y, dir.z, 0) + 2);
      }
      this.world.origin.set(0, 0, 0);
      rig.camera.position.copy(_pos);
      rig.camera.up.set(0, 1, 0);
      rig.camera.lookAt(MOON.center);
      const pitchDeg2 = num(q, 'pitch', -90);
      // Local east on the moon via the SHARED convention (M9.6): ?pitch/?hdg
      // mean the same thing on both bodies. (?hdg was missing here entirely,
      // which blocked any sun/star-alignment test from the moon — M9.4.)
      const eastM = eastAt(lat2, lon2, _east);
      if (pitchDeg2 !== -90) {
        const qTilt = _qt.setFromAxisAngle(eastM, (pitchDeg2 + 90) * Math.PI / 180);
        rig.camera.quaternion.premultiply(qTilt);
      }
      const hdgDeg2 = num(q, 'hdg', 0);
      if (hdgDeg2 !== 0) {
        const upM = _east.copy(_pos).sub(MOON.center).normalize();
        const qHdg = _qt.setFromAxisAngle(upM, -hdgDeg2 * Math.PI / 180);
        rig.camera.quaternion.premultiply(qHdg);
      }
      if (q.get('level') === '0') rig.autoLevel = false;
      this.homePos.copy(_pos);
      this.homeQ.copy(rig.camera.quaternion);
      return;
    }
    // Start on +Z (absolute), looking straight down at the surface below.
    // ?lat/&lon (deg) relocate the hover point onto any spot on the globe —
    // used by terrain tests to hover over known landmasses.
    // ?pitch (deg) tilts the view up from nadir (0 = straight down, -90 =
    // horizon) — reproduces grazing-angle LOD screens.
    // ?agl=1: ?alt is measured from the TERRAIN (AGL) instead of the sphere —
    // low-alt tests at high-elevation sites put the camera inside the ground
    // otherwise (the sphere-based alt ignores terrainHeight at the hover
    // point). Default 0 keeps the historical sphere-based behavior.
    const lat = num(q, 'lat', 0);
    const lon = num(q, 'lon', 0);
    const agl = q.get('agl') === '1';
    let hoverAlt = this.alt0;
    if (agl) {
      const dir = localToAbsolute(EARTH, lat, lon, 1, _tmp).normalize();
      const ground = terrainHeight(dir.x, dir.y, dir.z);
      hoverAlt = Math.max(this.alt0 + ground, 2);
      this.alt0 = hoverAlt;
      this.alt1 = hoverAlt;
    }
    const pos = _pos.copy(localToAbsolute(EARTH, lat, lon, hoverAlt, _pos));
    this.world.origin.set(0, 0, 0);
    rig.camera.position.copy(pos);
    rig.camera.up.set(0, 1, 0);
    // Tilt first (around the local east axis), then lookAt keeps the intent:
    // lookAt would collapse a horizon pitch, so rotate the nadir quaternion.
    // ?hdg (deg) then spins the view around the local zenith to match any
    // manually-flown pose; ?level=0 disables auto-level (repro of manual flights).
    const pitchDeg = num(q, 'pitch', -90);
    rig.camera.lookAt(0, 0, 0);
    const east = eastAt(lat, lon, _east);
    if (pitchDeg !== -90) {
      const qTilt = _qt.setFromAxisAngle(east, (pitchDeg + 90) * Math.PI / 180);
      rig.camera.quaternion.premultiply(qTilt);
    }
    const hdgDeg = num(q, 'hdg', 0);
    if (hdgDeg !== 0) {
      const up = _pos.copy(pos).normalize();
      const qHdg = _qt.setFromAxisAngle(up, -hdgDeg * Math.PI / 180);
      rig.camera.quaternion.premultiply(qHdg);
    }
    if (q.get('level') === '0') rig.autoLevel = false;
    // Remember the spawn pose for the H (home) key.
    this.homePos.copy(pos);
    this.homeQ.copy(rig.camera.quaternion);
  }

  /** Teleport back to the session spawn pose (position + orientation). */
  goHome(rig: CameraRig): void {
    rig.camera.position.copy(this.world.rel(this.homePos, _pos));
    rig.camera.quaternion.copy(this.homeQ);
    rig.clearPendingTurn();
    // Pending autopilot phases keep running in other modes; for hover the
    // phase is already 'done', and drop/bounce re-place the camera next step
    // anyway. Position correctness does not depend on them.
  }

  /** @returns HUD status line, or null when inactive */
  update(rig: CameraRig, dt: number): string | null {
    if (!this.mode || this.suspended) return null;
    if (this.phase === 'done') return `autopilot:${this.mode} done`;

    const planetR = PLANET_R;
    // Absolute camera position (floating-origin aware).
    const absP = this.world.abs(rig.camera.position, _abs);
    const alt = absP.length() - planetR;

    if (this.mode === 'hover') {
      this.phase = 'done';
      return `autopilot:hover alt=${fmt(alt)}`;
    }

    if (this.mode === 'drop') {
      // Radial descent proportional to altitude (smooth LOD sweep), clamped.
      const speed = Math.min(Math.max(alt * 0.5, 50), 8_000_000);
      const newAlt = Math.max(alt - speed * dt, this.alt1);
      this.place(rig, planetR, newAlt);
      if (newAlt <= this.alt1 + 0.5) this.phase = 'done';
      return `autopilot:drop ${this.phase} alt=${fmt(newAlt)} target=${fmt(this.alt1)}`;
    }

    if (this.mode === 'bounce') {
      const speed = Math.min(Math.max(alt * 0.5, 50), 8_000_000);
      if (this.phase === 'descend') {
        const newAlt = Math.max(alt - speed * dt, this.alt1);
        this.place(rig, planetR, newAlt);
        if (newAlt <= this.alt1 + 0.5) this.phase = 'ascend';
        return `autopilot:bounce phase=down alt=${fmt(newAlt)} lo=${fmt(this.alt1)} hi=${fmt(this.alt0)}`;
      }
      const newAlt = Math.min(alt + speed * dt, this.alt0);
      this.place(rig, planetR, newAlt);
      if (newAlt >= this.alt0 - 0.5) {
        this.phase = 'descend';
        this.cycles++;
      }
      return `autopilot:bounce phase=up alt=${fmt(newAlt)} lo=${fmt(this.alt1)} hi=${fmt(this.alt0)} cycles=${this.cycles}`;
    }

    if (this.mode === 'orbit') {
      // Lateral arc at fixed altitude: crosses floating-origin rebase
      // thresholds sideways, the direction real flight actually travels.
      this.orbitAngle += dt;
      const a = this.orbitAngle;
      const r = planetR + this.alt0;
      _abs.set(r * Math.sin(a), 0, r * Math.cos(a));
      this.setAbs(rig, _abs, ORIGIN);
      return `autopilot:orbit ang=${a.toFixed(2)} alt=${fmt(alt)}`;
    }

    if (this.mode === 'look') {
      // Position stays fixed; only attitude changes, via the rig's own
      // view-relative rotation path (same code the mouse drives).
      // Fixed per-substep increment: dt-proportional steps would alias
      // through the zenith at high speedup (skipping transition windows).
      const la = rig.getLookAngles();
      const p1 = la.pitch.toFixed(1);
      const b1 = la.bank.toFixed(1);
      const h1 = la.heading.toFixed(1);
      if (this.lookPhase === 'A') {
        // From straight-down (the degenerate pose for Euler yaw/pitch rigs),
        // pitch up to the horizon.
        rig.turn(0, 0.02);
        if (la.pitch >= -4) {
          this.lookPhase = 'B';
          this.h0 = la.heading;
        }
        return `autopilot:look A nadir->horizon pitch=${p1} bank=${b1}`;
      }
      if (this.lookPhase === 'B') {
        // Yaw ~90° around the own head axis, at the horizon.
        rig.turn(0.02, 0);
        const dh = ((la.heading - this.h0 + 540) % 360) - 180;
        if (Math.abs(dh) >= 85) this.lookPhase = 'C';
        return `autopilot:look B yaw dh=${dh.toFixed(1)} pitch=${p1} bank=${b1} hdg=${h1}`;
      }
      if (this.lookPhase === 'C') {
        // Pitch up to the zenith.
        rig.turn(0, 0.02);
        if (la.pitch >= 89.3) this.lookPhase = 'D';
        return `autopilot:look C to-zenith pitch=${p1} bank=${b1} hdg=${h1}`;
      }
      if (this.lookPhase === 'D') {
        // Keep rotating the same way: pitch wraps over the top and descends
        // on the other side — impossible in an Euler clamped rig.
        rig.turn(0, 0.02);
        if (la.pitch <= 82) this.lookPhase = 'E';
        return `autopilot:look D through-zenith pitch=${p1} bank=${b1} hdg=${h1}`;
      }
      rig.turn(0, -0.02);
      if (la.pitch <= 0.5) {
        this.phase = 'done';
        return `autopilot:look done pitch=${p1} bank=${b1} hdg=${h1}`;
      }
      return `autopilot:look E return pitch=${p1} bank=${b1} hdg=${h1}`;
    }
    if (this.mode === 'tour') return this.tourUpdate(rig, dt);
    return `autopilot:${this.mode} (unknown)`;
  }

  /** Compute a tour keyframe pose (absolute position + orientation), using
   * the exact hover-spawn math so a keyframe is bit-identical to the
   * equivalent ?demo=hover URL pose. Leaves the world origin at (0,0,0) and
   * the camera AT the pose — callers either keep that state (constructor)
   * or restore the previous camera state (tourStart). */
  private computePose(rig: CameraRig, k: TourKey, outPos: THREE.Vector3, outQ: THREE.Quaternion): void {
    const isMoon = k.body === 'moon';
    const body = isMoon ? MOON : EARTH;
    // The moon keyframes pin the orbit angle: swap the frozen center in for
    // the placement (the live MOON.center is mid-tween during travel) and
    // restore it afterwards. The main loop rewrites it every frame anyway.
    const saveCenter = _tA.copy(MOON.center);
    if (isMoon) moonPositionAtAngle(k.moonDeg * DEG, MOON.center);
    const center = _tB.copy(body.center);
    let alt = k.altM;
    if (k.agl && !isMoon) {
      const dir = localToAbsolute(EARTH, k.latDeg, k.lonDeg, 1, _tC).normalize();
      alt = Math.max(k.altM + terrainHeight(dir.x, dir.y, dir.z), 2);
    }
    const pos = localToAbsolute(body, k.latDeg, k.lonDeg, alt, _tC).clone();
    if (isMoon) {
      // Same above-the-terrain lift as the hover spawn (a sphere-relative
      // alt can bury the camera in a mountain — M11n9h).
      const dir = pos.clone().sub(center).normalize();
      pos.addScaledVector(dir, MOON.height(dir.x, dir.y, dir.z, 0) + 2);
    }
    if (isMoon) MOON.center.copy(saveCenter);
    this.world.origin.set(0, 0, 0);
    rig.camera.position.copy(pos);
    rig.camera.up.set(0, 1, 0);
    if (k.levelView) {
      // Level-camera orientation for views ABOVE the horizon (K2): the
      // tilt+spin composition rolls the camera once the tilt passes 90 deg
      // (first verification showed bank 148 deg). Build the orientation from
      // the desired view direction — elevation = pitchDeg above the horizon,
      // bearing = the up x east reference spun by hdgDeg, the SAME bearing
      // convention the tilt+spin path produces (HUD heading = bearing+180) —
      // with a level up vector.
      const upL = pos.clone().sub(center).normalize();
      const east = eastAt(k.latDeg, k.lonDeg, _east).clone();
      const fwd = upL.clone().cross(east).normalize()
        .applyQuaternion(_qt.setFromAxisAngle(upL, -k.hdgDeg * DEG))
        .multiplyScalar(Math.cos(k.pitchDeg * DEG))
        .addScaledVector(upL, Math.sin(k.pitchDeg * DEG))
        .normalize();
      const upV = upL.clone().addScaledVector(fwd, -upL.dot(fwd)).normalize();
      rig.camera.quaternion.setFromRotationMatrix(
        new THREE.Matrix4().lookAt(ORIGIN, fwd, upV));
    } else {
      rig.camera.lookAt(center);
      const east = eastAt(k.latDeg, k.lonDeg, _east);
      if (k.pitchDeg !== -90) {
        rig.camera.quaternion.premultiply(_qt.setFromAxisAngle(east, (k.pitchDeg + 90) * DEG));
      }
      if (k.hdgDeg !== 0) {
        // Scratch vector — pos itself must stay intact (it is the pose output).
        const up = _tC.copy(pos).sub(center).normalize();
        rig.camera.quaternion.premultiply(_qt.setFromAxisAngle(up, -k.hdgDeg * DEG));
      }
    }
    outPos.copy(pos);
    outQ.copy(rig.camera.quaternion);
  }

  /** Compute and cache keyframe i's pose, leaving the camera there. */
  private tourPlace(rig: CameraRig, i: number): void {
    const pos = new THREE.Vector3();
    const qq = new THREE.Quaternion();
    this.computePose(rig, this.tourK[i], pos, qq);
    this.tourPos[i] = pos;
    this.tourQ[i] = qq;
  }

  /** Begin travel from the current camera state to keyframe i. */
  private tourStart(rig: CameraRig, i: number): void {
    this.tourFromPos.copy(this.world.abs(rig.camera.position, _tA));
    this.tourFromQ.copy(rig.camera.quaternion);
    this.tourFromMoon = this.moonAngle ?? 0;
    this.tourPlace(rig, i);
    this.tourToMoon = this.tourK[i].moonDeg * DEG;
    // Restore the camera to the from-state (computePose reset the origin to
    // 0, so rel() is the identity here; the loop's floating-origin rebase
    // re-centers immediately after update() returns).
    rig.camera.position.copy(this.world.rel(this.tourFromPos, _tC));
    rig.camera.quaternion.copy(this.tourFromQ);
    this.tourIdx = i;
    this.tourTraveling = true;
    this.tourT = 0;
    this.tourDur = this.tourK[i].travelS;
  }

  /** Tour state machine: hold at keyframe i, travel to i+1, repeat. */
  private tourUpdate(rig: CameraRig, dt: number): string {
    const last = this.tourK.length - 1;
    if (this.tourTraveling) {
      this.tourT += dt;
      const s = Math.min(this.tourT / this.tourDur, 1);
      const e = s * s * (3 - 2 * s); // smoothstep ease in/out
      _tA.lerpVectors(this.tourFromPos, this.tourPos[this.tourIdx], e);
      rig.camera.position.copy(this.world.rel(_tA, _tC));
      rig.camera.quaternion.slerpQuaternions(this.tourFromQ, this.tourQ[this.tourIdx], e);
      this.moonAngle = this.tourFromMoon + (this.tourToMoon - this.tourFromMoon) * e;
      if (s < 1) {
        return `autopilot:tour ${this.tourIdx + 1}/${this.tourK.length} travel ${(s * 100).toFixed(0)}%`;
      }
      this.tourTraveling = false;
      this.tourHold = this.tourK[this.tourIdx].holdS;
    }
    // Hold: re-apply the keyframe pose every frame (origin-independent).
    rig.camera.position.copy(this.world.rel(this.tourPos[this.tourIdx], _tC));
    rig.camera.quaternion.copy(this.tourQ[this.tourIdx]);
    if (this.tourIdx < last) {
      this.tourHold -= dt;
      if (this.tourHold <= 0) {
        this.tourStart(rig, this.tourIdx + 1);
        return `autopilot:tour ${this.tourIdx + 1}/${this.tourK.length} depart`;
      }
      return `autopilot:tour ${this.tourIdx + 1}/${this.tourK.length} hold ${this.tourHold.toFixed(0)}s`;
    }
    return `autopilot:tour ${this.tourK.length}/${this.tourK.length} arrived`;
  }

  /** Place the camera at an absolute radial position (nadir view). */
  private place(rig: CameraRig, planetR: number, alt: number): void {
    _tmp.set(0, 0, planetR + alt);
    this.setAbs(rig, _tmp, ORIGIN);
  }

  /** Set camera (frame-relative) from an absolute position + look target. */
  private setAbs(rig: CameraRig, absPos: THREE.Vector3, lookAtAbs: THREE.Vector3): void {
    rig.camera.position.copy(this.world.rel(absPos, _rel));
    rig.camera.lookAt(this.world.rel(lookAtAbs, _relLook));
  }
}

const ORIGIN = new THREE.Vector3(0, 0, 0);
const _abs = new THREE.Vector3();
const _tmp = new THREE.Vector3();
const _pos = new THREE.Vector3();
const _east = new THREE.Vector3();
const _qt = new THREE.Quaternion();
const _rel = new THREE.Vector3();
const _relLook = new THREE.Vector3();
const _tA = new THREE.Vector3();
const _tB = new THREE.Vector3();
const _tC = new THREE.Vector3();

const num = (q: URLSearchParams, k: string, d: number): number => {
  const v = q.get(k);
  if (v === null) return d;
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};

const fmt = (m: number): string =>
  m >= 1e6 ? (m / 1e6).toFixed(1) + 'Mm' : m >= 1e3 ? (m / 1e3).toFixed(1) + 'km' : m.toFixed(1) + 'm';
