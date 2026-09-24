import * as THREE from 'three';
import type { CameraRig } from './cameraRig';
import type { WorldOrigin } from './world';

const PLANET_R = 6_371_000;

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
 * The autopilot only moves the camera; LOD behavior stays production code.
 */
export class AutoPilot {
  readonly mode: string;
  private readonly alt0: number;
  private readonly alt1: number;
  private phase: 'descend' | 'ascend' | 'done' = 'descend';
  private cycles = 0;
  private lookPhase: 'A' | 'B' | 'C' | 'D' | 'E' = 'A';
  private h0 = 0;
  private orbitAngle = 0;
  private readonly world: WorldOrigin;
  // Session spawn pose (absolute), captured after the constructor places the
  // camera — H ("home") restores exactly this spot and nadir orientation,
  // through any number of floating-origin rebases (stored absolute).
  private readonly homePos = new THREE.Vector3();
  private readonly homeQ = new THREE.Quaternion();

  constructor(rig: CameraRig, world: WorldOrigin) {
    const q = new URLSearchParams(location.search);
    this.mode = q.get('demo') ?? '';
    this.alt0 = num(q, 'alt0', 25_000_000);
    this.alt1 = num(q, 'alt1', 100);
    if (this.mode === 'hover' || this.mode === 'look' || this.mode === 'orbit') {
      const dflt = this.mode === 'hover' ? 200_000 : this.mode === 'look' ? 20_000 : 2_000_000;
      const alt = num(q, 'alt', dflt);
      this.alt0 = alt;
      this.alt1 = alt;
    }
    this.world = world;
    // Start on +Z (absolute), looking straight down at the surface below.
    // ?lat/&lon (deg) relocate the hover point onto any spot on the globe —
    // used by terrain tests to hover over known landmasses.
    // ?pitch (deg) tilts the view up from nadir (0 = straight down, -90 =
    // horizon) — reproduces grazing-angle LOD screens.
    const lat = num(q, 'lat', 0) * Math.PI / 180;
    const lon = num(q, 'lon', 0) * Math.PI / 180;
    const pos = _pos.set(
      Math.cos(lat) * Math.cos(lon),
      Math.sin(lat),
      Math.cos(lat) * Math.sin(lon),
    ).multiplyScalar(PLANET_R + this.alt0);
    this.world.origin.set(0, 0, 0);
    rig.camera.position.copy(pos);
    rig.camera.up.set(0, 1, 0);
    // Tilt first (around the local east axis), then lookAt keeps the intent:
    // lookAt would collapse a horizon pitch, so rotate the nadir quaternion.
    const pitchDeg = num(q, 'pitch', -90);
    rig.camera.lookAt(0, 0, 0);
    if (pitchDeg !== -90) {
      const east = _east.set(
        -Math.sin(lon),
        0,
        Math.cos(lat) * Math.cos(lon),
      ).normalize();
      const qTilt = _qt.setFromAxisAngle(east, (pitchDeg + 90) * Math.PI / 180);
      rig.camera.quaternion.premultiply(qTilt);
    }
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
    if (!this.mode) return null;
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
    return `autopilot:${this.mode} (unknown)`;
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

const num = (q: URLSearchParams, k: string, d: number): number => {
  const v = q.get(k);
  if (v === null) return d;
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};

const fmt = (m: number): string =>
  m >= 1e6 ? (m / 1e6).toFixed(1) + 'Mm' : m >= 1e3 ? (m / 1e3).toFixed(1) + 'km' : m.toFixed(1) + 'm';
