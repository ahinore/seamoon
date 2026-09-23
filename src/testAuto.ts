import * as THREE from 'three';
import type { CameraRig } from './cameraRig';

/**
 * Test instrumentation for verifying milestones without manual input.
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
  private readonly radialDir: THREE.Vector3;

  constructor(rig: CameraRig) {
    const q = new URLSearchParams(location.search);
    this.mode = q.get('demo') ?? '';
    this.alt0 = num(q, 'alt0', 25_000_000);
    this.alt1 = num(q, 'alt1', 100);
    if (this.mode === 'hover') {
      const alt = num(q, 'alt', 200_000);
      this.alt0 = alt;
      this.alt1 = alt;
    }
    if (this.mode === 'look') {
      const alt = num(q, 'alt', 20_000);
      this.alt0 = alt;
      this.alt1 = alt;
    }
    // Start on +Z, looking straight down at the surface point below.
    this.radialDir = new THREE.Vector3(0, 0, 1);
    rig.camera.position.copy(this.radialDir).multiplyScalar(6_371_000 + this.alt0);
    rig.camera.up.set(0, 1, 0);
    rig.camera.lookAt(0, 0, 0);
  }

  /** @returns HUD status line, or null when inactive */
  update(rig: CameraRig, dt: number): string | null {
    if (!this.mode) return null;
    if (this.phase === 'done') return `autopilot:${this.mode} done`;

    const planetR = 6_371_000;
    const p = rig.camera.position;
    const alt = p.length() - planetR;

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

  private place(rig: CameraRig, planetR: number, alt: number): void {
    rig.camera.position.copy(this.radialDir).multiplyScalar(planetR + alt);
    rig.camera.lookAt(0, 0, 0);
  }
}

const num = (q: URLSearchParams, k: string, d: number): number => {
  const v = q.get(k);
  if (v === null) return d;
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};

const fmt = (m: number): string =>
  m >= 1e6 ? (m / 1e6).toFixed(1) + 'Mm' : m >= 1e3 ? (m / 1e3).toFixed(1) + 'km' : m.toFixed(1) + 'm';
