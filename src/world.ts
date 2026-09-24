import * as THREE from 'three';

/**
 * Floating-origin bookkeeping (Phase 2).
 *
 * `origin` is the absolute-space position of the current frame's coordinate
 * origin. Everything double-precision (camera math, tile centers, LOD
 * decisions) works in ABSOLUTE coordinates; only what crosses into float32
 * (GPU matrices, and future float32 systems like physics/particles) works in
 * frame-relative coordinates, which stay small because the origin recenters
 * onto the camera whenever it drifts more than a threshold.
 *
 * Tile centers are kept absolute and immutable, so a rebase never invalidates
 * cached tile geometry — only render-time translation vectors change.
 */
export class WorldOrigin {
  readonly origin = new THREE.Vector3();
  /** Number of recenterings applied (diagnostics). */
  rebaseCount = 0;

  /** Absolute position of a frame-relative point. */
  abs(rel: THREE.Vector3, out: THREE.Vector3): THREE.Vector3 {
    return out.copy(this.origin).add(rel);
  }

  /** Frame-relative position of an absolute point. */
  rel(absP: THREE.Vector3, out: THREE.Vector3): THREE.Vector3 {
    return out.copy(absP).sub(this.origin);
  }

  /** Frame-relative position of the planet center (absolute 0,0,0). */
  centerRel(out: THREE.Vector3): THREE.Vector3 {
    return out.copy(this.origin).negate();
  }

  /**
   * Recenter the frame origin onto the camera if it drifted too far.
   * @returns the applied shift in frame coordinates (old camera position),
   *          or null when no rebase happened.
   */
  rebase(camPos: THREE.Vector3, threshold: number): THREE.Vector3 | null {
    if (camPos.lengthSq() <= threshold * threshold) return null;
    const shift = camPos.clone();
    this.origin.add(shift);
    camPos.set(0, 0, 0);
    this.rebaseCount++;
    return shift;
  }
}
