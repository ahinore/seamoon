import * as THREE from 'three';
import { buildTileGeometry, EARTH_BODY, type BodySurface } from './tileGeometry';
import { MOON_BODY } from './moonBody';

/**
 * Tile-builder worker (M10.1) — terrain tile geometry moves OFF the main
 * thread. Runs the exact same buildTileGeometry() the synchronous path used
 * (single source of truth: CPU/GPU agreement and LOD octave fading are
 * unchanged), then transfers the raw typed arrays back. The main thread
 * re-wraps them into a THREE.BufferGeometry (attribute upload is the only
 * main-thread cost left, ~0.3 ms/tile).
 *
 * Protocol (request → response):
 *   { seq, face, level, ix, iy, radius, res, body: 'earth' | 'moon' }
 * → { seq, position, normal, center, aGrid, color, index,
 *     centerAbs: [x,y,z], triangles }   (all buffers transferred)
 *
 * Cancel protocol: a job whose result arrives after the requesting node was
 * disposed is simply dropped by the pool (no round-trip cost — cancellation
 * happens on the main side, the worker always finishes what it started).
 *
 * Sea tiles stay on the main thread deliberately: buildSeaGeometry samples a
 * coarse 5x5 lattice (~2 ms) and its depth attribute feeds shore shading —
 * the complexity of a second worker protocol is not justified.
 */

interface Req {
  seq: number;
  face: number;
  level: number;
  ix: number;
  iy: number;
  radius: number;
  res: number;
  body: 'earth' | 'moon';
}

const BODIES: Record<string, BodySurface> = {
  earth: EARTH_BODY,
  moon: MOON_BODY,
};

// Minimal dedicated-worker scope shape (the DOM lib does not declare
// DedicatedWorkerGlobalScope in this project's tsconfig lib set).
interface WorkerScope {
  onmessage: ((e: MessageEvent) => void) | null;
  postMessage(msg: unknown, transfer?: Transferable[]): void;
}

const ctx = self as unknown as WorkerScope;

ctx.onmessage = (e: MessageEvent<Req>) => {
  const r = e.data;
  const body = BODIES[r.body] ?? EARTH_BODY;
  const built = buildTileGeometry(r.face, r.level, r.ix, r.iy, r.radius, r.res, body);
  const g = built.geometry;
  const pos = g.getAttribute('position').array as Float32Array;
  const nrm = g.getAttribute('normal').array as Float32Array;
  const cen = g.getAttribute('center').array as Float32Array;
  const grd = g.getAttribute('aGrid').array as Float32Array;
  const col = g.getAttribute('color').array as Float32Array;
  const idx = g.getIndex()!.array as Uint32Array;
  const c = built.center;
  const resp: Resp = {
    seq: r.seq,
    position: pos, normal: nrm, center: cen, aGrid: grd, color: col, index: idx,
    centerAbs: [c.x, c.y, c.z],
    triangles: idx.length / 3,
  };
  // Transfer everything — zero-copy handoff back to the main thread.
  ctx.postMessage(resp, [
    pos.buffer, nrm.buffer, cen.buffer, grd.buffer, col.buffer, idx.buffer,
  ] as unknown as Transferable[]);
};

interface Resp {
  seq: number;
  position: Float32Array;
  normal: Float32Array;
  center: Float32Array;
  aGrid: Float32Array;
  color: Float32Array;
  index: Uint32Array;
  centerAbs: [number, number, number];
  triangles: number;
}

// Keep THREE in the bundle graph (tileGeometry imports it); the worker never
// constructs scene objects.
void THREE;
