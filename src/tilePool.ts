import * as THREE from 'three';
import { TileMesh, type TransferredTile } from './tileMesh';
import type { BodySurface } from './tileGeometry';

/**
 * TilePool (M10.1) — asynchronous terrain-tile generation on a Web Worker.
 *
 * Replaces the synchronous time-budgeted builder for TERRAIN tiles (sea
 * tiles stay synchronous — they cost ~2 ms and need no second protocol).
 *
 * Why a worker: buildTileGeometry costs ~13 ms/tile at res 65. The old
 * time-budget (6 ms/frame) bounded the stall but still stole main-thread
 * time every frame during heavy streaming (descents, orbital re-approach);
 * the worker moves 100% of that off-thread. Main-thread cost per tile drops
 * to the attribute upload (~0.3 ms).
 *
 * Architecture:
 *  - `request()` enqueues a job; the pool keeps at most `maxInFlight`
 *    (default 4) running on ONE worker. Terrain build is pure CPU — one
 *    worker already saturates a core; more workers only add contention.
 *  - Results arrive tagged with the QNode that asked; if that node died or
 *    re-split meanwhile the tile goes to the LRU cache instead (it is fully
 *    built and may be re-requested soon).
 *  - Node disposal does NOT cancel in-flight jobs (no protocol for it) —
 *    wasted work is bounded by maxInFlight and the result is cached anyway.
 *
 * Priority (strategy-note M10.1): the PlanetView queue is sorted by camera
 * distance BEFORE requests are issued, and `lookahead` (a unit velocity
 * direction estimated from successive camera positions) biases the sort key:
 * tiles ahead of the motion are pulled ~1 edge forward in priority, so
 * forward flight rarely stalls on geometry that split mid-frame.
 */

interface Job {
  // identity of the requester for staleness checks
  key: string;
  face: number;
  level: number;
  ix: number;
  iy: number;
}

export interface TilePoolOptions {
  /** Parallel jobs allowed in flight (worker is single; queue depth). */
  maxInFlight?: number;
}

type Done = (tile: TileMesh) => void;

interface PendingDone {
  fn: Done;
  material: THREE.Material;
}

export class TilePool {
  private worker: Worker;
  private inflight = new Map<number, { nodeKey: string; face: number; level: number; ix: number; iy: number; radius: number; res: number; bodyKey: string }>();
  private seq = 0;
  private waiting: { fn: Done; job: Job }[] = [];
  readonly maxInFlight: number;
  /** Stats for the HUD: jobs finished on the worker. */
  built = 0;
  // Last request's build parameters — onResult re-pumps with them so the
  // waiting queue drains even when NO new request() arrives this frame
  // (a stalled pump here freezes the whole tile pipeline: observed live as
  // `queue 20 / built 10 (wk 4)` forever).
  private lastRadius = 0;
  private lastRes = 0;
  private lastMaterial: THREE.Material | null = null;
  private lastBodyKey = 'earth';

  constructor(opts: TilePoolOptions = {}) {
    this.maxInFlight = opts.maxInFlight ?? 4;
    // Vite resolves this to a bundled module worker.
    this.worker = new Worker(new URL('./tileWorker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<WorkerResp>) => this.onResult(e.data);
  }

  /**
   * Ask for a tile. Returns immediately; `done` fires on a later frame with
   * the mesh (main thread only wraps the transferred typed arrays into a
   * BufferGeometry — the cheap part).
   */
  request(
    nodeKey: string,
    face: number, level: number, ix: number, iy: number,
    radius: number, res: number,
    material: THREE.Material, bodyKey: string,
    done: Done,
  ): void {
    const job: Job = { key: nodeKey, face, level, ix, iy };
    this.waiting.push({ fn: done, job });
    this.pump(radius, res, material, bodyKey);
  }

  private pump(radius: number, res: number, material: THREE.Material, bodyKey: string): void {
    this.lastRadius = radius;
    this.lastRes = res;
    this.lastMaterial = material;
    this.lastBodyKey = bodyKey;
    this.dispatch();
  }

  private dispatch(): void {
    const radius = this.lastRadius, res = this.lastRes;
    const material = this.lastMaterial!, bodyKey = this.lastBodyKey;
    while (this.inflight.size < this.maxInFlight && this.waiting.length > 0) {
      const w = this.waiting.shift()!;
      const seq = ++this.seq;
      const j = w.job;
      this.inflight.set(seq, {
        nodeKey: j.key, face: j.face, level: j.level, ix: j.ix, iy: j.iy,
        radius, res, bodyKey,
      });
      // Remember the completion closure for this seq.
      this.pendingDone.set(seq, { fn: w.fn, material });
      this.worker.postMessage({
        seq, face: j.face, level: j.level, ix: j.ix, iy: j.iy,
        radius, res, body: bodyKey,
      });
    }
  }

  private pendingDone = new Map<number, PendingDone>();

  private onResult(r: WorkerResp): void {
    const meta = this.inflight.get(r.seq);
    const done = this.pendingDone.get(r.seq);
    this.inflight.delete(r.seq);
    this.pendingDone.delete(r.seq);
    if (!meta || !done) return;
    const t = TileMesh.fromTransfer(
      meta.face, meta.level, meta.ix, meta.iy, meta.radius, meta.res,
      done.material, r,
    );
    this.built++;
    done.fn(t);
    // Refill the pipeline: without this the pump only ran when a NEW request
    // arrived, so after the first 4 completions the waiting queue stalled
    // forever (live-observed: `queue 20  built 10 (wk 4)` frozen).
    this.dispatch();
  }

  /** In-flight count for stats/HUD. */
  get pending(): number {
    return this.inflight.size + this.waiting.length;
  }
}

interface WorkerResp {
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

// THREE is used for the geometry wrap below.
void THREE;
