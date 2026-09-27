import * as THREE from 'three';
import { TileMesh } from './tileMesh';
import { buildTileGeometry, cubeToSphereDir, EARTH_BODY, type BodySurface } from './tileGeometry';
import { MOON_BODY } from './moonBody';
import { buildSeaGeometry } from './seaGeometry';
import { TilePool } from './tilePool';

export interface PlanetOptions {
  /** Max quadtree depth. 17+ reaches ~1 m vertex spacing on Earth radius. */
  maxLevel?: number;
  /** Split threshold in pixels of screen-space error. */
  tauPx?: number;
  /** Vertex grid resolution per tile side (excluding skirt ring). */
  res?: number;
  /** Tile geometry cache size (LRU) to avoid rebuilds on re-approach. */
  cacheSize?: number;
  /** Max tile builds per frame (synchronous in M1; moves to a Worker in M3). */
  buildBudget?: number;
  /**
   * Wall-clock time budget for tile builds per update call (ms). Tile
   * generation is CPU-bound, so this — not a build count — is what keeps
   * frame spikes bounded. Replaces the old fixed-count budget.
   */
  buildBudgetMs?: number;
  /**
   * Screen-footprint cap in pixels: a tile spanning more than this on screen
   * is split regardless of curvature error. Keeps the region near the camera
   * refined at grazing angles, where a smooth sphere has almost no sag error
   * but terrain (Phase 3) will need real detail. Children halve their
   * footprint per level, so the merge threshold (0.1*tau) stays chatter-free.
   */
  /** Screen-footprint cap in pixels (see capPx docs). */
  capPx?: number;
  /**
   * Phase 7: sea mode — tiles are the OCEAN SHELL (smooth sphere at sea
   * level + per-vertex water-depth attribute). Use with a sea material and
   * seaLevel radius; coarser settings are fine (no terrain displacement).
   */
  seaMode?: boolean;
}

export interface LodStats {
  visibleTiles: number;
  triangles: number;
  maxVisibleLevel: number;
  pending: number;
  cached: number;
  built: number;
  evicted: number;
  cacheHits: number;
  /** M10.1: tiles finished on the worker (vs synchronous fallback). */
  workerBuilt: number;
}

interface QNode {
  face: number;
  level: number;
  ix: number;
  iy: number;
  /** Tile center on the sphere (double, meters). */
  center: THREE.Vector3;
  /** Bounding sphere radius (with margin). */
  boundRadius: number;
  /** Tile normal (unit) — used for true sphere-patch distance. */
  nx: number;
  ny: number;
  nz: number;
  /** Geometric error (curvature sagitta) driving the split decision. */
  geomError: number;
  /** Max edge length (m) — drives the screen-footprint split rule. */
  edgeLen: number;
  children: QNode[] | null;
  tile: TileMesh | null;
  dead: boolean;
}

const keyOf = (n: QNode) => `${n.face}/${n.level}/${n.ix}/${n.iy}`;

/**
 * Cube-sphere quadtree LOD planet.
 *
 * Split rule: screen-space error rho = geomError / distance * pxPerUnit > tauPx.
 * A node keeps rendering its own tile until ALL 4 children have tiles ready,
 * so there are never holes during asynchronous (here: budgeted) generation.
 * Merge threshold is 0.1 * tauPx: quadtree children have 1/4 the error of
 * their parent, so the merge threshold must sit below tau/4 to make the
 * split/merge hysteresis band [0.4*tau, tau] chatter-free.
 */
export class PlanetView {
  readonly root = new THREE.Group();
  readonly stats: LodStats = {
    visibleTiles: 0, triangles: 0, maxVisibleLevel: 0,
    pending: 0, cached: 0, built: 0, evicted: 0, cacheHits: 0,
    workerBuilt: 0,
  };
  /**
   * Phase 9: this body's center in ABSOLUTE coordinates. Earth stays at
   * (0,0,0); the moon's group node is translated to its orbital position
   * each frame and this field feeds the placement math. Frame-relative
   * tile placement is always `center + bodyCenter - origin` in double.
   */
  bodyCenter = new THREE.Vector3();
  /** The surface functions this planet's tiles sample. */
  private readonly body: BodySurface;

  private readonly o: Required<Omit<PlanetOptions, 'seaMode'>> & { seaMode: boolean };
  private readonly radius: number;
  private readonly material: THREE.Material;
  private readonly roots: QNode[] = [];
  private readonly cache = new Map<string, TileMesh>();
  private readonly queue: QNode[] = [];
  private readonly camPos = new THREE.Vector3();
  private readonly originV = new THREE.Vector3();
  private readonly _absC = new THREE.Vector3();
  private pxPerUnit = 1;
  private readonly frustum = new THREE.Frustum();
  private readonly sphere = new THREE.Sphere();
  private readonly projScreen = new THREE.Matrix4();
  // ---- M10.1: async worker pool + motion-lookahead priority ----
  private readonly pool = new TilePool({ maxInFlight: 4 });
  /** ?noworker=1 forces the old synchronous path (A/B diagnosis). */
  private readonly syncTiles: boolean;
  /** Worker body selector: terrain functions live in the worker bundle. */
  private readonly bodyKey: 'earth' | 'moon';
  /** EMA of camera velocity direction (unit) for queue lookahead. */
  private readonly lookAhead = new THREE.Vector3();
  private readonly lookAheadM = 3000;
  private prevCam = new THREE.Vector3();
  private havePrevCam = false;

  constructor(scene: THREE.Scene, radius: number, material: THREE.Material, opts: PlanetOptions = {}, body: BodySurface = EARTH_BODY) {
    this.radius = radius;
    this.material = material;
    this.body = body;
    this.o = {
      maxLevel: 20, tauPx: 2, res: 65, cacheSize: 300, buildBudget: 12, buildBudgetMs: 6, capPx: 350,
      seaMode: false,
      ...opts,
    } as Required<Omit<PlanetOptions, 'seaMode'>> & { seaMode: boolean };
    this.syncTiles = body === EARTH_BODY || body === MOON_BODY
      ? new URLSearchParams(location.search).get('noworker') === '1'
      : true; // unknown BodySurface: functions are not in the worker bundle
    this.bodyKey = body === MOON_BODY ? 'moon' : 'earth';

    // Roots are built synchronously so the planet exists from frame 1.
    for (let f = 0; f < 6; f++) {
      const node = this.makeNode(f, 0, 0, 0);
      this.roots.push(node);
      node.tile = this.acquireTile(node);
      node.tile.mesh.visible = true;
    }
    scene.add(this.root);
  }

  /** Re-place all currently visible tiles (e.g. after an origin rebase). */
  forceReposition(origin: THREE.Vector3): void {
    this.originV.copy(origin);
    for (const r of this.roots) this.repositionSubtree(r);
  }

  private repositionSubtree(node: QNode): void {
    if (node.tile) {
      node.tile.mesh.position.set(
        node.center.x + this.bodyCenter.x - this.originV.x,
        node.center.y + this.bodyCenter.y - this.originV.y,
        node.center.z + this.bodyCenter.z - this.originV.z,
      );
      node.tile.mesh.updateMatrix();
    }
    if (node.children) for (const c of node.children) this.repositionSubtree(c);
  }

  private makeNode(face: number, level: number, ix: number, iy: number): QNode {
    const scale = 2 / (1 << level);
    const u0 = -1 + ix * scale;
    const v0 = -1 + iy * scale;
    const dir = new THREE.Vector3();
    const p = new THREE.Vector3();

    cubeToSphereDir(face, u0 + scale * 0.5, v0 + scale * 0.5, dir);
    const center = dir.clone().multiplyScalar(this.radius);

    let boundRadius = 0;
    let maxEdge = 0;
    const corners: THREE.Vector3[] = [];
    for (const [du, dv] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) {
      cubeToSphereDir(face, u0 + scale * du, v0 + scale * dv, dir);
      p.copy(dir).multiplyScalar(this.radius);
      corners.push(p.clone());
      boundRadius = Math.max(boundRadius, p.distanceTo(center));
    }
    maxEdge = Math.max(
      corners[0].distanceTo(corners[1]),
      corners[0].distanceTo(corners[2]),
      corners[3].distanceTo(corners[1]),
      corners[3].distanceTo(corners[2]),
    );

    return {
      face, level, ix, iy, center,
      boundRadius: boundRadius * 1.05,
      nx: center.x / this.radius,
      ny: center.y / this.radius,
      nz: center.z / this.radius,
      geomError: (maxEdge * maxEdge) / (8 * this.radius),
      edgeLen: maxEdge,
      children: null,
      tile: null,
      dead: false,
    };
  }

  /**
   * @param camera frame-relative camera (its .position lives in frame space)
   * @param origin floating-origin: absolute position of the frame origin.
   * Tile centers are absolute and immutable; meshes are placed at
   * center - origin (double math) every frame, so origin rebases never
   * invalidate cached tile geometry.
   */
  update(camera: THREE.PerspectiveCamera, origin: THREE.Vector3, viewportHeightPx: number): void {
    camera.updateMatrixWorld();
    this.originV.copy(origin);
    this.projScreen.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projScreen);
    // Absolute camera position for LOD distance decisions.
    this.camPos.copy(origin).add(camera.position);
    this.pxPerUnit = (viewportHeightPx * 0.5) / Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5);

    this.processQueue();

    this.stats.visibleTiles = 0;
    this.stats.triangles = 0;
    this.stats.maxVisibleLevel = 0;
    for (const r of this.roots) this.visit(r);

    this.stats.pending = this.queue.length + this.pool.pending;
    this.stats.cached = this.cache.size;

    // M10.1 lookahead: EMA of the camera's motion direction (world frame).
    // First frame and teleports (rebase / demo reset) are skipped via the
    // displacement gate; the EMA smooths per-frame jitter in the direction.
    if (this.havePrevCam) {
      const step = this._stepV.copy(this.camPos).sub(this.prevCam);
      const len = step.length();
      // Gate: >10 m/frame = real motion (skip rebase jumps by magnitude? no —
      // rebases move the ORIGIN, camPos is absolute, so a rebase shows up as
      // near-zero here. Teleports (demo resets) are large; reset instead).
      if (len > 1e4) {
        this.lookAhead.set(0, 0, 0);
      } else if (len > 0.5) {
        step.multiplyScalar(1 / len);
        this.lookAhead.lerp(step, 0.15).normalize();
      }
    }
    this.prevCam.copy(this.camPos);
    this.havePrevCam = true;
  }
  private readonly _stepV = new THREE.Vector3();

  private visit(node: QNode): void {
    // Frustum test in frame-relative space (meshes live there too).
    // The body's absolute center is added first (moon orbit), then the
    // origin subtracted — all double math.
    const cx = node.center.x + this.bodyCenter.x - this.originV.x;
    const cy = node.center.y + this.bodyCenter.y - this.originV.y;
    const cz = node.center.z + this.bodyCenter.z - this.originV.z;
    this.sphere.center.set(cx, cy, cz);
    this.sphere.radius = node.boundRadius;
    if (!this.frustum.intersectsSphere(this.sphere)) {
      this.hideSubtree(node);
      return;
    }

    // Distance to the spherical patch. Near-nadir, the camera height along the
    // tile normal is the true distance (center minus boundRadius goes negative
    // for wide tiles, which over-splits). Off to the side, the bounding-sphere
    // distance (center minus radius) is the better (less optimistic) estimate,
    // so take the max of both — never below a small epsilon.
    // (Absolute camera position minus the body's absolute tile center.)
    const absCx = node.center.x + this.bodyCenter.x;
    const absCy = node.center.y + this.bodyCenter.y;
    const absCz = node.center.z + this.bodyCenter.z;
    const relX = this.camPos.x - absCx;
    const relY = this.camPos.y - absCy;
    const relZ = this.camPos.z - absCz;
    const height = relX * node.nx + relY * node.ny + relZ * node.nz;
    const dCenter = this.camPos.distanceTo(this._absC.set(absCx, absCy, absCz));
    const d = Math.max(height, dCenter - node.boundRadius, 0.05);
    // Combined error: curvature sagitta in px, plus a screen-footprint term
    // (tile edge in px, rescaled so exceeding capPx counts as tauPx error).
    // Both are monotone in level, so the existing hysteresis stays valid.
    const sagPx = (node.geomError / d) * this.pxPerUnit;
    const edgePx = (node.edgeLen / d) * this.pxPerUnit;
    const rho = Math.max(sagPx, (edgePx * this.o.tauPx) / this.o.capPx);

    if (node.children === null) {
      if (rho > this.o.tauPx && node.level < this.o.maxLevel) this.split(node);
      if (node.children === null || !this.childrenReady(node)) {
        this.show(node);
        return;
      }
    } else if (rho < this.o.tauPx * 0.1) {
      this.merge(node);
      this.show(node);
      return;
    } else if (!this.childrenReady(node)) {
      this.show(node);
      return;
    }

    // All 4 children ready -> render children instead of this tile.
    node.tile!.mesh.visible = false;
    for (const c of node.children!) this.visit(c);
  }

  private show(node: QNode): void {
    const t = node.tile!;
    // Place the mesh in frame-relative space: double subtraction here is the
    // camera/origin-relative handoff to float32 (the only quantization step).
    // Body center (moon orbit position) is included in the absolute->rel map.
    t.mesh.position.set(
      node.center.x + this.bodyCenter.x - this.originV.x,
      node.center.y + this.bodyCenter.y - this.originV.y,
      node.center.z + this.bodyCenter.z - this.originV.z,
    );
    t.mesh.updateMatrix();
    t.mesh.visible = true;
    this.stats.visibleTiles++;
    this.stats.triangles += t.triangles;
    if (node.level > this.stats.maxVisibleLevel) this.stats.maxVisibleLevel = node.level;
  }

  private hideSubtree(node: QNode): void {
    if (node.tile) node.tile.mesh.visible = false;
    if (node.children) for (const c of node.children) this.hideSubtree(c);
  }

  private split(node: QNode): void {
    const cx = node.ix * 2;
    const cy = node.iy * 2;
    node.children = [
      this.makeNode(node.face, node.level + 1, cx, cy),
      this.makeNode(node.face, node.level + 1, cx + 1, cy),
      this.makeNode(node.face, node.level + 1, cx, cy + 1),
      this.makeNode(node.face, node.level + 1, cx + 1, cy + 1),
    ];
    for (const c of node.children) {
      const key = keyOf(c);
      const hit = this.cache.get(key);
      if (hit) {
        this.cache.delete(key); // refresh LRU recency
        c.tile = hit;
        this.stats.cacheHits++;
      } else {
        this.queue.push(c);
      }
    }
  }

  private merge(node: QNode): void {
    for (const c of node.children!) this.disposeSubtree(c);
    node.children = null;
  }

  private disposeSubtree(node: QNode): void {
    node.dead = true;
    if (node.children) for (const c of node.children) this.disposeSubtree(c);
    if (node.tile) {
      node.tile.mesh.visible = false;
      const key = keyOf(node);
      // If a stale worker result already parked a tile under this key, that
      // entry is superseded — dispose it instead of leaking its geometry.
      const dup = this.cache.get(key);
      if (dup && dup !== node.tile) {
        this.cache.delete(key);
        dup.dispose();
      }
      if (this.cache.size >= this.o.cacheSize) {
        const oldest = this.cache.keys().next().value as string | undefined;
        if (oldest !== undefined) {
          this.cache.get(oldest)!.dispose();
          this.cache.delete(oldest);
          this.stats.evicted++;
        }
      }
      this.cache.set(key, node.tile);
      node.tile = null;
    }
  }

  private childrenReady(node: QNode): boolean {
    for (const c of node.children!) if (c.tile === null) return false;
    return true;
  }

  private processQueue(): void {
    if (this.queue.length === 0) return;
    // Priority sort (M10.1): nearest-first, biased by camera MOTION — tiles
    // ahead of the flight direction are pulled forward so forward flight
    // rarely waits on geometry that split mid-frame. The velocity estimate is
    // an exponential moving average of the camera displacement, scaled to the
    // tile-edge magnitude so it never dominates pure distance at low speeds.
    this.queue.sort((a, b) => this.priority(a) - this.priority(b));
    // Dispatch to the worker pool (terrain) or build inline (sea). The old
    // synchronous time budget only applied to terrain builds — those have
    // left the main thread entirely; sea builds stay inline (cheap lattice).
    const deadline = performance.now() + this.o.buildBudgetMs;
    while (this.queue.length > 0) {
      const node = this.queue.shift()!;
      if (node.dead || node.tile) continue;
      if (this.o.seaMode || this.syncTiles) {
        // Sea + fallback (?noworker=1): synchronous, time-budgeted as before.
        if (performance.now() >= deadline) {
          this.queue.unshift(node);
          break;
        }
        node.tile = this.acquireTile(node);
      } else {
        this.pool.request(
          keyOf(node), node.face, node.level, node.ix, node.iy,
          this.radius, this.o.res, this.material, this.bodyKey,
          (tile) => {
            if (node.dead || node.tile) {
              // Stale result: park it in the LRU cache instead of wiring it
              // into a dead node (it is fully built — re-approach reuses it).
              this.cachePutStale(node, tile);
              return;
            }
            node.tile = tile;
            this.root.add(tile.mesh);
            this.stats.built++;
            this.stats.workerBuilt++;
          },
        );
      }
    }
  }

  /** LRU-insert a tile whose requester died before the result arrived. */
  private cachePutStale(node: QNode, tile: TileMesh): void {
    tile.mesh.visible = false;
    const key = keyOf(node);
    const dup = this.cache.get(key);
    if (dup) {
      // Same key cached twice (e.g. merge+resplit raced an in-flight job):
      // dispose the old copy, keep the fresh one.
      this.cache.delete(key);
      dup.dispose();
    }
    if (this.cache.size >= this.o.cacheSize) {
      const oldest = this.cache.keys().next().value as string | undefined;
      if (oldest !== undefined) {
        this.cache.get(oldest)!.dispose();
        this.cache.delete(oldest);
        this.stats.evicted++;
      }
    }
    this.cache.set(key, tile);
  }

  /**
   * Sort key: distance to camera minus a lookahead bonus. Tiles in the
   * direction of motion get `lookaheadM` meters off their distance — enough
   * to jump ~1 edge ahead of lateral neighbours at flight speed, harmless at
   * hover (velocity ~ 0).
   */
  private priority(node: QNode): number {
    const d = this.camPos.distanceTo(node.center);
    if (this.lookAhead.lengthSq() === 0) return d;
    const toNode = this._pTmp.copy(node.center).sub(this.camPos);
    const dist = toNode.length();
    if (dist < 1) return d;
    toNode.multiplyScalar(1 / dist);
    const ahead = toNode.dot(this.lookAhead); // 1 = straight ahead
    return d - Math.max(0, ahead) * this.lookAheadM;
  }
  private readonly _pTmp = new THREE.Vector3();

  private acquireTile(node: QNode): TileMesh {
    if (this.o.seaMode) {
      // Ocean shell: smooth sphere + depth attribute, built via seaGeometry
      const built = buildSeaGeometry(node.face, node.level, node.ix, node.iy, this.radius, this.o.res);
      const t = new TileMesh(
        node.face, node.level, node.ix, node.iy, this.radius, this.o.res, this.material,
        built.geometry, built.center,
      );
      this.root.add(t.mesh);
      this.stats.built++;
      return t;
    }
    const built = buildTileGeometry(
      node.face, node.level, node.ix, node.iy, this.radius, this.o.res, this.body,
    );
    const t = new TileMesh(
      node.face, node.level, node.ix, node.iy, this.radius, this.o.res, this.material,
      built.geometry, built.center,
    );
    this.root.add(t.mesh);
    this.stats.built++;
    return t;
  }
}
