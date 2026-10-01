import * as THREE from 'three';
import { TileMesh } from './tileMesh';
import { buildTileGeometry, cubeToSphereDir, EARTH_BODY, type BodySurface } from './tileGeometry';
import { MOON_BODY } from './moonBody';
import { buildSeaGeometry } from './seaGeometry';
import { TilePool } from './tilePool';
import { buildTileScatter, makeTreeGeometry, makeRockGeometry } from './scatter';
import { makeVegMaterial } from './vegMaterial';

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
  /** M11m A/B: disable the side-plane cull clause (mis-cull diagnosis). */
  noCullSide?: boolean;
  /** M11m A/B: disable the near/far cull clause. */
  noCullNear?: boolean;
  /** M11m debug: record near culled nodes' test inputs in stats.cullDbg. */
  cullDbg?: boolean;
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
  /** M11j debug: visible tile count per level (index = level). */
  perLevel?: number[];
  /** M11m debug: test inputs of near culled nodes (cullDbg mode). */
  cullDbg?: { lvl: number; r: number; z: number; x: number; y: number; d: number; tanH: number; tanV: number; near: number; far: number }[];
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
  // M11h: worker results waiting to be attached (bounded application rate)
  private readonly staged: { node: QNode; tile: TileMesh }[] = [];
  private readonly camPos = new THREE.Vector3();
  private readonly originV = new THREE.Vector3();
  private readonly _absC = new THREE.Vector3();
  private pxPerUnit = 1;
  private readonly frustum = new THREE.Frustum();
  private readonly sphere = new THREE.Sphere();
  private readonly projScreen = new THREE.Matrix4();
  // M11c manual frustum basis
  private readonly _fwd = new THREE.Vector3();
  private readonly _right = new THREE.Vector3();
  private readonly _upv = new THREE.Vector3();
  private _tanV = 1;
  private _tanH = 1;
  private _near = 0.1;
  private _far = 1e9;
  /** M11m: camera position in FRAME-RELATIVE space (frustum test only). */
  private _camRel = new THREE.Vector3();
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
  // ---- M10.2: near-scene vegetation & rocks ----
  /** ?veg=0 disables scattering (A/B diagnosis). */
  private readonly vegOn: boolean;
  private readonly treeGeo = makeTreeGeometry();
  private readonly rockGeo = makeRockGeometry();
  // M10.5: light-free custom shader (the scene has zero THREE lights —
  // MeshLambertMaterial rendered every tree near-black). Takes the shared
  // uSunDir uniform object so trees agree with the ground lighting.
  // main.ts re-points uSunDir at the atmosphere's shared uniform object.
  readonly vegMaterial: ReturnType<typeof makeVegMaterial>;
  /** Number of tiles currently carrying scatter meshes (HUD). */
  vegTiles = 0;
  /** Shared sun-direction uniform the material must track (set by main). */
  readonly vegSunDir = { value: new THREE.Vector3(1, 0.3, 0.35).normalize() };

  constructor(scene: THREE.Scene, radius: number, material: THREE.Material, opts: PlanetOptions = {}, body: BodySurface = EARTH_BODY) {
    this.radius = radius;
    this.material = material;
    this.body = body;
    this.o = {
      maxLevel: 20, tauPx: 2, res: 65, cacheSize: 300, buildBudget: 12, buildBudgetMs: 6, capPx: 350,
      seaMode: false,
      ...opts,
    } as Required<Omit<PlanetOptions, 'seaMode'>> & { seaMode: boolean };
    this.vegMaterial = makeVegMaterial(this.vegSunDir);
    this.syncTiles = body === EARTH_BODY || body === MOON_BODY
      ? new URLSearchParams(location.search).get('noworker') === '1'
      : true; // unknown BodySurface: functions are not in the worker bundle
    this.bodyKey = body === MOON_BODY ? 'moon' : 'earth';
    // Vegetation only makes sense on Earth (moon is airless regolith) and
    // only on the TERRAIN view (never the sea shell).
    this.vegOn =
      !this.o.seaMode &&
      this.bodyKey === 'earth' &&
      new URLSearchParams(location.search).get('veg') !== '0';

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
    // M11c: manual frustum. setFromProjectionMatrix went degenerate at the
    // extreme far/near the lunar views need (near ~3.7 km, far 2e9): the
    // extracted side planes ended up ~2e9 off and culled ON-SCREEN tiles
    // (the lunar surface rasterized ~1% coverage). Basis-vector sphere
    // tests are numerically identical math without the matrix extraction.
    this.camPos.copy(origin).add(camera.position);
    // M11m CRITICAL FIX: the frustum test below mixed coordinate frames.
    // sphere.center is FRAME-RELATIVE (node.center + bodyCenter - originV)
    // but camPos was ABSOLUTE (origin + camera.position) — after the first
    // origin rebase (|origin| ~ Mm) the difference is wrong by |origin|,
    // so near tiles computed as being millions of meters off-axis and were
    // culled + detached: "the foreground disappears when I look at the
    // horizon" (worst at grazing pitches, invisible when origin==0).
    // The test now uses the camera's frame-relative position, matching
    // the comment's own intent; camPos stays ABSOLUTE for the distance
    // estimates (absolute tile centers) further down.
    this._camRel.copy(camera.position);
    camera.getWorldDirection(this._fwd);
    this._right.crossVectors(this._fwd, camera.up).normalize();
    this._upv.crossVectors(this._right, this._fwd).normalize();
    this._tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5);
    this._tanH = this._tanV * camera.aspect;
    this._near = camera.near;
    this._far = camera.far;
    this.pxPerUnit = (viewportHeightPx * 0.5) / this._tanV;

    this.processQueue();

    this.stats.visibleTiles = 0;
    this.stats.triangles = 0;
    this.stats.maxVisibleLevel = 0;
    this.stats.perLevel = new Array(21).fill(0);
    (this.stats as { cullDbg?: unknown[] }).cullDbg = (this.o as { cullDbg?: boolean }).cullDbg ? [] : undefined;
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
    // M11c: basis-vector sphere-vs-frustum (replaces the degenerate
    // matrix-extracted frustum; see the comment in update()).
    {
      // M11m: _camRel is FRAME-RELATIVE like sphere.center — see update().
      const ex = this.sphere.center.x - this._camRel.x;
      const ey = this.sphere.center.y - this._camRel.y;
      const ez = this.sphere.center.z - this._camRel.z;
      const z = ex * this._fwd.x + ey * this._fwd.y + ez * this._fwd.z;
      const r = this.sphere.radius;
      // M11m: clause-level A/B switches for the culling-eats-near-tiles bug
      // (?nocullside=1 / ?nocullnear=1) — bisects which test mis-culls.
      const oc = this.o as { noCullSide?: boolean; noCullNear?: boolean };
      let culled = false;
      if (oc.noCullNear !== true) culled = z + r < this._near || z - r > this._far;
      if (!culled && oc.noCullSide !== true) {
        const x = ex * this._right.x + ey * this._right.y + ez * this._right.z;
        const y = ex * this._upv.x + ey * this._upv.y + ez * this._upv.z;
        culled = Math.abs(x) - r > z * this._tanH || Math.abs(y) - r > z * this._tanV;
      }
      if (culled) {
        // M11m: optional dump of mis-cull suspects (cullDbg=1): records the
        // first N culled nodes per update with their test inputs. NEAR only
        // (< 1500 km) — far-side tiles are legitimately culled.
        if ((this.o as { cullDbg?: boolean }).cullDbg && this.stats.cullDbg && this.stats.cullDbg.length < 24 && this.sphere.center.distanceTo(this.camPos) < 8e6) {
          const x = ex * this._right.x + ey * this._right.y + ez * this._right.z;
          const y = ex * this._upv.x + ey * this._upv.y + ez * this._upv.z;
          this.stats.cullDbg.push({
            lvl: node.level, r: +r.toPrecision(4),
            z: +z.toPrecision(4), x: +x.toPrecision(4), y: +y.toPrecision(4),
            d: +this.sphere.center.distanceTo(this.camPos).toPrecision(5),
            tanH: +this._tanH.toPrecision(3), tanV: +this._tanV.toPrecision(3),
            near: this._near, far: this._far,
          });
        }
        if ((this.o as { noFrustumCull?: boolean }).noFrustumCull !== true) {
          this.hideSubtree(node);
          return;
        }
      }
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
    // M11h: detach the hidden interior tile's MESH from the scene graph.
    // The tile object (and geometry) stays for a cheap re-show, but a
    // detached mesh is skipped by the renderer's per-frame scene walk —
    // a long mission otherwise accumulates ~2400 live-but-invisible
    // nodes under the view root (the gradual slowdown).
    const hidden = node.tile!.mesh;
    if (hidden.parent) hidden.removeFromParent();
    for (const c of node.children!) this.visit(c);
  }

  /**
   * M10.2: build + attach this tile's vegetation/rock InstancedMeshes.
   * The scatter lives in tile-local space, so it inherits the tile's world
   * transform (position = absolute center − origin). Deterministic: same
   * tile key always yields the same forest.
   */
  private attachScatter(node: QNode, t: TileMesh): void {
    // Tangent basis at the tile center: U/V span the local uv plane.
    const nrm = new THREE.Vector3(node.nx, node.ny, node.nz);
    // Pick the least-parallel world axis for a stable tangent.
    const ref = Math.abs(nrm.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
    const U = new THREE.Vector3().crossVectors(ref, nrm).normalize();
    const V = new THREE.Vector3().crossVectors(nrm, U).normalize();

    const { trees, rocks } = buildTileScatter(
      node.face, node.level, node.ix, node.iy,
      node.center, U, V, node.edgeLen,
      { res: this.o.res },
    );
    let added = 0;
    const place = (matrices: THREE.Matrix4[], geo: THREE.BufferGeometry) => {
      if (matrices.length === 0) return;
      const im = new THREE.InstancedMesh(geo, this.vegMaterial, matrices.length);
      for (let i = 0; i < matrices.length; i++) im.setMatrixAt(i, matrices[i]);
      im.instanceMatrix.needsUpdate = true;
      im.frustumCulled = false; // culling is the tile quadtree's job
      im.renderOrder = 2;
      t.mesh.add(im);
      added++;
    };
    place(trees, this.treeGeo);
    place(rocks, this.rockGeo);
    if (added > 0) this.vegTiles++;
  }

  private show(node: QNode): void {
    const t = node.tile!;
    // M10.2/M10.5: attach near-scene scatter when this tile first becomes
    // visible at vegetation depth. Attached ONCE per tile instance (node
    // .vegDone guard); the InstancedMeshes live as children of the tile mesh
    // so repositioning and visibility follow the tile for free.
    // M10.5 fix: the guard previously fired only on the FIRST visible tile at
    // level>=8 along a branch — when the camera closed in and the quadtree
    // split deeper, the parent (carrying every tree) went hidden while the
    // fresh children had none, so nearby forests vanished. Attaching at EVERY
    // level>=8 show (children included) keeps trees present at all LOD depths;
    // buildTileScatter is deterministic per (face,level,ix,iy) so the forest
    // re-generates consistently for each deeper tile.
    if (this.vegOn && !t.vegDone && node.level >= 8) {
      t.vegDone = true;
      this.attachScatter(node, t);
    }
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
    // M11h: re-attach if the mesh was detached while hidden (see visit())
    if (!t.mesh.parent) this.root.add(t.mesh);
    this.stats.visibleTiles++;
    if (this.stats.perLevel && node.level < this.stats.perLevel.length) {
      this.stats.perLevel[node.level]++;
    }
    this.stats.triangles += t.triangles;
    if (node.level > this.stats.maxVisibleLevel) this.stats.maxVisibleLevel = node.level;
  }

  private hideSubtree(node: QNode): void {
    if (node.tile) {
      node.tile.mesh.visible = false;
      // M11h: fully out of view — detach from the scene graph as well
      if (node.tile.mesh.parent) node.tile.mesh.removeFromParent();
    }
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
    // M11h: drain staged worker results first, a bounded batch per frame so
    // a burst of finished tiles never uploads all its buffers in one render
    if (this.staged.length > 0) {
      const budget = Math.min(this.staged.length, 6);
      for (let i = 0; i < budget; i++) {
        const { node, tile } = this.staged.shift()!;
        if (node.dead || node.tile) {
          this.cachePutStale(node, tile);
          continue;
        }
        node.tile = tile;
        this.root.add(tile.mesh);
        this.stats.built++;
        this.stats.workerBuilt++;
      }
    }
    if (this.queue.length === 0 && this.staged.length === 0) return;
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
            // M11h: apply worker results in bounded batches. When a burst of
            // tiles resolves on the same frame (descent start: 1400 queued),
            // attaching them all at once makes the renderer upload every new
            // buffer in one draw call pass — the 150 ms hitch. Stage them;
            // processQueue drains the staging list a few per frame.
            this.staged.push({ node, tile });
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
