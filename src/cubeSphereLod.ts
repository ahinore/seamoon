import * as THREE from 'three';
import { TileMesh } from './tileMesh';
import { cubeToSphereDir } from './tileGeometry';

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
   * Screen-footprint cap in pixels: a tile spanning more than this on screen
   * is split regardless of curvature error. Keeps the region near the camera
   * refined at grazing angles, where a smooth sphere has almost no sag error
   * but terrain (Phase 3) will need real detail. Children halve their
   * footprint per level, so the merge threshold (0.1*tau) stays chatter-free.
   */
  capPx?: number;
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
  };

  private readonly o: Required<PlanetOptions>;
  private readonly radius: number;
  private readonly material: THREE.Material;
  private readonly roots: QNode[] = [];
  private readonly cache = new Map<string, TileMesh>();
  private readonly queue: QNode[] = [];
  private readonly camPos = new THREE.Vector3();
  private readonly originV = new THREE.Vector3();
  private pxPerUnit = 1;
  private readonly frustum = new THREE.Frustum();
  private readonly sphere = new THREE.Sphere();
  private readonly projScreen = new THREE.Matrix4();

  constructor(scene: THREE.Scene, radius: number, material: THREE.Material, opts: PlanetOptions = {}) {
    this.radius = radius;
    this.material = material;
    this.o = { maxLevel: 20, tauPx: 2, res: 65, cacheSize: 300, buildBudget: 12, capPx: 350, ...opts };

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
      node.tile.mesh.position.copy(node.center).sub(this.originV);
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

    this.stats.pending = this.queue.length;
    this.stats.cached = this.cache.size;
  }

  private visit(node: QNode): void {
    // Frustum test in frame-relative space (meshes live there too).
    const cx = node.center.x - this.originV.x;
    const cy = node.center.y - this.originV.y;
    const cz = node.center.z - this.originV.z;
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
    const relX = this.camPos.x - node.center.x;
    const relY = this.camPos.y - node.center.y;
    const relZ = this.camPos.z - node.center.z;
    const height = relX * node.nx + relY * node.ny + relZ * node.nz;
    const dCenter = this.camPos.distanceTo(node.center);
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
    t.mesh.position.copy(node.center).sub(this.originV);
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
    this.queue.sort(
      (a, b) => this.camPos.distanceToSquared(a.center) - this.camPos.distanceToSquared(b.center),
    );
    const budget = Math.min(this.o.buildBudget, this.queue.length);
    for (let i = 0; i < budget; i++) {
      const node = this.queue.shift()!;
      if (node.dead || node.tile) continue;
      node.tile = this.acquireTile(node);
    }
  }

  private acquireTile(node: QNode): TileMesh {
    const t = new TileMesh(node.face, node.level, node.ix, node.iy, this.radius, this.o.res, this.material);
    this.root.add(t.mesh);
    this.stats.built++;
    return t;
  }
}
