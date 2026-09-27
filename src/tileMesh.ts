import * as THREE from 'three';
import { buildTileGeometry } from './tileGeometry';

/**
 * A rendered quadtree tile. The mesh is positioned at the tile center (double)
 * while the geometry stores camera-independent local offsets (float) — the
 * layout required for Phase 2 precision.
 */

/** Wire format of a worker-built tile (matches tileWorker.ts response). */
export interface TransferredTile {
  position: Float32Array;
  normal: Float32Array;
  center: Float32Array;
  aGrid: Float32Array;
  color: Float32Array;
  index: Uint32Array;
  centerAbs: [number, number, number];
  triangles: number;
}

export class TileMesh {
  readonly mesh: THREE.Mesh;
  readonly center: THREE.Vector3;
  readonly triangles: number;

  constructor(
    face: number,
    level: number,
    ix: number,
    iy: number,
    radius: number,
    res: number,
    material: THREE.Material,
    // Prebuilt geometry/center (sea shell path). When omitted, terrain
    // geometry is built here as before.
    prebuiltGeometry?: THREE.BufferGeometry,
    prebuiltCenter?: THREE.Vector3,
  ) {
    const geometry = prebuiltGeometry ?? buildTileGeometry(face, level, ix, iy, radius, res).geometry;
    this.center = prebuiltCenter ?? geometry.boundingSphere!.center.clone();
    this.triangles = geometry.index ? geometry.index.count / 3 : 0;
    this.mesh = new THREE.Mesh(geometry, material);
    if (prebuiltGeometry) this.mesh.renderOrder = 1; // sea draws after terrain
    this.mesh.position.copy(this.center);
    this.mesh.matrixAutoUpdate = false;
    this.mesh.updateMatrix();
    this.mesh.visible = false;
  }

  /**
   * M10.1: worker path — wraps typed arrays transferred from the tile worker
   * into a BufferGeometry. Attribute upload to the GPU is the only
   * main-thread cost; the expensive sampling happened off-thread.
   */
  static fromTransfer(
    face: number,
    level: number,
    ix: number,
    iy: number,
    radius: number,
    res: number,
    material: THREE.Material,
    t: TransferredTile,
  ): TileMesh {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(t.position, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(t.normal, 3));
    geometry.setAttribute('center', new THREE.BufferAttribute(t.center, 3));
    geometry.setAttribute('aGrid', new THREE.BufferAttribute(t.aGrid, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(t.color, 3));
    geometry.setIndex(new THREE.BufferAttribute(t.index, 1));
    geometry.computeBoundingSphere();
    const center = new THREE.Vector3(t.centerAbs[0], t.centerAbs[1], t.centerAbs[2]);
    void face; void level; void ix; void iy; void radius; void res;
    const mesh = new THREE.Mesh(geometry, material);
    const tile = Object.create(TileMesh.prototype) as TileMesh;
    (tile as { mesh: THREE.Mesh }).mesh = mesh;
    (tile as { center: THREE.Vector3 }).center = center;
    (tile as { triangles: number }).triangles = t.triangles;
    mesh.position.copy(center);
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();
    mesh.visible = false;
    return tile;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
  }
}
