import * as THREE from 'three';
import { buildTileGeometry } from './tileGeometry';

/**
 * A rendered quadtree tile. The mesh is positioned at the tile center (double)
 * while the geometry stores camera-independent local offsets (float) — the
 * layout required for Phase 2 precision.
 */
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

  dispose(): void {
    this.mesh.geometry.dispose();
  }
}
