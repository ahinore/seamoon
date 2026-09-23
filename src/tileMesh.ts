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
  ) {
    const built = buildTileGeometry(face, level, ix, iy, radius, res);
    this.center = built.center;
    this.triangles = built.geometry.index ? built.geometry.index.count / 3 : 0;
    this.mesh = new THREE.Mesh(built.geometry, material);
    this.mesh.position.copy(this.center);
    this.mesh.matrixAutoUpdate = false;
    this.mesh.updateMatrix();
    this.mesh.visible = false;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
  }
}
