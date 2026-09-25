import * as THREE from 'three';
import { cubeToSphereDir } from './tileGeometry';
import { terrainHeight } from './terrain';

/**
 * Builds one quadtree tile of the OCEAN SHELL (Phase 7).
 *
 * The sea surface is a displaced-terrain-free sphere at sea-level radius:
 * elevation is 0 everywhere by definition (terrain that would stick out is
 * simply covered by the land mesh in front of it). What the shader needs is
 * COASTAL INFORMATION, so each vertex carries the water DEPTH below — sampled
 * from the same terrainHeight() the land mesh displaces with, guaranteeing
 * the shoreline sits exactly where landColor(h=0) does.
 *
 * Geometry cost is tiny (coarse 5x5 depth lattice, res grid for smoothness),
 * so the ocean adds almost nothing to the tile-build budget.
 *
 * Depth convention (aDepth, meters, >= 0):
 *   depth = max(0, -terrainHeight(dir)) — 0 on land/shore, grows offshore.
 * The shader uses it for shallow-water color, foam, and wave damping.
 */
export function buildSeaGeometry(
  face: number,
  level: number,
  ix: number,
  iy: number,
  radius: number,
  res: number,
): { geometry: THREE.BufferGeometry; center: THREE.Vector3 } {
  const n = res;
  const nu = n + 2; // skirt ring (kept for TileMesh compatibility)
  const scale = 2 / (1 << level);
  const u0 = -1 + ix * scale;

  const dir = new THREE.Vector3();
  const pA = new THREE.Vector3();
  const pB = new THREE.Vector3();
  const v0 = -1 + iy * scale;
  const u1 = u0 + scale;
  const v1 = v0 + scale;

  cubeToSphereDir(face, u0 + scale * 0.5, v0 + scale * 0.5, dir);
  const center = dir.clone().multiplyScalar(radius);

  cubeToSphereDir(face, u0, v0, pA);
  cubeToSphereDir(face, u1, v0, pB);
  const edge = pA.distanceTo(pB);
  // skirt only needs to cover curvature gaps (no elevation on water)
  const oneLevel = (edge * edge) / (2 * (res - 1) * (res - 1) * radius);
  const skirtDepth = oneLevel * 20;

  const positions = new Float32Array(nu * nu * 3);
  const normals = new Float32Array(nu * nu * 3);
  const centers = new Float32Array(nu * nu * 3);
  const depths = new Float32Array(nu * nu);
  const grids = new Float32Array(nu * nu * 3);

  // Per-vertex sampling: 30x30 cell grid -> 961 terrainHeight calls/tile
  // (~2 us each = ~2 ms). Gives smooth shorelines and correct water edges
  // against the land mesh (which samples the same function at res 65).
  const DN2 = res - 1; // actual per-vertex lattice resolution
  const dLat = new Float64Array((DN2 + 1) * (DN2 + 1));
  for (let J = 0; J <= DN2; J++) {
    for (let I = 0; I <= DN2; I++) {
      const u = u0 + (scale * I) / DN2;
      const v = v0 + (scale * J) / DN2;
      cubeToSphereDir(face, u, v, dir);
      dLat[J * (DN2 + 1) + I] = Math.max(0, -terrainHeight(dir.x, dir.y, dir.z));
    }
  }

  const sampleDepth = (ii: number, jj: number): number =>
    dLat[jj * (DN2 + 1) + ii];

  let ptr = 0;
  let dptr = 0;
  let gptr = 0;
  for (let j = 0; j < nu; j++) {
    for (let i = 0; i < nu; i++) {
      const skirt = i === 0 || j === 0 || i === nu - 1 || j === nu - 1;
      const ii = Math.min(Math.max(i - 1, 0), n - 1);
      const jj = Math.min(Math.max(j - 1, 0), n - 1);
      // exact grid position in [-1..n] parameter space
      const uu = u0 + (scale * ii) / (n - 1);
      const vv = v0 + (scale * jj) / (n - 1);
      cubeToSphereDir(face, uu, vv, dir);
      // Skirt walls drop well below the deepest seafloor so that, when a
      // neighbouring terrain tile is coarser than this sea tile, the land
      // mesh's skirt (also dug 9.2 km deep) still wins the depth fight and
      // no black gap opens between land and sea.
      const r = skirt ? radius - Math.max(skirtDepth, 12000) : radius;
      positions[ptr] = dir.x * r - center.x;
      positions[ptr + 1] = dir.y * r - center.y;
      positions[ptr + 2] = dir.z * r - center.z;
      normals[ptr] = dir.x;
      normals[ptr + 1] = dir.y;
      normals[ptr + 2] = dir.z;
      centers[ptr] = dir.x * radius - center.x;
      centers[ptr + 1] = dir.y * radius - center.y;
      centers[ptr + 2] = dir.z * radius - center.z;
      depths[dptr++] = sampleDepth(ii, jj);
      grids[gptr] = ii;
      grids[gptr + 1] = jj;
      grids[gptr + 2] = skirt ? 1 : 0;
      ptr += 3;
      gptr += 3;
    }
  }

  const idx = new Uint32Array((nu - 1) * (nu - 1) * 6);
  let q = 0;
  for (let j = 0; j < nu - 1; j++) {
    for (let i = 0; i < nu - 1; i++) {
      const a = j * nu + i;
      const b = a + 1;
      const c = a + nu;
      const d = c + 1;
      idx[q++] = a; idx[q++] = d; idx[q++] = c;
      idx[q++] = a; idx[q++] = b; idx[q++] = d;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.setAttribute('center', new THREE.BufferAttribute(centers, 3));
  geometry.setAttribute('aDepth', new THREE.BufferAttribute(depths, 1));
  geometry.setAttribute('aGrid', new THREE.BufferAttribute(grids, 3));
  geometry.setIndex(new THREE.BufferAttribute(idx, 1));
  geometry.computeBoundingSphere();

  return { geometry, center };
}
