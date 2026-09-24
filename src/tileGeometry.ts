import * as THREE from 'three';

// Cube face definitions.
// axis = outward normal of the face, u/v = tangents with cross(u, v) = axis.
// This guarantees consistent CCW winding (front faces point outward).
export const FACES: readonly {
  axis: [number, number, number];
  u: [number, number, number];
  v: [number, number, number];
}[] = [
  { axis: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0] }, // +X
  { axis: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] }, // -X
  { axis: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1] }, // +Y
  { axis: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] }, // -Y
  { axis: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] }, // +Z
  { axis: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0] }, // -Z
];

/**
 * Maps a face parameter (u, v) in [-1, 1]^2 to a unit direction on the sphere.
 * Uses the tangent (tan) mapping: much more uniform vertex density across the
 * face than plain normalization, and no UV-sphere pole artifacts.
 */
export function cubeToSphereDir(face: number, u: number, v: number, out: THREE.Vector3): THREE.Vector3 {
  const f = FACES[face];
  const s = Math.tan(u * (Math.PI / 4));
  const t = Math.tan(v * (Math.PI / 4));
  out.set(
    f.axis[0] + f.u[0] * s + f.v[0] * t,
    f.axis[1] + f.u[1] * s + f.v[1] * t,
    f.axis[2] + f.u[2] * s + f.v[2] * t,
  );
  return out.normalize();
}

export interface BuiltTile {
  geometry: THREE.BufferGeometry;
  /** Tile center on the sphere (double precision, meters). */
  center: THREE.Vector3;
}

/**
 * Builds one quadtree tile of the cube sphere.
 *
 * Vertex layout: (res x res) surface grid + 1-vertex skirt ring around it.
 * Local vertex offsets are computed in double (JS number) BEFORE being stored
 * into the Float32Array: p_local = dir * r - center. This is the "center
 * (double) + local offset (float)" layout from Phase 2, applied from day one,
 * so small tiles never suffer world-magnitude float32 quantization.
 *
 * Skirt depth scales with the level: the T-junction gap against a one-level
 * coarser neighbour is ~ edge^2 / (2 * (res-1)^2 * R); we use 3x that.
 */
export function buildTileGeometry(
  face: number,
  level: number,
  ix: number,
  iy: number,
  radius: number,
  res: number,
): BuiltTile {
  const n = res;
  const nu = n + 2; // skirt ring included
  const scale = 2 / (1 << level);
  const u0 = -1 + ix * scale;
  const v0 = -1 + iy * scale;
  const u1 = u0 + scale;
  const v1 = v0 + scale;

  const dir = new THREE.Vector3();
  const pA = new THREE.Vector3();
  const pB = new THREE.Vector3();

  // Tile center (on the sphere surface)
  cubeToSphereDir(face, u0 + scale * 0.5, v0 + scale * 0.5, dir);
  const center = dir.clone().multiplyScalar(radius);

  // Skirt depth: deep enough to swallow T-junction cracks against neighbours
  // several levels coarser (the gap at a level difference of k scales as
  // (2^k - 1)^2 * oneLevelGap^2 / (2R)). 20x one-level covers ~3 levels of
  // difference; deeper skirts cost nothing except hidden surface slivers.
  cubeToSphereDir(face, u0, v0, pA);
  cubeToSphereDir(face, u1, v0, pB);
  const edge = pA.distanceTo(pB);
  const oneLevel = (edge * edge) / (2 * (res - 1) * (res - 1) * radius);
  const skirtDepth = oneLevel * 20;

  const positions = new Float32Array(nu * nu * 3);
  const normals = new Float32Array(nu * nu * 3);
  const centers = new Float32Array(nu * nu * 3);
  const grids = new Float32Array(nu * nu * 3);

  let ptr = 0;
  let gptr = 0;
  for (let j = 0; j < nu; j++) {
    for (let i = 0; i < nu; i++) {
      const skirt = i === 0 || j === 0 || i === nu - 1 || j === nu - 1;
      const ii = Math.min(Math.max(i - 1, 0), n - 1);
      const jj = Math.min(Math.max(j - 1, 0), n - 1);
      const u = u0 + (scale * ii) / (n - 1);
      const v = v0 + (scale * jj) / (n - 1);
      cubeToSphereDir(face, u, v, dir);
      const r = skirt ? radius - skirtDepth : radius;
      // double-precision subtraction before float32 quantization
      positions[ptr] = dir.x * r - center.x;
      positions[ptr + 1] = dir.y * r - center.y;
      positions[ptr + 2] = dir.z * r - center.z;
      normals[ptr] = dir.x;
      normals[ptr + 1] = dir.y;
      normals[ptr + 2] = dir.z;
      centers[ptr] = dir.x * radius - center.x;
      centers[ptr + 1] = dir.y * radius - center.y;
      centers[ptr + 2] = dir.z * radius - center.z;
      // grid coordinates + skirt flag for shader-drawn wireframe
      grids[gptr] = ii;
      grids[gptr + 1] = jj;
      grids[gptr + 2] = skirt ? 1 : 0;
      ptr += 3;
      gptr += 3;
    }
  }

  // Two triangles per cell, split along the A-D diagonal, CCW seen from outside
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
  geometry.setAttribute('aGrid', new THREE.BufferAttribute(grids, 3));
  geometry.setIndex(new THREE.BufferAttribute(idx, 1));
  geometry.computeBoundingSphere();

  return { geometry, center };
}
