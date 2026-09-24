import * as THREE from 'three';
import { terrainHeight, terrainColor, MAX_ELEV } from './terrain';

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
 * Builds one quadtree tile of the cube sphere WITH terrain (Phase 3).
 *
 * Elevation comes from terrainHeight(dir) — the CPU truth also used for
 * collision later. Vertex layout: (res x res) surface grid + 1-vertex skirt
 * ring around it. Local offsets are computed in double BEFORE being stored
 * into the Float32Array: p_local = dir * (R + h) - center ("center (double) +
 * local offset (float)" from Phase 2).
 *
 * Skirt vertices drop skirtDepth below the terrain of their edge, so cracks
 * against coarser neighbours stay covered even with elevation.
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

  // Tile center: use the SPHERE point (not elevated terrain) as the frame
  // origin of the tile — stable, and local offsets stay small either way.
  cubeToSphereDir(face, u0 + scale * 0.5, v0 + scale * 0.5, dir);
  const center = dir.clone().multiplyScalar(radius);

  // Skirt depth: swallow T-junction gaps against neighbours up to ~3 levels
  // coarser, PLUS full terrain amplitude so skirts stay buried under relief.
  cubeToSphereDir(face, u0, v0, pA);
  cubeToSphereDir(face, u1, v0, pB);
  const edge = pA.distanceTo(pB);
  const oneLevel = (edge * edge) / (2 * (res - 1) * (res - 1) * radius);
  const skirtDepth = oneLevel * 20 + MAX_ELEV;

  const positions = new Float32Array(nu * nu * 3);
  const normals = new Float32Array(nu * nu * 3);
  const centers = new Float32Array(nu * nu * 3);
  const grids = new Float32Array(nu * nu * 3);
  const colors = new Float32Array(nu * nu * 3);

  let ptr = 0;
  let gptr = 0;
  // Terrain normal sampling: forward differences over the height field.
  // eps ~ one vertex spacing keeps normals LOD-appropriate (coarse tiles get
  // smooth normals => no aliasing at distance; detail fades in on approach).
  const spacing = edge / (n - 1);
  const eps = spacing / radius; // angular step in radians
  const t1 = new THREE.Vector3();
  const t2 = new THREE.Vector3();
  const up = Math.abs(dir.z) < 0.9 ? pB.set(0, 0, 1) : pB.set(1, 0, 0);
  const p0 = new THREE.Vector3();
  const p1 = new THREE.Vector3();
  const p2 = new THREE.Vector3();
  const nrm = new THREE.Vector3();
  const samplePos = (dx: number, dy: number, dz: number, out: THREE.Vector3) => {
    out.set(dir.x + dx, dir.y + dy, dir.z + dz).normalize();
    const h = terrainHeight(out.x, out.y, out.z);
    return out.multiplyScalar(radius + h);
  };
  for (let j = 0; j < nu; j++) {
    for (let i = 0; i < nu; i++) {
      const skirt = i === 0 || j === 0 || i === nu - 1 || j === nu - 1;
      const ii = Math.min(Math.max(i - 1, 0), n - 1);
      const jj = Math.min(Math.max(j - 1, 0), n - 1);
      const u = u0 + (scale * ii) / (n - 1);
      const v = v0 + (scale * jj) / (n - 1);
      cubeToSphereDir(face, u, v, dir);
      const h = terrainHeight(dir.x, dir.y, dir.z);
      const r = skirt ? radius + h - skirtDepth : radius + h;
      // double-precision subtraction before float32 quantization
      positions[ptr] = dir.x * r - center.x;
      positions[ptr + 1] = dir.y * r - center.y;
      positions[ptr + 2] = dir.z * r - center.z;
      if (skirt) {
        // skirt copies its edge vertex normal; computed below on the seam pass
        normals[ptr] = dir.x;
        normals[ptr + 1] = dir.y;
        normals[ptr + 2] = dir.z;
      } else {
        // height-field normal via forward differences (3 height evals)
        up.set(0, 0, 1);
        if (Math.abs(dir.z) > 0.9) up.set(1, 0, 0);
        t1.crossVectors(up, dir).normalize();
        t2.crossVectors(dir, t1).normalize();
        samplePos(0, 0, 0, p0);
        samplePos(t1.x * eps, t1.y * eps, t1.z * eps, p1);
        samplePos(t2.x * eps, t2.y * eps, t2.z * eps, p2);
        nrm.crossVectors(p1.sub(p0), p2.sub(p0)).normalize();
        // ensure outward
        if (nrm.dot(dir) < 0) nrm.negate();
        normals[ptr] = nrm.x;
        normals[ptr + 1] = nrm.y;
        normals[ptr + 2] = nrm.z;
      }
      centers[ptr] = dir.x * radius - center.x;
      centers[ptr + 1] = dir.y * radius - center.y;
      centers[ptr + 2] = dir.z * radius - center.z;
      // grid coordinates + skirt flag for shader-drawn wireframe
      grids[gptr] = ii;
      grids[gptr + 1] = jj;
      grids[gptr + 2] = skirt ? 1 : 0;
      const c = terrainColor(dir.x, dir.y, dir.z, h);
      colors[ptr] = c[0];
      colors[ptr + 1] = c[1];
      colors[ptr + 2] = c[2];
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
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(new THREE.BufferAttribute(idx, 1));
  geometry.computeBoundingSphere();

  return { geometry, center };
}
