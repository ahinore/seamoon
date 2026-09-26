import * as THREE from 'three';
import { terrainHeight, terrainColor, MAX_ELEV } from './terrain';

/**
 * A renderable celestial body's surface: the height/color pair the tile
 * pipeline samples plus its max elevation (skirt depth). Earth and the moon
 * (Phase 9) both implement this, so ONE quadtree/geometry pipeline serves
 * every body ("the planet system with different parameters").
 */
export interface BodySurface {
  /** Elevation above the body's sphere, meters. Unit dir in, spacing for LOD. */
  height(x: number, y: number, z: number, spacing: number): number;
  /** Linear RGB in [0,1]. */
  color(x: number, y: number, z: number, h: number, slope: number): [number, number, number];
  /** Max plausible elevation — skirt depth sizing. */
  maxElev: number;
}

/** The default body (Earth) — preserves the pre-Phase-9 call signature. */
export const EARTH_BODY: BodySurface = {
  height: terrainHeight,
  color: terrainColor,
  maxElev: MAX_ELEV,
};

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
 * PERF (profiled): terrainHeight is ~2 us/call, so per-vertex sampling x4
 * (position + 3 forward-difference evals for the normal) cost ~33 ms/tile —
 * the worst frame-time offender at 12 builds/frame. Instead we sample ONE
 * (n+2)x(n+2) height lattice (the vertex grid extended by 1 cell per side)
 * and derive:
 *   - positions: dir * (R + h_lattice)
 *   - normals:   forward differences ACROSS the lattice (zero extra evals)
 *   - colors:    from the same lattice height + a lattice slope estimate
 * => ~4.5k height evals per tile instead of ~18k. The difference stencil
 * spans the same one-cell spacing as before, so results match the old
 * normals to float precision.
 *
 * Vertex layout: (res x res) surface grid + 1-vertex skirt ring around it.
 * Local offsets are computed in double BEFORE being stored into the
 * Float32Array: p_local = dir * (R + h) - center ("center (double) + local
 * offset (float)" from Phase 2).
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
  body: BodySurface = EARTH_BODY,
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
  const skirtDepth = oneLevel * 20 + body.maxElev;

  // meters between adjacent lattice points — also drives LOD octave fading
  const spacing = edge / (n - 1);

  // ---- shared height lattice: grid extended by 1 cell on every side ----
  // Lattice (I,J) with I,J in 0..n+1 maps to grid (ii, jj) = I-1, J-1 in
  // [-1..n]; vertices use ii/jj clamped to [0..n-1] (=> lattice 1..n), and
  // the outer ring serves the forward-difference stencil at the borders.
  const nLat = n + 2;
  const hLat = new Float64Array(nLat * nLat);
  const dirLat = new Float64Array(nLat * nLat * 3);
  for (let J = 0; J < nLat; J++) {
    for (let I = 0; I < nLat; I++) {
      const ii = I - 1;
      const jj = J - 1;
      const u = u0 + (scale * ii) / (n - 1);
      const v = v0 + (scale * jj) / (n - 1);
      cubeToSphereDir(face, u, v, dir);
      const k = J * nLat + I;
      // spacing of THIS tile's vertex grid — the height field fades its
      // finest octaves accordingly (anti-aliasing, kills LOD seam lakes)
      hLat[k] = body.height(dir.x, dir.y, dir.z, spacing);
      dirLat[k * 3] = dir.x;
      dirLat[k * 3 + 1] = dir.y;
      dirLat[k * 3 + 2] = dir.z;
    }
  }
  // meters between adjacent lattice points
  void spacing;

  const positions = new Float32Array(nu * nu * 3);
  const normals = new Float32Array(nu * nu * 3);
  const centers = new Float32Array(nu * nu * 3);
  const grids = new Float32Array(nu * nu * 3);
  const colors = new Float32Array(nu * nu * 3);

  const p0 = new THREE.Vector3();
  const p1 = new THREE.Vector3();
  const p2 = new THREE.Vector3();
  const nrm = new THREE.Vector3();

  let ptr = 0;
  let gptr = 0;
  for (let j = 0; j < nu; j++) {
    for (let i = 0; i < nu; i++) {
      const skirt = i === 0 || j === 0 || i === nu - 1 || j === nu - 1;
      const ii = Math.min(Math.max(i - 1, 0), n - 1);
      const jj = Math.min(Math.max(j - 1, 0), n - 1);
      const I = ii + 1;
      const J = jj + 1;
      const k = J * nLat + I;
      const hx = dirLat[k * 3];
      const hy = dirLat[k * 3 + 1];
      const hz = dirLat[k * 3 + 2];
      const h = hLat[k];
      const r = skirt ? radius + h - skirtDepth : radius + h;
      // double-precision subtraction before float32 quantization
      positions[ptr] = hx * r - center.x;
      positions[ptr + 1] = hy * r - center.y;
      positions[ptr + 2] = hz * r - center.z;

      if (skirt) {
        // skirt is a hidden wall: sphere normal, shading irrelevant
        normals[ptr] = hx;
        normals[ptr + 1] = hy;
        normals[ptr + 2] = hz;
      } else {
        // Height-field normal via lattice forward differences: pR/pU are the
        // neighbor positions (+u / +v axes); cross(pR-p0, pU-p0) is outward
        // for the CCW (u x v = axis) parameterization.
        const kR = J * nLat + (I + 1);
        const kU = (J + 1) * nLat + I;
        const r0 = radius + h;
        const rR = radius + hLat[kR];
        const rU = radius + hLat[kU];
        p0.set(hx * r0, hy * r0, hz * r0);
        p1.set(dirLat[kR * 3] * rR, dirLat[kR * 3 + 1] * rR, dirLat[kR * 3 + 2] * rR).sub(p0);
        p2.set(dirLat[kU * 3] * rU, dirLat[kU * 3 + 1] * rU, dirLat[kU * 3 + 2] * rU).sub(p0);
        nrm.crossVectors(p1, p2).normalize();
        if (nrm.dot(p0) < 0) nrm.negate();
        normals[ptr] = nrm.x;
        normals[ptr + 1] = nrm.y;
        normals[ptr + 2] = nrm.z;
      }
      centers[ptr] = hx * radius - center.x;
      centers[ptr + 1] = hy * radius - center.y;
      centers[ptr + 2] = hz * radius - center.z;
      // grid coordinates + skirt flag for shader-drawn wireframe
      grids[gptr] = ii;
      grids[gptr + 1] = jj;
      grids[gptr + 2] = skirt ? 1 : 0;
      if (!skirt) {
        // slope for biome rock: max |dh| over one cell / spacing (tangent)
        const dR = Math.abs(hLat[J * nLat + (I + 1)] - h);
        const dL = Math.abs(h - hLat[J * nLat + (I - 1)]);
        const dU = Math.abs(hLat[(J + 1) * nLat + I] - h);
        const dD = Math.abs(h - hLat[(J - 1) * nLat + I]);
        const slope = Math.max(dR, dL, dU, dD) / spacing;
        const c = body.color(hx, hy, hz, h, slope);
        colors[ptr] = c[0];
        colors[ptr + 1] = c[1];
        colors[ptr + 2] = c[2];
      }
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
