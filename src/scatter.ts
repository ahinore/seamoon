import * as THREE from 'three';
import { hash3i, fbm3 } from './noise';
import { terrainHeight, SEED } from './terrain';

/**
 * Vegetation & rocks (Phase 10 M10.2 — strategy note: "植生と岩：近景だけ
 * GPU インスタンシングで配置。配置も決定論的ノイズで").
 *
 * Design:
 *  - Scattering is per CAMERA-LEVEL tile (level >= minLevel). The set is
 *    derived purely from the tile's (face, level, ix, iy) — same input, same
 *    objects, on any machine, any thread, forever (deterministic-noise rule).
 *  - Placement lattice: cells of `cellM` meters in the tile's LOCAL uv
 *    rectangle. A cell holds an object when a hash falls below the biome
 *    density, which mirrors terrainColor's rules (same noise, same seeds):
 *      forest (wet lowland)  -> tree,   density up to ~0.5
 *      dry lowland           -> sparse trees + rocks
 *      bare rock (slope)     -> rocks
 *      snow / beach / ocean  -> nothing
 *  - GPU instancing: the caller wraps the returned matrices into one
 *    THREE.InstancedMesh per object KIND per tile (trees, rocks), so even
 *    500+ objects cost 2 draw calls per tile.
 *
 * Precision: instance matrices are TILE-LOCAL (origin = tile center, meters,
 * |offset| <= tile edge/2) — the tile mesh position already carries the
 * absolute placement (same pattern as tile vertices). No large-magnitude
 * float32 anywhere.
 */

export interface ScatterOptions {
  /** Min quadtree level at which scattering appears. */
  minLevel?: number;
  /** Placement cell size in meters. */
  cellM?: number;
  /** Tile vertex-grid resolution (must match the host tile's build res). */
  res?: number;
}

const _up = new THREE.Vector3(0, 1, 0);
const _q = new THREE.Quaternion();
const _pos = new THREE.Vector3();
const _scale = new THREE.Vector3();
const _n = new THREE.Vector3();
const _pU = new THREE.Vector3();
const _pV = new THREE.Vector3();

/**
 * Deterministic scatter for one tile, in TILE-LOCAL space
 * (+u = tangentU, +v = tangentV, +y = local up = surface normal).
 * Heights are sampled from the SAME terrainHeight the tile mesh displaces
 * with, so objects sit exactly on the visible ground.
 *
 * @param tangentU tangentV unit tangent vectors at the tile center (basis
 *   vectors of the local uv plane, meters).
 * @param edgeM tile edge length in meters.
 */
export function buildTileScatter(
  face: number, level: number, ix: number, iy: number,
  center: THREE.Vector3,
  tangentU: THREE.Vector3, tangentV: THREE.Vector3,
  edgeM: number,
  opts: ScatterOptions = {},
): { trees: THREE.Matrix4[]; rocks: THREE.Matrix4[] } {
  const minLevel = opts.minLevel ?? 8;
  const cellM = opts.cellM ?? 55;
  // World size multipliers: tree geometry is ~1.2 units tall, rock radius
  // ~0.5 unit. Instance scales alone give 1-2 m "bonsai" trees; these lift
  // trees to ~9-26 m and rocks to ~1-6 m as the design note intends.
  const TREE_M = 10;
  const ROCK_M = 6;
  // M10.5 fix: sample ground height with the SAME LOD spacing the host tile's
  // mesh uses (tileGeometry: spacing = edge / (res-1), res=65 default). The
  // height field fades its fine octaves per-spacing, so a deep tile's surface
  // carries detail a fixed 55 m sampling cannot see — scatter grounded at the
  // wrong spacing sits tens of meters above/below the visible mesh and
  // "vanishes" when viewed at grazing angles (trees were buried, nadir views
  // still showed the canopy through the terrain).
  const res = opts.res ?? 65;
  const spacing = Math.max(edgeM / (res - 1), 0);
  const R = center.length();
  const trees: THREE.Matrix4[] = [];
  const rocks: THREE.Matrix4[] = [];
  if (level < minLevel || edgeM <= 0) return { trees, rocks };

  // Tile-local placement helper. The tile mesh's local frame is WORLD-AXIS
  // aligned (geometry stores dir*(radius+h) - center; the mesh carries no
  // rotation), so the instance ground point must be built the same way:
  // world dir*(R+h) minus the tile center. Writing (offU, h, offV) here used
  // to bury every tree ~|h*sin(lat)| meters underground and displace it
  // horizontally by the U/V frame mismatch.
  const groundLocal = (h: number) => {
    const r = R + h;
    _pos.set(_n.x * r - center.x, _n.y * r - center.y, _n.z * r - center.z);
    _q.setFromUnitVectors(_up, _n);
  };

  // Lattice cells covering the tile rectangle (cap for pathological tiles).
  const cells = Math.min(Math.max(Math.round(edgeM / cellM), 1), 64);

  for (let cy = 0; cy < cells; cy++) {
    for (let cx = 0; cx < cells; cx++) {
      // Deterministic per-cell hashes (jitter x/z, kind/scale y).
      const cz = ((face * 63 + level) * 104729 + iy * 1013 + ix * 7) | 0;
      const r0 = hash3i(cx, cy, cz, SEED + 9001); // jitter x
      const r1 = hash3i(cx, cy, cz, SEED + 9002); // jitter v / scale
      const r2 = hash3i(cx, cy, cz, SEED + 9003); // kind roll
      const jx = (cx + 0.15 + 0.7 * r0) / cells - 0.5; // [-0.5, 0.5] w/ margin
      const jy = (cy + 0.15 + 0.7 * r1) / cells - 0.5;
      const offU = jx * edgeM;
      const offV = jy * edgeM;

      // Local surface position: project center + offsets back onto the sphere
      // so wide tiles keep objects on the curved surface.
      _n.copy(center)
        .addScaledVector(tangentU, offU)
        .addScaledVector(tangentV, offV)
        .normalize();

      const h = terrainHeight(_n.x, _n.y, _n.z, spacing);
      if (h <= 0) continue; // ocean / lake bottoms

      // Slope estimate: finite differences along the tangent basis, on the
      // sphere (eps is the angular step that corresponds to one cell).
      const eps = cellM / center.length();
      _pU.copy(_n).addScaledVector(tangentU, eps).normalize();
      _pV.copy(_n).addScaledVector(tangentV, eps).normalize();
      const hU = terrainHeight(_pU.x, _pU.y, _pU.z, spacing);
      const hV = terrainHeight(_pV.x, _pV.y, _pV.z, spacing);
      const slope = Math.min((Math.abs(hU - h) + Math.abs(hV - h)) / (2 * cellM), 1);

      // Biome gate — the same math terrainColor uses, so scatter and ground
      // shading can never disagree about where the forest is.
      // M10.5 retune: matches terrain.ts's corrected snowline (the old
      // power-0.62 curve put the snowline at ~500 m by lat 38, which barred
      // trees from the very tiles the camera flies over).
      const m = fbm3(_n.x * 8, _n.y * 8, _n.z * 8, SEED + 555, 2);
      const latRad = Math.asin(Math.min(Math.max(_n.y, -1), 1));
      const temp =
        Math.pow(Math.abs(latRad) / (Math.PI / 2), 1.35) * (3400 / 3000) +
        h / 6000 * 0.55 -
        m * 0.06;
      const snowH = 3400 - temp * 3000;
      if (h > snowH) continue;   // snow: nothing grows
      if (h < 12) continue;      // beach
      if (h > snowH * 0.72 || slope > 0.55) {
        // Bare-rock band: rocks only, sparse.
        if (r2 < 0.05) {
          const s = (0.6 + r0 * 1.3) * ROCK_M;
          groundLocal(h);
          _scale.set(s, s * (0.7 + r1 * 0.6), s);
          rocks.push(new THREE.Matrix4().compose(_pos, _q, _scale));
        }
        continue;
      }

      // Vegetated band. Forest density by moisture; dry areas mostly rocks.
      const dry = m < -0.15;
      const forestness = dry ? 0.08 : 0.28 + m * 0.3;
      if (r2 < forestness) {
        const scale = (0.8 + r0 * 0.9) * TREE_M;
        groundLocal(h);
        _scale.set(scale, scale * (0.9 + r1 * 0.4), scale);
        trees.push(new THREE.Matrix4().compose(_pos, _q, _scale));
      } else if (r2 < forestness + 0.03) {
        const s = (0.5 + r0 * 1.2) * ROCK_M;
        groundLocal(h);
        _scale.set(s, s * (0.7 + r1 * 0.6), s);
        rocks.push(new THREE.Matrix4().compose(_pos, _q, _scale));
      }
    }
  }
  return { trees, rocks };
}

/**
 * Low-poly tree: trunk (cylinder) + canopy (cone), merged into ONE geometry
 * with vertex colors so a single material serves the whole InstancedMesh.
 * Height ~1 unit => scaled by instance matrices (world tree ~10-25 m).
 */
export function makeTreeGeometry(): THREE.BufferGeometry {
  const trunk = new THREE.CylinderGeometry(0.06, 0.1, 0.4, 5);
  trunk.translate(0, 0.2, 0);
  const canopy = new THREE.ConeGeometry(0.32, 0.9, 6);
  canopy.translate(0, 0.75, 0);
  const parts = [trunk, canopy];
  const colors: number[][] = [
    [0.24, 0.15, 0.08], // trunk brown
    [0.11, 0.28, 0.1],  // canopy green (linear)
  ];
  const geoms = parts.map((g, i) => {
    const cnt = g.getAttribute('position').count;
    const arr = new Float32Array(cnt * 3);
    for (let v = 0; v < cnt; v++) {
      arr[v * 3] = colors[i][0];
      arr[v * 3 + 1] = colors[i][1];
      arr[v * 3 + 2] = colors[i][2];
    }
    g.setAttribute('aCol', new THREE.BufferAttribute(arr, 3));
    return g;
  });
  // Simple merge (three r170 has no BufferGeometryUtils import in this
  // project; positions/normals/index concatenation is enough — both parts
  // are non-indexed-compatible via toNonIndexed).
  const nonIdx = geoms.map((g) => g.toNonIndexed());
  let total = 0;
  for (const g of nonIdx) total += g.getAttribute('position').count;
  const pos = new Float32Array(total * 3);
  const nrm = new Float32Array(total * 3);
  const col = new Float32Array(total * 3);
  let o = 0;
  for (const g of nonIdx) {
    pos.set(g.getAttribute('position').array as Float32Array, o);
    nrm.set(g.getAttribute('normal').array as Float32Array, o);
    col.set(g.getAttribute('aCol').array as Float32Array, o);
    o += g.getAttribute('position').count * 3;
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  out.setAttribute('aCol', new THREE.BufferAttribute(col, 3));
  for (const g of [...nonIdx, ...geoms]) g.dispose();
  return out;
}

/**
 * Low-poly rock: icosahedron with per-vertex radius jitter (deterministic),
 * gray-brown vertex colors.
 */
export function makeRockGeometry(): THREE.BufferGeometry {
  const g = new THREE.IcosahedronGeometry(0.5, 1).toNonIndexed();
  const p = g.getAttribute('position') as THREE.BufferAttribute;
  const cnt = p.count;
  const col = new Float32Array(cnt * 3);
  for (let v = 0; v < cnt; v++) {
    const x = p.getX(v), y = p.getY(v), z = p.getZ(v);
    // Deterministic per-vertex jitter keyed on rounded position.
    const j = 0.75 + hash3i(
      Math.round(x * 97), Math.round(y * 97), Math.round(z * 97), SEED + 9100,
    ) * 0.5;
    p.setXYZ(v, x * j, y * j * 0.72, z * j); // slightly flattened
    const shade = 0.32 + j * 0.14;
    col[v * 3] = shade * 1.05;
    col[v * 3 + 1] = shade;
    col[v * 3 + 2] = shade * 0.92;
  }
  g.setAttribute('aCol', new THREE.BufferAttribute(col, 3));
  g.computeVertexNormals();
  return g;
}
