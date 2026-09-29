import * as THREE from 'three';

/**
 * Vegetation material (M10.5). The scene deliberately carries ZERO THREE
 * lights (every other material is a hand-written ShaderMaterial — terrain,
 * sea, clouds, moon), but M10.2 attached scatter as MeshLambertMaterial,
 * which is light-dependent: with no lights the Lambert term is 0 and every
 * tree/rock rendered near-black (probe showed 000000 crowns from 600 m).
 *
 * This shader replaces it: same wrapped-Lambert + fixed ambient model as the
 * terrain material, driven by the shared uSunDir uniform object so trees and
 * ground always agree on lighting. Vertex colors (trunk brown / canopy green
 * / rock gray) come from the merged scatter geometry.
 *
 * Fog: scatter only exists at level>=8 tiles near the camera, so aerial
 * perspective is negligible at those distances — a cheap exp2 height-fog
 * toward the sky color keeps distant trees from popping out of hazy scenes.
 */
export function makeVegMaterial(sunDir: { value: THREE.Vector3 }): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uSunDir: sunDir,
      // M10.3: custom shaders run their own colorspace_fragment, so they
      // never see renderer.toneMapping — inline ACES behind a toggle.
      uToneMap: { value: 1 },
    },
    vertexShader: /* glsl */ `
      #include <common>
      // 'color' is a reserved-ish name in three's common chunk when
      // USE_COLOR isn't defined — use an explicit attribute name instead.
      attribute vec3 aCol;
      varying vec3 vN;
      varying vec3 vCol;
      varying vec3 vWorld;
      varying float vShrink;
      void main() {
        vec3 pos = position;
        vec3 nrm = normal;
        #ifdef USE_INSTANCING
        // M10.6: distance-based tree shrink. Scatter is per-tile with a fixed
        // 55 m lattice, so on far tiles the trees sub-pixel-pop and shimmer.
        // Scaling the instance toward its ground point as it recedes removes
        // the pop without alpha (no sorting/depth issues). Full size under
        // 12 km, gone by 28 km — beyond the level>=8 scatter attach range.
        //
        // The distance is measured to the INSTANCE's own ground point in
        // WORLD space (modelMatrix * instanceMatrix translation column), NOT
        // to the tile center: the world position of a tree's base never moves
        // — rebases shift every frame-space position by the same amount, so
        // the world-space distance (and hence the shrink factor) is INVARIANT
        // under rebase and LOD transitions. Measuring from the tile center
        // made the shrink jump whenever a tile was repositioned, which read
        // as trees visibly sliding while they "grew" near the camera.
          vec3 iTrans = instanceMatrix[3].xyz;
          mat3 iLinear = mat3(instanceMatrix);
          vec3 baseWorld = (modelMatrix * vec4(iTrans, 1.0)).xyz;
          vShrink = 1.0 - smoothstep(12000.0, 28000.0, distance(cameraPosition, baseWorld));
          pos = iTrans + iLinear * pos * vShrink;
          nrm = iLinear * nrm;
        #else
          vShrink = 1.0;
        #endif
        vN = normalize(mat3(modelMatrix) * nrm);
        vCol = aCol;
        vWorld = (modelMatrix * vec4(pos, 1.0)).xyz;
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: /* glsl */ `
      #include <common>
      uniform vec3 uSunDir;
      uniform float uToneMap;
      varying vec3 vN;
      varying vec3 vCol;
      varying vec3 vWorld;
      varying float vShrink;

      // ACES filmic (Narkowicz) — matches terrain/sea shaders (M10.3).
      vec3 acesToneMap(vec3 x) {
        return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
      }

      void main() {
        // M10.6: fully shrunk trees are still rasterized as sub-pixel dots at
        // glancing angles — fade them out completely past the shrink range so
        // they never shimmer. (vShrink==0 => discard.)
        if (vShrink <= 0.001) discard;
        vec3 N = normalize(vN);
        // wrapped Lambert, same soft-terminator constants as the terrain
        float ndl = clamp((dot(N, uSunDir) + 0.18) / 1.18, 0.0, 1.0);
        // Trees are small; their shaded side would go pure black under a
        // pure directional term. A fixed ~25% sky ambient (bluish, like the
        // atmosphere's in-scatter) models sky + ground bounce cheaply.
        vec3 ambient = vCol * vec3(0.30, 0.34, 0.40) * 0.55;
        vec3 col = vCol * (0.10 + 0.90 * ndl) + ambient * (1.0 - ndl);

        // Cheap height fog toward the horizon haze color: only matters for
        // the handful of scatter objects seen from >2 km (renderOrder 2).
        float dist = distance(cameraPosition, vWorld);
        float fog = 1.0 - exp(-dist * 6e-5);
        col = mix(col, vec3(0.55, 0.65, 0.80), fog * 0.6);

        col = mix(col, acesToneMap(col), uToneMap);
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}
