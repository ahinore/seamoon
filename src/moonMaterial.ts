import * as THREE from 'three';

/**
 * Moon material (Phase 9): Lambert lit grayscale surface, NO atmosphere.
 *
 * Deliberately plain: no aerial-perspective fog (there is no air), no ocean,
 * no specular. The shared uSunDir gives the correct phase automatically —
 * when the camera looks at the moon from Earth, the terminator faces the
 * same sun the Earth does. A slight wrap keeps the terminator soft like
 * real regolith's multiple-scattering, and a touch of distance-based
 * blue-gray desaturation when seen through Earth's atmosphere is skipped
 * on purpose (the moon is rendered in space only for now).
 */
export function makeMoonMaterial(sunDir: { value: THREE.Vector3 }): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uSunDir: sunDir,
      uWire: { value: 0 },
      // M11j: level visualization (?lvl=1) — color tiles by quadtree
      // level (blue→red ramp) to diagnose LOD coverage questions.
      uLevelDebug: { value: 0 },
      // M11c: lander floodlight. A camera-anchored point light for the
      // final night-side approach: position/distance/decay follow the
      // standard three.js point-light convention, computed on the CPU
      // each frame (moon view only: see main.ts updateLandingLight).
      uLampPos: { value: new THREE.Vector3(0, 0, 0) },
      uLampOn: { value: 0 },
    },
    vertexShader: /* glsl */ `
      #include <common>
      attribute vec3 center;
      attribute vec3 aGrid;
      attribute vec3 color;
      varying vec3 vN;
      varying vec3 vGrid;
      varying vec3 vCol;
      varying vec3 vWorld;
      void main() {
        vN = normalize(mat3(modelMatrix) * normal);
        vGrid = aGrid;
        vCol = color;
        vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uSunDir;
      uniform float uWire;
      uniform float uLevelDebug;
      uniform vec3 uLampPos;
      uniform float uLampOn;
      varying vec3 vN;
      varying vec3 vGrid;
      varying vec3 vCol;
      varying vec3 vWorld;
      void main() {
        // wrapped Lambert: regolith scatters a little past the terminator
        float ndl = clamp((dot(normalize(vN), uSunDir) + 0.12) / 1.12, 0.0, 1.0);
        // regolith is dark: albedo ~0.12, so cap diffuse well below white
        vec3 col = vCol * (0.035 + 0.965 * ndl) * 0.95;

        // M11c: floodlight pool — inverse-square falloff with a soft near
        // field and a warm white tint, only where the surface faces the lamp
        if (uLampOn > 0.001) {
          vec3 toLamp = uLampPos - vWorld;
          float d2 = max(dot(toLamp, toLamp), 4.0);
          float atten = min(uLampOn * 3.0e4 / d2, 30.0);
          float ndlLamp = clamp(dot(normalize(vN), normalize(toLamp)), 0.0, 1.0);
          // soft near cut so the pool doesn't blow out right under the craft
          float near = smoothstep(1.5, 5.0, sqrt(d2));
          col += vec3(1.0, 0.96, 0.88) * (ndlLamp * atten * near) * vCol * 6.0;
        }

        // M11j: distance-band debug (?lvl=1) — near ground red, far blue:
        // answers 'is the descent black band near or far terrain' at a
        // glance. Skirts (vGrid.z) keep the tint so walls are visible too.
        if (uLevelDebug > 0.5) {
          float dist = length(vWorld - cameraPosition);
          float band = clamp((dist - 5000.0) / 40000.0, 0.0, 1.0);
          col = mix(vec3(1.0, 0.2, 0.1), vec3(0.1, 0.3, 1.0), band);
          col *= (0.3 + 0.7 * ndl);
        }
        // wireframe overlay (same shader-drawn style as the terrain)
        if (uWire > 0.5) {
          float fx = fwidth(vGrid.x) + 1e-6;
          float fy = fwidth(vGrid.y) + 1e-6;
          float b = min(min(vGrid.x, 64.0 - vGrid.x), min(vGrid.y, 64.0 - vGrid.y));
          float boundary = 1.0 - clamp(min(b / fx, b / fy), 0.0, 1.0);
          vec2 gf = abs(fract(vGrid.xy - 0.5) - 0.5);
          float grid = 1.0 - clamp(min(gf.x / fx, gf.y / fy), 0.0, 1.0);
          float fade = clamp((0.25 - max(fx, fy)) / 0.125, 0.0, 1.0);
          float line = max(boundary, grid * fade) * (1.0 - vGrid.z);
          col = mix(col, vec3(0.95), line * 0.85);
        }
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}
