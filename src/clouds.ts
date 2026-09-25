import * as THREE from 'three';

/**
 * Cloud layer (Phase 8): a single sphere shell rendered BackSide with a
 * fragment-shader cloud field — the same inverted-hull trick as the
 * atmosphere shell, so it reads correctly from inside (looking up at clouds)
 * and outside (from space, the cloud PATTERN on the limb and dayside).
 *
 * Design vs. the strategy note:
 *  - The note's endgame is raymarched volumetrics near the camera; that is a
 *    later iteration. This milestone delivers the reads that matter most:
 *    cloud cover shapes from space, overcast/blue sky from the ground, and
 *    an altitude-based fade so flying THROUGH the layer doesn't pop.
 *  - The cloud field is procedural in the SHADER: 3D-ish noise from two
 *    crossing planar fbm samples over the cube-face direction (cheap stand-in
 *    for a weather cubemap), drifting slowly with time.
 *  - Coverage is modulated by a low-frequency "weather" pattern; latitude
 *    bands (ITCZ + mid-latitude storm tracks) are hinted via the direction's
 *    latitude. No shadows on terrain yet (Phase 8 follow-up).
 */
export const CLOUD_BOTTOM = 1800; // m above sea level
export const CLOUD_TOP = 4200; // m above sea level (layer thickness)

export interface CloudUniforms {
  uSunDir: { value: THREE.Vector3 };
  uPlanetR: { value: number };
  uCamPos: { value: THREE.Vector3 };
  uOrigin: { value: THREE.Vector3 };
  uTime: { value: number };
  uCover: { value: number }; // 0..1 global coverage bias
}

export function makeCloudUniforms(planetR: number): CloudUniforms {
  return {
    uSunDir: { value: new THREE.Vector3(1, 0.3, 0.35).normalize() },
    uPlanetR: { value: planetR },
    uCamPos: { value: new THREE.Vector3() },
    uOrigin: { value: new THREE.Vector3() },
    uTime: { value: 0 },
    uCover: { value: 0.42 },
  };
}

export function makeCloudMesh(planetR: number, uniforms: CloudUniforms): THREE.Mesh {
  // shell at the middle of the layer
  const geo = new THREE.SphereGeometry(planetR + (CLOUD_BOTTOM + CLOUD_TOP) / 2, 128, 96);
  const mat = new THREE.ShaderMaterial({
    uniforms: uniforms as unknown as { [k: string]: THREE.IUniform },
    // DoubleSide: BackSide alone is only visible from INSIDE the sphere
    // (ground looking up). From space the front faces face away, so the
    // cloud pattern must render on the FRONT hull too.
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
    vertexShader: /* glsl */ `
      varying vec3 vWorld;
      void main() {
        vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uSunDir;
      uniform float uPlanetR;
      uniform vec3 uCamPos;
      uniform vec3 uOrigin;
      uniform float uTime;
      uniform float uCover;
      varying vec3 vWorld;

      float hash13(vec3 p) {
        p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
        p *= 17.0;
        return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
      }
      float noise3(vec3 x) {
        vec3 i = floor(x);
        vec3 f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        return mix(
          mix(mix(hash13(i), hash13(i + vec3(1,0,0)), f.x),
              mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), f.x), f.y),
          mix(mix(hash13(i + vec3(0,0,1)), hash13(i + vec3(1,0,1)), f.x),
              mix(hash13(i + vec3(0,1,1)), hash13(i + vec3(1,1,1)), f.x), f.y),
          f.z);
      }
      float fbm4(vec3 p) {
        float a = 0.5;
        float s = 0.0;
        for (int i = 0; i < 4; i++) {
          s += a * noise3(p);
          p *= 2.13;
          a *= 0.5;
        }
        return s;
      }

      void main() {
        vec3 pc = -uOrigin;
        vec3 up = normalize(vWorld - pc);
        // camera altitude above sea level
        float camAlt = length(uCamPos - pc) - uPlanetR;
        // fragment's own altitude within the layer [0..1]
        float hLayer = clamp((length(vWorld - pc) - uPlanetR - ${CLOUD_BOTTOM.toFixed(1)}) /
                             ${((CLOUD_TOP - CLOUD_BOTTOM)).toFixed(1)}, 0.0, 1.0);

        // ---- weather/coverage field on the sphere (slow drift) ----
        // project direction onto a stable planar frame (cheap cube-ish uv)
        vec3 q = up * 2.2; // weather scale ~ R/2.2
        float wTime = uTime * 0.002;
        // weather: large-scale coverage
        float weather = fbm4(q + vec3(wTime, wTime * 0.7, -wTime * 0.6));
        // latitude bands: ITCZ near equator + storm tracks ~55deg
        float lat = asin(clamp(up.y, -1.0, 1.0));
        float bands = 0.55 + 0.45 * cos(lat * 6.0) * 0.5 + 0.25 * exp(-pow((abs(lat) - 0.15) * 3.0, 2.0));
        float cover = clamp(uCover * bands * 1.6 * weather + (weather - 0.5) * 0.4, 0.0, 1.0);
        cover = pow(cover, 0.7); // bias toward more visible coverage

        // ---- cloud density field: detail noise carved by coverage ----
        vec3 dq = up * 40.0; // detail scale ~ R/40 (~160 km cells)
        float t2 = uTime * 0.006;
        float detail = fbm4(dq + vec3(t2, t2 * 1.3, -t2) + weather * 1.5);
        // detail ~[0,0.94] centered ~0.47; remap around 0.5 and threshold by
        // coverage: low cover = rare peaks only, high cover = broad deck
        float dn = detail - 0.47;
        float thr = mix(0.28, -0.12, cover); // higher cover -> lower threshold
        float density = smoothstep(thr, thr + 0.22, dn);
        density = clamp(density * 1.4, 0.0, 1.0);

        // vertical profile: puffy middle, wispy top/bottom
        density *= 0.65 + 0.35 * sin(hLayer * 3.14159);

        if (density < 0.01) discard;

        // ---- shading: cheap directional wrap + silver lining ----
        float ndl = clamp(dot(up, uSunDir) * 0.6 + 0.4, 0.0, 1.0);
        vec3 col = vec3(1.0) * (0.35 + 0.65 * ndl);
        // sun-facing rim glow through thin clouds
        vec3 V = normalize(uCamPos - vWorld);
        float rim = pow(clamp(dot(V, uSunDir) * 0.5 + 0.5, 0.0, 1.0), 3.0);
        col += vec3(1.0, 0.9, 0.75) * rim * (1.0 - density) * 0.6;

        // camera is ABOVE the cloud layer top: use the front hull (overcast
        // deck below); INSIDE the band or below: use the back hull overhead.
        // Testing altitude directly avoids relying on face orientation.
        float aboveTop = step(4500.0, camAlt);
        float qSide = mix(1.0, 0.0, aboveTop); // 1 = back hull, 0 = front
        // front hull seen from above reads brighter (direct sun on tops)
        if (qSide < 0.5) {
          // soften the pattern seen from space (distance mutes contrast)
          density = density * 0.96 + 0.04 * cover;
        }
        float inBand = smoothstep(0.0, 0.25, hLayer) * (1.0 - smoothstep(0.75, 1.0, hLayer));
        float farFade = clamp(abs(camAlt - ${((CLOUD_BOTTOM + CLOUD_TOP) / 2).toFixed(1)}) / 6000.0, 0.0, 1.0);
        float alpha = density * mix(0.35, 0.95, farFade * inBand + farFade * (1.0 - inBand));
        // top-down view from space: stronger
        if (camAlt > 8000.0) alpha = density * 0.95;

        gl_FragColor = vec4(col, clamp(alpha, 0.0, 1.0));
        #include <colorspace_fragment>
      }
    `,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  mesh.renderOrder = 4; // before the atmosphere shell (5), after sea (1)
  return mesh;
}
