import * as THREE from 'three';

/**
 * Analytic atmosphere shell (Phase 5, milestone A5-1).
 *
 * A single inverted-hull mesh around the planet, shaded in the fragment
 * shader with an analytic approximation of single-scattering Rayleigh
 * + Mie. Not a Bruneton/LUT solver — the goal for M1 is the *reads*:
 *   - blue limb glow from space (rim)
 *   - sky that stays blue overhead on the ground and whitens to the horizon
 *   - sunset reddening when looking along the sun ray
 *   - aerial perspective fades terrain into the atmosphere color (applied
 *     in the terrain shader separately, via shared uniforms)
 *
 * Physics parameters are real-ish: scale heights Rayleigh 8.5 km,
 * Mie 1.2 km, atmosphere top 60 km. Planet radius comes in via uniform so
 * the Moon (no atmosphere) just skips adding the shell.
 */
export const ATMOSPHERE_TOP = 60_000; // m above surface

export interface AtmosphereUniforms {
  uSunDir: { value: THREE.Vector3 };
  uPlanetR: { value: number };
  uAtmoR: { value: number };
  uCamPos: { value: THREE.Vector3 }; // frame-relative camera position
  uBetaR: { value: THREE.Vector3 };
  uBetaM: { value: THREE.Vector3 };
  uHR: { value: number };
  uHM: { value: number };
  uOrigin: { value: THREE.Vector3 }; // floating origin (frame-space planet center = -origin)
}

export function makeAtmosphereUniforms(planetR: number): AtmosphereUniforms {
  return {
    uSunDir: { value: new THREE.Vector3(1, 0.3, 0.35).normalize() },
    uPlanetR: { value: planetR },
    uAtmoR: { value: planetR + ATMOSPHERE_TOP },
    uCamPos: { value: new THREE.Vector3() },
    // Rayleigh scattering coefficients at sea level (per meter, 550nm-ish)
    uBetaR: { value: new THREE.Vector3(5.8e-6, 13.5e-6, 33.1e-6) },
    // Mie: 4e-6 with absorption folded in (g ~ 0.76 not needed for M1)
    uBetaM: { value: new THREE.Vector3(4e-6, 4e-6, 4e-6) },
    uHR: { value: 8500 },
    uHM: { value: 1200 },
    uOrigin: { value: new THREE.Vector3() },
  };
}

export function makeAtmosphereMesh(planetR: number, uniforms: AtmosphereUniforms): THREE.Mesh {
  // Inverted hull: sphere of radius planetR + TOP, rendered BackSide so the
  // camera sees the far shell from inside AND outside.
  const geo = new THREE.SphereGeometry(planetR + ATMOSPHERE_TOP, 96, 64);
  const mat = new THREE.ShaderMaterial({
    uniforms: uniforms as unknown as { [k: string]: THREE.IUniform },
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
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
      uniform float uAtmoR;
      uniform vec3 uCamPos;
      uniform vec3 uBetaR;
      uniform vec3 uBetaM;
      uniform float uHR;
      uniform float uHM;
      uniform vec3 uOrigin;
      varying vec3 vWorld;

      // ray-sphere: returns [tNear, tFar] or (-1) miss
      vec2 raySphere(vec3 ro, vec3 rd, float r) {
        float b = dot(ro, rd);
        float c = dot(ro, ro) - r * r;
        float h = b * b - c;
        if (h < 0.0) return vec2(-1.0);
        h = sqrt(h);
        return vec2(-b - h, -b + h);
      }

      // analytic optical depth along a ray that stays in the shell
      // (Chapman-like approximation: exponential falloff integrated)
      vec3 opticalDepth(vec3 ro, vec3 rd, float tMax) {
        // sample 8 points, exponential density each
        vec3 od = vec3(0.0);
        const int N = 8;
        float dt = tMax / float(N);
        for (int i = 0; i < N; i++) {
          vec3 p = ro + rd * (float(i) + 0.5) * dt;
          float hgt = length(p) - uPlanetR;
          float dR = exp(-hgt / uHR) * dt;
          float dM = exp(-hgt / uHM) * dt;
          od += vec3(dR) + dM; // R per-channel, M gray
        }
        return vec3(od.x * uBetaR.x, od.y * uBetaR.y, od.z * uBetaR.z) + od * uBetaM * 0.0 + od * (uBetaM.x);
      }

      void main() {
        // planet center is at -origin in frame space
        vec3 pc = -uOrigin;
        vec3 ro = uCamPos - pc;
        vec3 rd = normalize(vWorld - uCamPos);

        // intersect the atmosphere shell
        vec2 tAtmo = raySphere(ro, rd, uAtmoR);
        if (tAtmo.y < 0.0) discard;
        float tStart = max(tAtmo.x, 0.0);
        float tEnd = tAtmo.y;

        // intersect the solid planet: cut the ray at the surface
        vec2 tPlanet = raySphere(ro, rd, uPlanetR * 0.999);
        bool hitsPlanet = tPlanet.x > 0.0;
        if (hitsPlanet && tPlanet.x < tEnd) tEnd = tPlanet.x;
        if (tEnd <= tStart) discard;

        // light optical depth toward the sun from a few sample points
        vec3 totalOd = vec3(0.0);
        vec3 inscatter = vec3(0.0);
        const int NS = 8;
        float dt = (tEnd - tStart) / float(NS);
        for (int i = 0; i < NS; i++) {
          vec3 p = ro + rd * (tStart + (float(i) + 0.5) * dt);
          float hgt = length(p) - uPlanetR;
          float dR = exp(-hgt / uHR) * dt;
          float dM = exp(-hgt / uHM) * dt;
          // shadow: does the sun ray from p hit the planet?
          vec2 tSun = raySphere(p, uSunDir, uPlanetR * 0.999);
          bool shadowed = tSun.x > 0.0;
          if (!shadowed) {
            // optical depth from p toward the sun (to top of atmosphere)
            vec2 tOut = raySphere(p, uSunDir, uAtmoR);
            vec3 odSun = opticalDepth(p, uSunDir, tOut.y);
            // phase functions: Rayleigh 3/16pi(1+cos^2), Mie HG g=0.76
            float mu = dot(rd, uSunDir);
            float phR = 3.0 / (16.0 * 3.14159) * (1.0 + mu * mu);
            float g = 0.76;
            float phM = 3.0 / (8.0 * 3.14159) * ((1.0 - g*g)*(1.0+mu*mu)) / ((2.0+g*g)*pow(1.0+g*g-2.0*g*mu, 1.5));
            vec3 tau = vec3(dR) * uBetaR + dM * uBetaM;
            inscatter += (phR * vec3(dR) * uBetaR + phM * dM * uBetaM) * exp(-odSun);
          }
          totalOd += vec3(dR) * uBetaR + dM * uBetaM;
        }

        // transmittance along the view ray
        vec3 T = exp(-totalOd);
        vec3 col = inscatter; // already includes phase * density * sun attenuation
        // night side: fade by sun height at the sample midpoint
        vec3 mid = ro + rd * ((tStart + tEnd) * 0.5);
        float sunH = dot(normalize(mid), uSunDir);
        col *= smoothstep(-0.25, 0.05, sunH);

        float alpha = 1.0 - T; // how much atmosphere is in front
        gl_FragColor = vec4(col * 1.0, clamp(alpha * 1.2, 0.0, 1.0));
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  mesh.renderOrder = 5; // after terrain
  return mesh;
}
