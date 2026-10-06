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
        vec3 odR = vec3(0.0);
        float odM = 0.0;
        const int N = 8;
        float dt = tMax / float(N);
        for (int i = 0; i < N; i++) {
          vec3 p = ro + rd * (float(i) + 0.5) * dt;
          float hgt = length(p) - uPlanetR;
          float dR = exp(-hgt / uHR) * dt;
          float dM = exp(-hgt / uHM) * dt;
          odR += vec3(dR);
          odM += dM;
        }
        return odR * uBetaR + vec3(odM) * uBetaM;
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
            vec3 tau = vec3(dR * uBetaR.x, dR * uBetaR.y, dR * uBetaR.z) + dM * uBetaM;
            vec3 sc = vec3(dR * uBetaR.x, dR * uBetaR.y, dR * uBetaR.z) * phR + dM * uBetaM * phM;
            inscatter += sc * exp(-odSun);
          }
          totalOd += vec3(dR * uBetaR.x, dR * uBetaR.y, dR * uBetaR.z) + dM * uBetaM;
        }

        // transmittance along the view ray
        vec3 T = exp(-totalOd);
        vec3 col = inscatter; // already includes phase * density * sun attenuation
        // M11n: scene-exposure match. The physically-integrated inscatter
        // (~0.025 zenith blue in these units) reads near-black next to
        // sunlit terrain (~0.7). Gain + Reinhard shoulder: zenith lifts to a
        // visible blue, the bright horizon compresses instead of clipping.
        col = col * 6.0 / (1.0 + 2.2 * col);
        // night side: fade by sun height at the sample midpoint
        vec3 mid = ro + rd * ((tStart + tEnd) * 0.5);
        float sunH = dot(normalize(mid), uSunDir);
        col *= smoothstep(-0.25, 0.05, sunH);

        float alpha = 1.0 - min(min(T.x, T.y), T.z); // how much atmosphere is in front
        // M11n FIX: unpremultiply — the blend does src*alpha + dst*(1-alpha),
        // so handing out raw col darkened the sky by exactly alpha (the
        // zenith read near-black while the horizon was fine). Dividing by
        // alpha restores the physical inscatter; dst (stars) still shows
        // through at (1-alpha) = T.
        float aOut = clamp(alpha * 1.2, 0.0, 1.0);
        gl_FragColor = vec4(col / max(aOut, 0.02), aOut);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  // M11w3: 3 = before the cloud hulls (far 4, near 5). The old 5 ("after
  // terrain") was unnecessary — terrain is OPAQUE and always draws in the
  // opaque pass before every transparent object — and harmful: among
  // transparent objects the atmosphere sorted AFTER the far hull, so its
  // nearly-opaque horizon glow composited OVER the far hull's horizon-band
  // clouds. Measured at 6 km level: the far hull contributed 0/156000 band
  // pixels with the atmosphere on vs 77905/156000 with ?atmo=0 — the
  // user's "near clouds are missing near the horizon; the horizon
  // brightness paints over them". Sun/moon discs (-5/-3) and stars stay
  // behind the shell (physically correct — the air is in front of them).
  mesh.renderOrder = 3;
  return mesh;
}
