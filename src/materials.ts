import * as THREE from 'three';

/**
 * Planet material: single color + Lambert term against a fixed sun direction.
 * Deliberately a custom ShaderMaterial so Phases 3-5 can replace the shading
 * without touching the render pipeline. Depth handling is reversed-Z: the
 * renderer clears depth to 0 and tests GreaterEqual, and the camera's
 * projection matrix flips the z axis, so shaders need no depth output work.
 */
export function makePlanetMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uSunDir: { value: new THREE.Vector3(1, 0.3, 0.35).normalize() },
      uBase: { value: new THREE.Color(0x1f5fa8) },
      uWire: { value: 0 },
      // Aerial perspective (Phase 5): distance fog toward the in-scattered
      // atmosphere color. Shared uniform objects with the atmosphere shell
      // so terrain and shell always agree on sun/camera/frame geometry.
      uCamPos: { value: new THREE.Vector3() },
      uOrigin: { value: new THREE.Vector3() },
      uPlanetR: { value: 6_371_000 },
      uAtmoR: { value: 6_371_000 + 60_000 },
      uBetaR: { value: new THREE.Vector3(5.8e-6, 13.5e-6, 33.1e-6) },
      uBetaM: { value: new THREE.Vector3(4e-6, 4e-6, 4e-6) },
      uHR: { value: 8500 },
      uHM: { value: 1200 },
    },
    vertexShader: /* glsl */ `
      #include <common>
      attribute vec3 center;
      attribute vec3 aGrid;
      attribute vec3 color;
      varying vec3 vN;
      varying vec3 vC;
      varying vec3 vGrid;
      varying vec3 vCol;
      varying vec3 vWorld;
      void main() {
        vN = normalize(mat3(modelMatrix) * normal);
        vC = center;
        vGrid = aGrid;
        vCol = color;
        vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uSunDir;
      uniform vec3 uBase;
      uniform float uWire;
      uniform vec3 uCamPos;
      uniform vec3 uOrigin;
      uniform float uPlanetR;
      uniform float uAtmoR;
      uniform vec3 uBetaR;
      uniform vec3 uBetaM;
      uniform float uHR;
      uniform float uHM;
      varying vec3 vN;
      varying vec3 vC;
      varying vec3 vGrid;
      varying vec3 vCol;
      varying vec3 vWorld;
      void main() {
        // wrapped Lambert: soft terminator instead of a hard day/night cut
        float ndl = clamp((dot(normalize(vN), uSunDir) + 0.18) / 1.18, 0.0, 1.0);
        // per-vertex terrain color (sRGB-ish values authored in linear space)
        vec3 col = vCol * (0.10 + 0.90 * ndl);

        // Aerial perspective: cheap single-scatter fog toward the atmosphere
        // color. Optical depth from the exponential density over the view
        // distance (no per-pixel integration — 2 samples suffice for fog).
        // Planet center is at -uOrigin in frame space; vWorld/uCamPos are
        // frame-relative, matching the atmosphere shell's frame.
        {
          vec3 pc = -uOrigin;
          vec3 ro = uCamPos - pc;
          vec3 rEnd = vWorld - pc;
          float dist = distance(uCamPos, vWorld);
          vec3 rd = (rEnd - ro) / max(dist, 1e-3);
          // midpoint density as the fog weight (fine for 60 km shell)
          vec3 mid = ro + rd * (dist * 0.5);
          float hgt = max(length(mid) - uPlanetR, 0.0);
          float dR = exp(-hgt / uHR) * dist;
          float dM = exp(-hgt / uHM) * dist;
          vec3 odView = vec3(dR * uBetaR.x, dR * uBetaR.y, dR * uBetaR.z) + dM * uBetaM;
          // light optical depth: from the midpoint toward the sun (2 samples)
          vec3 p0 = mid;
          vec3 p1 = mid + uSunDir * (uHR * 2.0);
          float h0 = max(length(p0) - uPlanetR, 0.0);
          float h1 = max(length(p1) - uPlanetR, 0.0);
          float sLen = distance(p0, p1);
          float sdR = (exp(-h0 / uHR) + exp(-h1 / uHR)) * 0.5 * sLen;
          float sdM = (exp(-h0 / uHM) + exp(-h1 / uHM)) * 0.5 * sLen;
          vec3 odSun = vec3(sdR * uBetaR.x, sdR * uBetaR.y, sdR * uBetaR.z) + sdM * uBetaM;
          float mu = dot(rd, uSunDir);
          float phR = 3.0 / (16.0 * 3.14159) * (1.0 + mu * mu);
          float g = 0.76;
          float phM = 3.0 / (8.0 * 3.14159) * ((1.0 - g*g)*(1.0+mu*mu)) / ((2.0+g*g)*pow(1.0+g*g-2.0*g*mu, 1.5));
          vec3 inscatter = (vec3(dR * uBetaR.x, dR * uBetaR.y, dR * uBetaR.z) * phR + dM * uBetaM * phM) * exp(-odSun);
          // sun height fade at the fragment (night side: fog vanishes)
          float sunH = dot(normalize(mid), uSunDir);
          inscatter *= smoothstep(-0.15, 0.1, sunH);
          float fog = clamp(1.0 - exp(-min(dR * uBetaR.x + dM * uBetaM.x, 12.0)), 0.0, 1.0);
          // never fully swallow nearby terrain; cap fog at 85%
          col = mix(col, inscatter * 1.15, min(fog, 0.85));
        }
        // Wireframe overlay drawn IN the surface shader (front faces only,
        // depth-tested). Two layers:
        //  - tile boundary lines: always (1 px), uniform at every LOD level
        //  - interior cell grid: only when a cell spans >= ~4 px on screen,
        //    fading out below that — the 1-3 px band would alias into moire
        // Skirt vertices (vGrid.z = 1) never get lines (they would read as
        // bright walls at grazing angles).
        if (uWire > 0.5) {
          float fx = fwidth(vGrid.x) + 1e-6;
          float fy = fwidth(vGrid.y) + 1e-6;
          // tile boundary: distance to the tile edge in grid units
          float b = min(min(vGrid.x, 64.0 - vGrid.x), min(vGrid.y, 64.0 - vGrid.y));
          float boundary = 1.0 - clamp(min(b / fx, b / fy), 0.0, 1.0);
          // interior grid at integer lines, anti-aliased via fwidth
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

/**
 * Ocean material (Phase 7): sea-surface sphere shell with wave shading at
 * three LODs — analytic waves (near), normal-perturbed shading (mid), and
 * pure sun-glint (far) — blended by the projected pixel size of a reference
 * 1024 m wave patch. Sun glint uses a GGX-style specular lobe; shallow water
 * lightens and foams via the per-vertex aDepth attribute (meters below the
 * surface, 0 at the shoreline).
 *
 * Wave LOD design (matches the strategy note's 3-tier plan without FFT):
 * near/mid waves are ANALYTIC and their phase is computed from the absolute
 * position modulo a fixed 1024 m wavelength — so the pattern is continuous
 * across tile boundaries, LOD transitions, and floating-origin rebases.
 */
export function makeSeaMaterial(shared: {
  uSunDir: { value: THREE.Vector3 };
  uCamPos: { value: THREE.Vector3 };
  uOrigin: { value: THREE.Vector3 };
  uPlanetR: { value: number };
  uAtmoR: { value: number };
  uBetaR: { value: THREE.Vector3 };
  uBetaM: { value: THREE.Vector3 };
  uHR: { value: number };
  uHM: { value: number };
}): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      // shared objects — updated once per frame alongside the terrain
      uSunDir: shared.uSunDir,
      uCamPos: shared.uCamPos,
      uOrigin: shared.uOrigin,
      uPlanetR: shared.uPlanetR,
      uAtmoR: shared.uAtmoR,
      uBetaR: shared.uBetaR,
      uBetaM: shared.uBetaM,
      uHR: shared.uHR,
      uHM: shared.uHM,
      uTime: { value: 0 },
      uWire: { value: 0 },
      uFovTan: { value: Math.tan(THREE.MathUtils.degToRad(60) * 0.5) },
      uViewportH: { value: 1000 },
    },
    vertexShader: /* glsl */ `
      #include <common>
      uniform float uFovTan;      // tan(fov/2)
      uniform float uViewportH;   // viewport height in px
      uniform vec3 uOrigin;       // floating origin (absolute frame offset)
      attribute vec3 center;
      attribute float aDepth;
      varying vec3 vN;
      varying vec3 vWorld;
      varying vec3 vAbsPos;
      varying float vDepth;
      varying float vPxPerM;
      void main() {
        vN = normalize(mat3(modelMatrix) * normal);
        vDepth = aDepth;
        vec3 wPos = (modelMatrix * vec4(position, 1.0)).xyz;
        vWorld = wPos;
        // absolute position for wave phase (continuous across rebases)
        vAbsPos = wPos + uOrigin;
        // meters per screen pixel at this fragment's distance
        float dist = distance(cameraPosition, wPos);
        vPxPerM = dist * 2.0 * uFovTan / uViewportH;
        vec4 mvPosition = viewMatrix * vec4(wPos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uSunDir;
      uniform vec3 uCamPos;
      uniform vec3 uOrigin;
      uniform float uPlanetR;
      uniform float uAtmoR;
      uniform vec3 uBetaR;
      uniform vec3 uBetaM;
      uniform float uHR;
      uniform float uHM;
      uniform float uTime;
      uniform float uWire;
      varying vec3 vN;
      varying vec3 vWorld;
      varying vec3 vAbsPos;
      varying float vDepth;
      varying float vPxPerM;

      // analytic Gerstner-ish wave normal: 3 octaves, phase from absolute pos
      vec3 waveNormal(vec3 n, vec3 t, vec3 b, vec3 absP, float amp) {
        vec2 d1 = vec2(0.8, 0.6);   // wave direction (unit)
        vec2 d2 = vec2(-0.6, 0.8);
        vec2 d3 = vec2(0.45, -0.89);
        float k1 = 6.2831853 / 128.0;  // 128 m swell
        float k2 = 6.2831853 / 32.0;   // 32 m chop
        float k3 = 6.2831853 / 8.0;    // 8 m ripples
        float p1 = dot(absP.xz, d1) * k1 + uTime * 1.2;
        float p2 = dot(absP.xz, d2) * k2 + uTime * 2.0;
        float p3 = dot(absP.xz, d3) * k3 + uTime * 3.1;
        // slope in tangent frame (t = east-ish, b = north-ish)
        float st = cos(p1) * k1 * amp + cos(p2) * k2 * amp * 0.5 * d2.x / d1.x + cos(p3) * k3 * amp * 0.22;
        float sb = cos(p1) * k1 * amp * d1.y + cos(p2) * k2 * amp * 0.5 * d2.y + cos(p3) * k3 * amp * 0.22 * d3.y;
        return normalize(n - t * st * 0.35 - b * sb * 0.35);
      }

      void main() {
        vec3 V = normalize(uCamPos - vWorld);
        vec3 N0 = normalize(vN);
        // tangent frame on the sphere (east/north-ish)
        vec3 up = normalize(vWorld + uOrigin);
        vec3 t0 = normalize(cross(vec3(0.0, 1.0, 0.0), up));
        vec3 b0 = cross(up, t0);

        // ---- wave LOD selection (px per reference 1024 m patch) ----
        float pxPerPatch = 1024.0 / vPxPerM;
        // wave amplitude fades with depth (surf zone damping) and with
        // grazing LOD: far view = glassy (glint only)
        float nearW = clamp((pxPerPatch - 24.0) / 200.0, 0.0, 1.0);
        float midW  = clamp((pxPerPatch - 2.0) / 60.0, 0.0, 1.0) * (1.0 - nearW);
        float amp = nearW * 0.9 + midW * 0.45;
        amp *= clamp(vDepth / 25.0, 0.15, 1.0); // calm in the surf zone

        vec3 N = N0;
        if (amp > 0.001) N = waveNormal(N0, t0, b0, vAbsPos, amp);

        // ---- water body color: deep -> shallow by depth ----
        vec3 deep = vec3(0.012, 0.055, 0.115);
        vec3 shallow = vec3(0.10, 0.42, 0.42);
        float dfac = clamp(vDepth / 45.0, 0.0, 1.0);
        vec3 body = mix(shallow, deep, sqrt(dfac));

        // ---- lighting ----
        float ndl = clamp(dot(N0, uSunDir) * 0.5 + 0.5, 0.0, 1.0);
        vec3 col = body * (0.08 + 0.92 * ndl);

        // Fresnel: sky reflection stronger at grazing angles
        float fres = 0.02 + 0.98 * pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 5.0);
        vec3 skyRef = vec3(0.35, 0.55, 0.85) * clamp(dot(N0, uSunDir) * 0.5 + 0.6, 0.0, 1.2);

        // ---- sun glint: GGX-ish specular, always on (far-LOD shading) ----
        vec3 H = normalize(V + uSunDir);
        float rough = mix(0.08, 0.38, 1.0 - nearW * 0.7); // calmer look far away
        float a2 = rough * rough;
        float ndh = clamp(dot(N, H), 0.0, 1.0);
        float d = a2 / (3.14159 * pow(ndh * ndh * (a2 - 1.0) + 1.0, 2.0));
        // tone-map the glint (smooth cap) instead of a hard clamp: keeps a
        // bright hotspot without blowing out to white streaks
        float glint = 1.0 - exp(-d * fres * 0.12);
        col += vec3(1.0, 0.95, 0.85) * glint * clamp(dot(N0, uSunDir) + 0.3, 0.0, 1.0);

        // ---- foam near the shore (depth < ~3 m) ----
        float foamBand = 1.0 - smoothstep(0.5, 3.5, vDepth);
        float foamPat = 0.6 + 0.4 * sin(dot(vAbsPos.xz, vec2(0.11, 0.07)) + uTime * 1.5);
        col = mix(col, vec3(0.9), foamBand * foamPat * 0.7);

        // combine sky reflection (Fresnel-weighted)
        col = mix(col, skyRef, fres * 0.65);

        if (uWire > 0.5) {
          col = mix(col, vec3(0.6, 0.9, 1.0), 0.4);
        }

        // night fade
        float sunH = dot(up, uSunDir);
        col *= smoothstep(-0.12, 0.08, sunH);

        // aerial perspective — same fog as terrain (frame-relative math)
        {
          vec3 pc = -uOrigin;
          vec3 ro = uCamPos - pc;
          float dist = distance(uCamPos, vWorld);
          vec3 rd = (vWorld - uCamPos) / max(dist, 1e-3);
          vec3 mid = ro + rd * (dist * 0.5);
          float hgt = max(length(mid) - uPlanetR, 0.0);
          float dR = exp(-hgt / uHR) * dist;
          float dM = exp(-hgt / uHM) * dist;
          vec3 odView = vec3(dR * uBetaR.x, dR * uBetaR.y, dR * uBetaR.z) + dM * uBetaM;
          vec3 p0 = mid;
          vec3 p1 = mid + uSunDir * (uHR * 2.0);
          float h0 = max(length(p0) - uPlanetR, 0.0);
          float h1 = max(length(p1) - uPlanetR, 0.0);
          float sLen = distance(p0, p1);
          float sdR = (exp(-h0 / uHR) + exp(-h1 / uHR)) * 0.5 * sLen;
          float sdM = (exp(-h0 / uHM) + exp(-h1 / uHM)) * 0.5 * sLen;
          vec3 odSun = vec3(sdR * uBetaR.x, sdR * uBetaR.y, sdR * uBetaR.z) + sdM * uBetaM;
          float mu = dot(rd, uSunDir);
          float phR = 3.0 / (16.0 * 3.14159) * (1.0 + mu * mu);
          float g = 0.76;
          float phM = 3.0 / (8.0 * 3.14159) * ((1.0 - g*g)*(1.0+mu*mu)) / ((2.0+g*g)*pow(1.0+g*g-2.0*g*mu, 1.5));
          vec3 inscatter = (vec3(dR * uBetaR.x, dR * uBetaR.y, dR * uBetaR.z) * phR + dM * uBetaM * phM) * exp(-odSun);
          float sh2 = dot(normalize(mid), uSunDir);
          inscatter *= smoothstep(-0.15, 0.1, sh2);
          float fog = clamp(1.0 - exp(-min(dR * uBetaR.x + dM * uBetaM.x, 12.0)), 0.0, 1.0);
          col = mix(col, inscatter * 1.15, min(fog, 0.85));
        }
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
    // The sea shell draws AFTER the terrain (renderOrder set below) and
    // still writes depth: its surface is a real opaque body. The shoreline
    // fight is won by geometry — sea skirts are dug 12 km deep, so land
    // skirts/vertices always sit in front of the sea wall behind them.
    depthWrite: true,
  });
}

/** Simple background star points (placeholder until Phase 9). */
export function makeStars(count = 2500, radius = 6e8): THREE.Points {
  const pos = new Float32Array(count * 3);
  const mag = new Float32Array(count);
  const v = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    // uniform direction sampling
    const z = Math.random() * 2 - 1;
    const phi = Math.random() * Math.PI * 2;
    const r = Math.sqrt(1 - z * z);
    v.set(r * Math.cos(phi), z, r * Math.sin(phi)).multiplyScalar(radius);
    pos[i * 3] = v.x;
    pos[i * 3 + 1] = v.y;
    pos[i * 3 + 2] = v.z;
    mag[i] = 0.35 + Math.random() * 0.65;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('aMag', new THREE.BufferAttribute(mag, 1));
  const m = new THREE.ShaderMaterial({
    uniforms: { uSize: { value: 1.6 } },
    vertexShader: /* glsl */ `
      #include <common>
      uniform float uSize;
      attribute float aMag;
      varying float vMag;
      void main() {
        vMag = aMag;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = uSize;
      }
    `,
    fragmentShader: /* glsl */ `
      varying float vMag;
      void main() {
        gl_FragColor = vec4(vec3(0.8 + 0.2 * vMag), 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
  const pts = new THREE.Points(g, m);
  pts.frustumCulled = false;
  return pts;
}
