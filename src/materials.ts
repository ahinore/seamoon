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
      // M10.3: 1 = apply ACES tone mapping in-shader (custom ShaderMaterials
      // never run three's tonemap chunk); 0 = pass-through (?tonemap=0).
      uToneMap: { value: 1 },
      // M10.4: 1 = per-pixel procedural detail splatting (?detail=0 A/B).
      uDetail: { value: 1 },
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
      uniform float uToneMap;
      uniform float uDetail;
      varying vec3 vN;
      varying vec3 vC;
      varying vec3 vGrid;
      varying vec3 vCol;
      varying vec3 vWorld;

      // ACES filmic (Narkowicz approximation) — the same curve three's
      // ACESFilmicToneMapping applies to built-in materials, inlined because
      // custom ShaderMaterials skip the tonemap chunk.
      vec3 acesToneMap(vec3 x) {
        return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
      }

      // M10.4: cheap hash-based value noise for per-pixel detail splatting.
      // The vertex colors change every ~1-70 m (res grid), so between
      // vertices the ground is a smooth gradient — flat plastic look up
      // close. Two octaves of world-space noise restore texture at sub-meter
      // scale without any texture fetch (terrain is procedural; splatting a
      // bitmap would break the deterministic-noise rule anyway).
      float hash13(vec3 p) {
        p = fract(p * 0.1031);
        p += dot(p, p.zyx + 31.32);
        return fract((p.x + p.y) * p.z);
      }
      float vnoise(vec3 p) {
        vec3 i = floor(p);
        vec3 f = fract(p);
        f = f * f * (3.0 - 2.0 * f); // smoothstep fade
        float n000 = hash13(i + vec3(0.0, 0.0, 0.0));
        float n100 = hash13(i + vec3(1.0, 0.0, 0.0));
        float n010 = hash13(i + vec3(0.0, 1.0, 0.0));
        float n110 = hash13(i + vec3(1.0, 1.0, 0.0));
        float n001 = hash13(i + vec3(0.0, 0.0, 1.0));
        float n101 = hash13(i + vec3(1.0, 0.0, 1.0));
        float n011 = hash13(i + vec3(0.0, 1.0, 1.0));
        float n111 = hash13(i + vec3(1.0, 1.0, 1.0));
        return mix(
          mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
          mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
          f.z);
      }

      void main() {
        // wrapped Lambert: soft terminator instead of a hard day/night cut
        float ndl = clamp((dot(normalize(vN), uSunDir) + 0.18) / 1.18, 0.0, 1.0);
        // per-vertex terrain color (sRGB-ish values authored in linear space)
        vec3 col = vCol * (0.10 + 0.90 * ndl);

        // M10.4: per-pixel detail splatting. World-space noise (absolute pos
        // via vWorld + uOrigin) so the pattern is continuous across tile
        // boundaries, LOD transitions and floating-origin rebases. Oct1 ~3 m
        // patches (albedo mottling), oct2 ~0.5 m speckle. Weighted by view
        // distance so far tiles don't shimmer (noise < 1px aliases).
        if (uDetail > 0.5) {
          vec3 absP = vWorld + uOrigin;
          float dist = distance(uCamPos, vWorld);
          float w = clamp(1.0 - (dist - 1500.0) / 4500.0, 0.0, 1.0);
          if (w > 0.001) {
            float d1 = vnoise(absP * 0.33) - 0.5;         // ~3 m mottling
            float d2 = vnoise(absP * 2.1) - 0.5;          // ~0.5 m speckle
            col *= 1.0 + w * (d1 * 0.22 + d2 * 0.10);
          }
        }

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
        // M10.3: HDR rolloff before sRGB conversion (uToneMap note above).
        col = mix(col, acesToneMap(col), uToneMap);
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
      // M10.3: in-shader ACES toggle (custom shaders skip three's tonemap).
      uToneMap: { value: 1 },
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
      uniform float uToneMap;
      varying vec3 vN;
      varying vec3 vWorld;
      varying vec3 vAbsPos;
      varying float vDepth;
      varying float vPxPerM;

      // ACES filmic (Narkowicz) — matches the terrain shader (M10.3).
      vec3 acesToneMap(vec3 x) {
        return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
      }

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
        // M10.3: HDR rolloff (glint hotspot) before sRGB conversion.
        col = mix(col, acesToneMap(col), uToneMap);
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

/**
 * Starfield (Phase 9 M9.4). Deterministic (seeded PRNG — same sky every
 * session), magnitude-weighted (few bright / many dim, power-law), with
 * blackbody-ish color classes and a density enhancement along a galactic
 * band. Points are centered on the camera every frame (directions only).
 * ?starsize=<px> overrides the base point size (diagnosis/A-B).
 */
function urlStarSize(): number {
  const v = Number(new URLSearchParams(location.search).get('starsize') ?? '1.6');
  return Number.isFinite(v) && v > 0 ? Math.min(v, 64) : 1.6;
}
export function makeStars(count = 6000, radius = 6e8): THREE.Points {
  // mulberry32 — tiny deterministic PRNG; the sky must not reshuffle on
  // every reload (stars are "catalog" objects, not effects).
  const rnd = (() => {
    let s = 0x9e3779b9;
    return () => {
      s |= 0; s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  })();
  const pos = new Float32Array(count * 3);
  const mag = new Float32Array(count);
  const col = new Float32Array(count * 3);
  const v = new THREE.Vector3();
  // Galactic band plane: an arbitrary fixed tilt (not tied to any real
  // catalog — visual density cue only). Stars concentrate near this plane.
  const bn = v.set(0.2, 0.95, 0.35).normalize().clone();
  const w = new THREE.Vector3();
  let i = 0;
  let guard = 0;
  while (i < count && guard++ < count * 20) {
    const z = rnd() * 2 - 1;
    const phi = rnd() * Math.PI * 2;
    const r = Math.sqrt(1 - z * z);
    v.set(r * Math.cos(phi), z, r * Math.sin(phi));
    // Band acceptance: uniform background + gaussian concentration around
    // the band plane (sigma ~0.18 rad). Rejection sampling keeps directions
    // exactly uniform where the band adds nothing.
    const bandAng = Math.asin(Math.min(Math.abs(v.dot(bn)), 1));
    const weight = 1 + 2.8 * Math.exp(-(bandAng * bandAng) / (0.18 * 0.18));
    if (rnd() > weight / 3.8) continue;
    v.multiplyScalar(radius);
    pos[i * 3] = v.x;
    pos[i * 3 + 1] = v.y;
    pos[i * 3 + 2] = v.z;
    // Magnitude: power law — most stars dim, a handful bright.
    const m = Math.pow(rnd(), 4);
    mag[i] = m;
    // Color class by "temperature": blue-white / white / yellow / orange /
    // red, weighted toward white-yellow like the real sky.
    const t = rnd();
    let cr = 1, cg = 1, cb = 1;
    if (t < 0.10) { cr = 0.72; cg = 0.82; cb = 1.0; }       // blue-white
    else if (t < 0.55) { cr = 1.0; cg = 0.98; cb = 0.95; }  // white
    else if (t < 0.80) { cr = 1.0; cg = 0.93; cb = 0.80; }  // yellow
    else if (t < 0.94) { cr = 1.0; cg = 0.82; cb = 0.62; }  // orange
    else { cr = 1.0; cg = 0.72; cb = 0.58; }                // red
    col[i * 3] = cr;
    col[i * 3 + 1] = cg;
    col[i * 3 + 2] = cb;
    i++;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('aMag', new THREE.BufferAttribute(mag, 1));
  g.setAttribute('aCol', new THREE.BufferAttribute(col, 3));
  const m = new THREE.ShaderMaterial({
    uniforms: { uSize: { value: urlStarSize() } },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      #include <common>
      uniform float uSize;
      attribute float aMag;
      attribute vec3 aCol;
      varying float vMag;
      varying vec3 vCol;
      void main() {
        vMag = aMag;
        vCol = aCol;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        // Bright stars are larger points (size grows with squared magnitude).
        gl_PointSize = uSize * (0.8 + 2.6 * aMag * aMag);
      }
    `,
    fragmentShader: /* glsl */ `
      varying float vMag;
      varying vec3 vCol;
      void main() {
        // Round point sprite (fade the square edge), additive brightness.
        vec2 d = gl_PointCoord * 2.0 - 1.0;
        float fall = 1.0 - smoothstep(0.6, 1.0, length(d));
        float b = 0.30 + 1.5 * vMag * vMag;
        gl_FragColor = vec4(vCol * b * fall, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
  const pts = new THREE.Points(g, m);
  pts.frustumCulled = false;
  pts.renderOrder = -10;
  return pts;
}

/**
 * Sun disc (Phase 9 M9.4): a camera-facing quad with an analytic disk and a
 * subtle glare, positioned along the sun direction every frame (inside the
 * far plane). Depth-tested, so the planet occludes it naturally; additive so
 * the atmosphere glow layers over it. Physical angular radius from Earth is
 * ~0.267 deg (0.00465 rad) — the on-screen size comes out right without any
 * texture.
 */
export function makeSunDisc(distance = 1e9): THREE.Mesh {
  const angR = 0.00465;             // solar angular radius (rad) from 1 AU
  const halfRad = angR * 5;         // quad half-size: disk + glare margin
  const size = halfRad * distance;
  const g = new THREE.PlaneGeometry(size * 2, size * 2);
  const m = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() {
        vUv = uv * 2.0 - 1.0;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec2 vUv;
      void main() {
        float r = length(vUv);                       // 1.0 at quad edge
        float edge = 1.0 / 5.0;                      // disk edge (angR/halfRad)
        float disk = 1.0 - smoothstep(edge - 0.02, edge + 0.02, r);
        float glare = exp(-r * 8.0) * 0.25;          // soft halo
        float a = disk + glare;
        if (a <= 0.001) discard;
        // Limb darkening: hotter white core, warmer edge.
        vec3 col = mix(vec3(1.0, 0.96, 0.90), vec3(1.0, 0.85, 0.55), smoothstep(0.0, edge * 1.4, r));
        gl_FragColor = vec4(col * a, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
  const mesh = new THREE.Mesh(g, m);
  mesh.frustumCulled = false;
  mesh.renderOrder = -5; // transparent pass: stars, then sun, then atmo glow
  return mesh;
}
