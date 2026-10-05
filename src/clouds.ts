import * as THREE from 'three';

/**
 * Cloud layer (Phase 8, M10.6: two-LOD hybrid).
 *
 * FAR (camAlt > ~60 km): flat 2D texture shell. The shell shader samples the
 * SAME density field the volumetric march uses (cloudDensity's billow fbm +
 * weather gate), evaluated at the mid-slab altitude along the fragment's
 * radial direction — so the texture IS the cloud map the near view renders,
 * and puffs sit at identical world positions in both LODs.
 *
 * NEAR (camAlt < ~60 km): raymarched VOLUMETRIC clouds inside the slab
 * [CLOUD_BOTTOM, CLOUD_TOP]. The density field is "metaball-like":
 * billowed (1-|2x-1|) fbm gives rounded, overlapping puffs; the
 * low-frequency weather field gates where puffs are allowed;
 * high-frequency noise erodes puff edges. Each sample is lit by a
 * short march toward the sun (Beer-Lambert shadowing + phase + powder
 * term) and the eye ray accumulates front-to-back with early exit —
 * real thickness, self-shadowing, silver linings.
 *
 * The LOD handoff is cross-faded: the shell's coverage fades IN from
 * 60-120 km while the volumetric term fades OUT, so the switch is
 * invisible from either side.
 */
export const CLOUD_BOTTOM = 1800; // m above sea level
export const CLOUD_TOP = 4200; // m above sea level (slab thickness)

export interface CloudUniforms {
  uSunDir: { value: THREE.Vector3 };
  uPlanetR: { value: number };
  uCamPos: { value: THREE.Vector3 };
  uOrigin: { value: THREE.Vector3 };
  uTime: { value: number };
  uCover: { value: number }; // 0..1 global coverage bias
  uVolSteps: { value: number }; // volumetric raymarch steps (quality knob)
  uTanHalfFov: { value: number }; // screen-space LOD: km per pixel
  uViewportH: { value: number };
  uCloudDbg: { value: number }; // 1 = color-code deck suppression sources
  // M11n9k: aerial perspective for the shell (shared with the atmosphere)
  uBetaR: { value: THREE.Vector3 };
  uBetaM: { value: THREE.Vector3 };
  uHR: { value: number };
  uHM: { value: number };
}

/**
 * Shared weather-field GLSL: the deck's MACRO weather (fbm2 over gradient
 * noise) as a drop-in snippet for OTHER shaders. M11n4 cloud shadows use it
 * in the terrain/sea shaders to evaluate the same system layout the cloud
 * hulls draw (sun-ray slab crossing → weatherM → deck presence), so shadows
 * line up with the visible deck without marching.
 *
 * Parameterized by a function-name prefix: consumers may already define
 * hash13 (the terrain shader's vnoise does) — 'cw' keeps the copies distinct
 * (cwfbm2 etc.).
 */
export const cloudWeatherGLSL = (p: string): string => /* glsl */ `
      float ${p}hash13(vec3 p3) {
        p3 = fract(p3 * 0.1031);
        p3 += dot(p3, p3.zyx + 31.32);
        return fract((p3.x + p3.y) * p3.z);
      }
      float ${p}gdot(float cx, float cy, float cz, float dx, float dy, float dz) {
        float z2 = ${p}hash13(vec3(cx, cy, cz)) * 2.0 - 1.0;
        float az = ${p}hash13(vec3(cx + 19.19, cy + 19.19, cz + 19.19)) * 6.2831853;
        float r2 = sqrt(max(0.0, 1.0 - z2 * z2));
        return (r2 * cos(az)) * dx + (r2 * sin(az)) * dy + z2 * dz;
      }
      float ${p}gnoise3(vec3 x) {
        x = mod(x, 2048.0);
        vec3 i = floor(x);
        vec3 f = x - i;
        vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
        vec3 i1 = mod(i + 1.0, 2048.0);
        float n000 = ${p}gdot(i.x, i.y, i.z,   f.x,     f.y,     f.z);
        float n100 = ${p}gdot(i1.x, i.y, i.z, f.x-1.0, f.y,     f.z);
        float n010 = ${p}gdot(i.x, i1.y, i.z, f.x,     f.y-1.0, f.z);
        float n110 = ${p}gdot(i1.x, i1.y, i.z, f.x-1.0, f.y-1.0, f.z);
        float n001 = ${p}gdot(i.x, i.y, i1.z, f.x,     f.y,     f.z-1.0);
        float n101 = ${p}gdot(i1.x, i.y, i1.z, f.x-1.0, f.y,     f.z-1.0);
        float n011 = ${p}gdot(i.x, i1.y, i1.z, f.x,     f.y-1.0, f.z-1.0);
        float n111 = ${p}gdot(i1.x, i1.y, i1.z, f.x-1.0, f.y-1.0, f.z-1.0);
        return mix(
          mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
          mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
          u.z) * 1.15;
      }
      float ${p}fbm2(vec3 pw) {
        return 0.6 * (0.5 + ${p}gnoise3(pw) * 1.1)
             + 0.3 * (0.5 + ${p}gnoise3(pw * 2.13) * 1.1) + 0.05;
      }
      // M11n9d: value-noise octave matching clouds.ts noise3 exactly (same
      // hash13 lattice + wrap) — the region/body masks must be the SAME
      // field the cloud hulls draw, or shadows drift off the clouds
      float ${p}noise3(vec3 x) {
        x = mod(x, 2048.0);
        vec3 i = floor(x);
        vec3 f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        vec3 i1 = mod(i + 1.0, 2048.0);
        return mix(
          mix(mix(${p}hash13(i), ${p}hash13(i1), f.x),
              mix(${p}hash13(vec3(i.x, i1.y, i.z)), ${p}hash13(vec3(i1.x, i1.y, i.z)), f.x), f.y),
          mix(mix(${p}hash13(vec3(i.x, i.y, i1.z)), ${p}hash13(vec3(i1.x, i.y, i1.z)), f.x),
              mix(${p}hash13(vec3(i.x, i1.y, i1.z)), ${p}hash13(i1), f.x), f.y),
          f.z);
      }
      float ${p}fbm4(vec3 pw) {
        float a = 0.5, s = 0.0;
        for (int i = 0; i < 4; i++) { s += a * ${p}noise3(pw); pw *= 2.13; a *= 0.5; }
        return s;
      }
`;

export function makeCloudUniforms(planetR: number): CloudUniforms {
  return {
    uSunDir: { value: new THREE.Vector3(1, 0.3, 0.35).normalize() },
    uPlanetR: { value: planetR },
    uCamPos: { value: new THREE.Vector3() },
    uOrigin: { value: new THREE.Vector3() },
    uTime: { value: 0 },
    uCover: { value: 0.42 },
    uVolSteps: { value: 18 },
    // M11n9k: aerial perspective for the shell — the betas/scale heights are
    // SHARED with the atmosphere/terrain so the distant clouds wash into the
    // horizon haze exactly like the terrain does
    uBetaR: { value: new THREE.Vector3(5.8e-6, 13.5e-6, 33.1e-6) },
    uBetaM: { value: new THREE.Vector3(4e-6, 4e-6, 4e-6) },
    uHR: { value: 8500 },
    uHM: { value: 1200 },
    uTanHalfFov: { value: Math.tan((60 * Math.PI) / 360) },
    uViewportH: { value: 900 },
    uCloudDbg: { value: 0 },
  };
}

export function makeCloudMesh(planetR: number, uniforms: CloudUniforms): THREE.Group {
  // TWO HULLS share this material family. The single 9.8 km shell made the
  // NEAR view broken: from 8 km looking down, every shell fragment lies
  // BEHIND the ground (the ray exits the 9.8 km sphere beyond the planet
  // horizon), so depth culling removed the whole deck and the volumetric
  // march — which lives in the shell's fragment shader — never ran (the
  // deck hugged the horizon only). Geometry:
  //   FAR hull  @ R+9.8 km — above all terrain; owns the far map (wShell).
  //   NEAR hull @ R+2.6 km — inside the slab; downward rays from any flight
  //   altitude cross it BEFORE the ground, and uplooking rays from below
  //   the deck cross it overhead. Owns the volumetric march (wVol).
  const uNear = { value: 0 };
  const makeMat = (near: number) => {
    const m = new THREE.ShaderMaterial({
      uniforms: { ...(uniforms as unknown as { [k: string]: THREE.IUniform }), uNearHull: { value: near } },
      // DoubleSide: back hull serves ground-level views, front hull space views
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
      fragmentShader: FRAGMENT,
    });
    return m;
  };
  void uNear;
  const FRAGMENT = /* glsl */ `
      uniform vec3 uSunDir;
      uniform float uPlanetR;
      uniform vec3 uCamPos;
      uniform vec3 uOrigin;
      uniform float uTime;
      uniform float uCover;
      uniform float uVolSteps;
      uniform float uTanHalfFov;
      uniform float uViewportH;
      uniform float uCloudDbg;
      uniform float uNearHull;
      uniform vec3 uBetaR;
      uniform vec3 uBetaM;
      uniform float uHR;
      uniform float uHM;
      varying vec3 vWorld;

      // Float32-safe hash (IQ): fract() FIRST bounds every intermediate,
      // so it stays valid for lattice coordinates up to ~2048. The naive
      // product-form hash degenerates to a constant for large inputs
      // (planet-frame positions -> 1e13 products -> fract() garbage).
      float hash13(vec3 p3) {
        p3 = fract(p3 * 0.1031);
        p3 += dot(p3, p3.zyx + 31.32);
        return fract((p3.x + p3.y) * p3.z);
      }
      float noise3(vec3 x) {
        // wrap the lattice into [0,2048) cells: bounded magnitudes for the
        // hash (planet-frame coords arrive here), continuous at the seams
        // (floor/fract both wrap consistently), no visible repetition —
        // fbm octaves scale coordinates by 2.17, so the composite field's
        // effective periods are incommensurate.
        // M10.7 fix: the NEIGHBOR references must wrap too. x=mod(x,2048)
        // alone left i+1 == 2048 un-wrapped, so hash(2048,y,z) != hash(0,y,z)
        // and the field jumped across EVERY multiple of 2048 — including the
        // x/y/z = 0 planes, which from orbit are straight cloud cuts along
        // the equator and the prime meridian (the user's "torn in half" line).
        x = mod(x, 2048.0);
        vec3 i = floor(x);
        vec3 f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        vec3 i1 = mod(i + 1.0, 2048.0); // wrapped +1 neighbors
        return mix(
          mix(mix(hash13(i), hash13(i1), f.x),
              mix(hash13(vec3(i.x, i1.y, i.z)), hash13(vec3(i1.x, i1.y, i.z)), f.x), f.y),
          mix(mix(hash13(vec3(i.x, i.y, i1.z)), hash13(vec3(i1.x, i.y, i1.z)), f.x),
              mix(hash13(vec3(i.x, i1.y, i1.z)), hash13(i1), f.x), f.y),
          f.z);
      }
      // Gradient (Perlin) noise: lattice nodes carry a random UNIT VECTOR
      // and contribute dot(grad, offset) — ZERO at every node. Value noise
      // instead plateaus AT the node with the full hash value, so any
      // threshold slices rows of identical round blobs out of the node
      // lattice ("clouds in a grid", "holes in a grid"). Gradient noise
      // structurally cannot produce node-aligned blobs: extrema sit BETWEEN
      // nodes as ridges. Costs 8 hash13 + 8 dots per eval (vs 8 hashes).
      float gdot(float cx, float cy, float cz, float dx, float dy, float dz) {
        float z2 = hash13(vec3(cx, cy, cz)) * 2.0 - 1.0;
        float az = hash13(vec3(cx + 19.19, cy + 19.19, cz + 19.19)) * 6.2831853;
        float r2 = sqrt(max(0.0, 1.0 - z2 * z2));
        return (r2 * cos(az)) * dx + (r2 * sin(az)) * dy + z2 * dz;
      }
      float gnoise3(vec3 x) {
        x = mod(x, 2048.0);
        vec3 i = floor(x);
        vec3 f = x - i;
        vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0); // quintic fade
        vec3 i1 = mod(i + 1.0, 2048.0);
        float n000 = gdot(i.x, i.y, i.z,   f.x,     f.y,     f.z);
        float n100 = gdot(i1.x, i.y, i.z, f.x-1.0, f.y,     f.z);
        float n010 = gdot(i.x, i1.y, i.z, f.x,     f.y-1.0, f.z);
        float n110 = gdot(i1.x, i1.y, i.z, f.x-1.0, f.y-1.0, f.z);
        float n001 = gdot(i.x, i.y, i1.z, f.x,     f.y,     f.z-1.0);
        float n101 = gdot(i1.x, i.y, i1.z, f.x-1.0, f.y,     f.z-1.0);
        float n011 = gdot(i.x, i1.y, i1.z, f.x,     f.y-1.0, f.z-1.0);
        float n111 = gdot(i1.x, i1.y, i1.z, f.x-1.0, f.y-1.0, f.z-1.0);
        return mix(
          mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
          mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
          u.z) * 1.15; // Perlin 3D range ~±0.87 → rescale toward ±1
      }

      float fbm3o(vec3 p, float detail) {
        // detail fades the 3rd octave's amplitude (LOD) instead of rescaling
        // the lattice: rescaling planet-frame coordinates per pixel jumps the
        // noise grid by whole cells -> concentric ripple rings. Amplitude
        // fade is continuous in the fade value, so it cannot ring.
        // M11n9t3 PERF: below detail 0.25 the 3rd octave's amplitude is
        // < 6% of the first — skip its noise eval entirely (the density
        // chain calls this per march step; far samples all qualify).
        float a = 0.5, s = 0.0;
        s += a * noise3(p); p *= 2.17; a *= 0.5;
        s += a * noise3(p); p *= 2.17; a *= 0.5 * detail;
        if (a > 0.0125) s += a * noise3(p);
        return s;
      }
      // Shell-specific: THREE WORLD-ANCHORED octave cells (70/32.2/14.8 km),
      // whose amplitudes fade when their cell drops below ~16 px on screen
      // (full at 16px, zero at 6px). The lattice is FIXED in world space —
      // the field at a world point NEVER depends on the camera — so the
      // deck cannot be "rebuilt" while approaching; only whole octaves
      // fade in, anchored to the same world features. (The previous
      // cellKm=kmPerPx*30 lattice RESCALED the world→cell mapping with
      // altitude: 8% of world points re-crossed the threshold per altitude
      // step in the 2.58Mm→224km band — clouds visibly rebuilt at every
      // altitude change, stable only where cellKm clamped at 70 or 6 km.)
      // De-latticing, two layers:
      //   1. GRADIENT NOISE (gnoise3) — lattice nodes carry dot(grad,off)=0,
      //      so no threshold can slice round blobs out of the node lattice
      //      (value noise plateaus AT nodes → the "clouds/holes in a grid").
      //   2. DOMAIN WARP + PER-OCTAVE ROTATION — breaks any remaining
      //      alignment. Both world-fixed (no camera dependence) & continuous.
      // Shell-specific FAR MAP. p comes in world-anchored (70 km units) but
      // the caller may pass an ANISOTROPICALLY stretched coordinate (banding).
      // The two coarsest octaves (330/152 km anchors) deliberately sample
      // the UN-stretched sphere direction pU instead: the anisotropy
      // multiplies the lattice pitch (330/0.62 = 532 km = 85 px at 4.7 Mm)
      // and the anchor's per-cell minima become a visible 89 px dot grid
      // in vertical columns (the user's screenshot). Coarse weather
      // structure stays isotropic; only the band-forming octaves stretch.
      //
      // M10.9d ANTI-FLICKER: the old OCT_BLEND keyed each octave's blend
      // to kmPerPx (camera altitude) with an 8-px-wide fade band — during
      // approach the 32/15 km octave mixes swept CONTINUOUSLY, and every
      // deck-edge pixel whose coarse/fine values straddled the threshold
      // flipped in/out frame after frame (the "snow flicker at the cloud
      // edges" the user filmed at 1.3-1.5 Mm). Fix: key each octave to
      // ALTITUDE with a NARROW crossfade at a fixed altitude, where the
      // octave's cells are still 25-50 px (no speckle at the switch).
      // Between switches the field is EXACTLY camera-independent — zero
      // shimmer. An octave drops out where its cell would be <~20 px.
      // camAltKm = camera altitude above the surface in km.
      float fbm3oLod(vec3 p, vec3 pU, float camAltKm, float detail,
                     out float ampSum) {
        // gate: 1 = octave present. Narrow altitude crossfades (±15%).
        //   15 km oct:  out above 400 km   (cell 43 px there)
        //   32 km oct:  out above 1.0 Mm   (cell 66 px there)
        //   70 km oct:  out above 3.2 Mm   (cell 60 px there)
        //   152 km oct: out above 9.0 Mm   (cell 58 px there)
        #define OCT_GATE(loKm, hiKm) smoothstep(hiKm, loKm, camAltKm)
        float g0 = OCT_GATE(300.0, 420.0);   // 15 km octave
        float g1 = OCT_GATE(700.0, 1000.0);  // 32 km octave
        float g2 = OCT_GATE(2200.0, 3200.0); // 70 km octave
        float g3 = OCT_GATE(6000.0, 9000.0); // 152 km octave
        #undef OCT_GATE
        vec3 w = (gnoise3(pU * 0.461 + 7.7) * 0.45
                + gnoise3(pU + 31.7) * 0.30) * vec3(1.0, 0.8, 1.1);
        // octave 4 (330 km) — ANCHOR: the coarsest octave is never blended
        // away, so every finer octave always has a coarser partner to
        // converge into. THREE components at INCOMMENSURATE frequencies
        // (1 : √2 : 1.7 rotated): equal-frequency lattices beat with a
        // periodic pattern — aligned minima recur on a grid and read as
        // rows of translucent gray dots after the dip floor. Irrational
        // ratios destroy the global period, so minima never line up.
        float v4 = 0.5 + (gnoise3(pU * 0.2135 + 41.3)
                        + gnoise3(vec3(pU.z, pU.x, pU.y) * 0.3019 + 13.9)
                        + gnoise3(pU * 0.3630 + 77.7)) * 0.40;
        // octave 3 (152 km): fades out above ~6-9 Mm (converges into the
        // 330 km anchor through the crossfade)
        float b3 = g3;
        float v3 = mix(v4, 0.5 + gnoise3(pU * 0.461 + 7.7) * 1.2, b3);
        // octave 2 (70 km): fades out above ~2.2-3.2 Mm
        float b2 = g2;
        float v2 = mix(v3, 0.5 + gnoise3(p + w) * 1.2, b2);
        // octave 1 (32 km): fades out above ~0.7-1.0 Mm
        float b1 = min(g1, b2);
        vec3 p2 = vec3(p.y, p.z, p.x);
        float v1 = mix(v2, 0.5 + gnoise3(p2 * 2.17 + w) * 1.2, b1);
        // octave 0 (15 km): fades out above ~300-420 km, detail-gated
        float b0 = min(g0, b1) * detail;
        vec3 p3r = vec3(p.z, p.x, p.y);
        float v0 = mix(v1, 0.5 + gnoise3(p3r * 4.71 + w) * 1.2, b0);
        ampSum = 1.0;
        return 0.20 * v4 + 0.26 * v3 + 0.30 * v2 + 0.14 * v1 + 0.10 * v0;
      }
      float fbm4(vec3 p) {
        float a = 0.5, s = 0.0;
        for (int i = 0; i < 4; i++) { s += a * noise3(p); p *= 2.13; a *= 0.5; }
        return s;
      }
      // 2-octave macro weather: gate/cover ride this (see main()). GRADIENT
      // noise — value-noise node plateaus carved PERIODIC round gate-holes
      // (the 'gray dot grid': dark terrain seen through shell alpha=0 where
      // the gate suppresses the deck in value-noise plateau wells).
      float fbm2(vec3 p) {
        return 0.6 * (0.5 + gnoise3(p) * 1.1)
             + 0.3 * (0.5 + gnoise3(p * 2.13) * 1.1) + 0.05;
      }

      // ray-sphere: [tNear, tFar] or (-1,-1) on miss
      vec2 raySphere(vec3 ro, vec3 rd, float r) {
        float b = dot(ro, rd);
        float c = dot(ro, ro) - r * r;
        float h = b * b - c;
        if (h < 0.0) return vec2(-1.0);
        h = sqrt(h);
        return vec2(-b - h, -b + h);
      }

      // ---- unified cloud density ---------------------------------------
      // ONE function drives the volumetric march. billowed (1-|2x-1|) fbm
      // gives rounded, overlapping puffs; the coverage threshold picks the
      // peaks; the weather field gates clusters (systems, not a carpet).
      // windOff: drift in METERS, divided by the cell size here (3000 m) —
      // the SAME physical wind speed the far shell uses (see main()).
      float cloudDensity(vec3 p, float cover, float weather, vec3 windOff, float edge, float detail, float weatherM2) {
        vec3 pw = p * (1.0 / 3000.0) + windOff * (1.0 / 3000.0); // puff cells ~3 km
        float f1 = fbm3o(pw, detail);
        float billow = 1.0 - abs(2.0 * f1 - 1.0); // rounded blobs [0,1]
        // Billow field statistics (200k samples): mean 0.19, median 0.23,
        // q70 0.52, q85 0.76.
        // M11n9 REDESIGN — limited regions + discrete bodies + towering
        // cumulonimbus. The old region/fine/floor chain always left a deck
        // veil covering the ground at every altitude (user report: "the
        // ground is completely covered — rework the cloud implementation
        // from scratch"). The new structure:
        //   REGION (60 km cells, ~25% of the planet): where clouds exist
        //   at all — everywhere else is clear sky, matching the far view.
        //   BODY (4 km cells): discrete cloud bodies within a region.
        //   TOWER (9 km cells, ~30% of the bodies): cumulonimbus towers
        //   rising from the 1.8 km base to ~12 km — thick vertical cores
        //   (入道雲).
        float thr = mix(0.45, 0.30, cover);
        float d = smoothstep(thr, thr + edge, billow * (0.72 + 0.28 * cover));
        d = clamp(d * 1.35, 0.0, 1.0);
        // Cluster gate: the wide gate grows proper clusters; below it a
        // thin sparse haze keeps clear skies in genuinely dry troughs.
        float wx = max(weather, weatherM2);
        float gate = max(smoothstep(0.40, 0.58, wx),
                         0.30 * smoothstep(0.15, 0.35, wx));
        d *= gate;
        float regionN = noise3(p * (1.0 / 60000.0) + windOff * (1.0 / 60000.0));
        float region = smoothstep(0.50, 0.60, regionN);
        // M11n9l: the body duty dropped to ~12% (threshold raised again) —
        // the Ace-Combat look needs DISCRETE bodies with blue gaps even on
        // horizontal views at the tower-band altitude: a 32 km horizontal
        // ray crosses ~8 body cells, and any duty > 20% saturates the whole
        // horizon into a white sheet (measured 90% cover at 8.7 km)
        float bodyN = noise3(p * (1.0 / 4000.0) + windOff * (1.0 / 4000.0));
        float body = smoothstep(0.55, 0.66, bodyN);
        float towerN = noise3(p * (1.0 / 9000.0) + windOff * (1.0 / 9000.0));
        // M11n9l: towers are RARE GIANTS (~2% of 9 km cells) — a horizontal
        // ray at the tower-band altitude crosses ~19 cells, and any duty
        // above ~5% saturates the whole sky into white (measured 64% cover
        // at 8.7 km even with the tower at 10%). Rare towers keep the
        // Ace-Combat look: blue sky with a few massive storm columns.
        float tower = smoothstep(0.72, 0.88, towerN);
        // vertical profile in slab units: hE 0 = 1.8 km base, 1 = 4.2 km
        // (the old deck top), 4.3 = ~12 km (tower top).
        // M11n9b: the base deck fades 0.70-2.2 (up to ~7 km) instead of
        // hard-stopping at 1.1 — horizon rays from 8-11 km sample the field
        // FAR away where the ray altitude is 6-11 km; with the old 1.1 cap
        // every horizon sample was outside the deck and the horizon
        // atmosphere band showed no clouds at all (user report).
        float hE = (length(p) - uPlanetR - ${CLOUD_BOTTOM.toFixed(1)}) /
                   ${((CLOUD_TOP - CLOUD_BOTTOM)).toFixed(1)};
        float vertBase = smoothstep(0.0, 0.10, hE) * (1.0 - smoothstep(0.70, 2.20, hE));
        float vertTower = smoothstep(0.0, 0.30, hE) * (1.0 - smoothstep(3.20, 4.40, hE));
        // M11n9f ANVIL: real cumulonimbus spread out at the top — above
        // hE 2 (~6.6 km) the tower mask widens (lower threshold on the
        // same 9 km lattice) so towers flare outward into anvil caps
        // instead of ending as vertical columns.
        // M11n9l fix: the anvil threshold kept ~40% of the sky covered at
        // the camera's own altitude band (hE 2.5-3.5 = 8-10 km) — flying
        // there the whole sky filled with flat anvil sheet (user report).
        // Narrower anvil mask (0.48-0.60 → ~20% sky) keeps open blue
        // between the caps.
        // M11n9l fix: the anvil mask is RARER than the tower (0.78-0.92 vs
        // 0.72-0.88) — only the greatest towers get caps.
        float towerAnvil = smoothstep(0.78, 0.92, towerN);
        float towerMask = mix(tower, towerAnvil, smoothstep(2.0, 3.0, hE));
        float vert = max(vertBase, vertTower * towerMask);
        // M11n9v RAIN SHAFTS: precipitation curtains hang below the
        // strongest towers — narrower than the tower (0.78-0.88 on the same
        // 9 km lattice, no new noise), from just under the cloud base down
        // to hE -0.65 (~240 m AGL), fading at both ends. Scaled 0.45 so the
        // curtain stays translucent gray-blue (the ambient shading handles
        // the darkness — rain samples sit below the base where the
        // height-ambient term is 0).
        float rainN = smoothstep(0.78, 0.88, towerN);
        float vertRain = (1.0 - smoothstep(-0.65, -0.20, hE)) * smoothstep(0.05, -0.05, hE);
        vert = max(vert, vertRain * rainN * 0.45);
        d *= region * mix(0.55, 1.0, body) * vert;
        return d;
      }
      // M11n9b v10 probe: diagnose each gate layer for the horizon band
      float cloudGateDebug(vec3 p, float cover, float weather, vec3 windOff, float weatherM2, out float oRegion, out float oBody, out float oVert) {
        float regionN = noise3(p * (1.0 / 60000.0) + windOff * (1.0 / 60000.0));
        oRegion = smoothstep(0.50, 0.60, regionN);
        float bodyN = noise3(p * (1.0 / 4000.0) + windOff * (1.0 / 4000.0));
        oBody = smoothstep(0.44, 0.56, bodyN);
        float hE = (length(p) - uPlanetR - ${CLOUD_BOTTOM.toFixed(1)}) /
                   ${((CLOUD_TOP - CLOUD_BOTTOM)).toFixed(1)};
        float vertBase = smoothstep(0.0, 0.10, hE) * (1.0 - smoothstep(0.70, 2.20, hE));
        float vertTower = smoothstep(0.0, 0.30, hE) * (1.0 - smoothstep(3.20, 4.40, hE));
        float towerN = noise3(p * (1.0 / 9000.0) + windOff * (1.0 / 9000.0));
        float tower = smoothstep(0.55, 0.72, towerN);
        oVert = max(vertBase, vertTower * tower);
        return oRegion * mix(0.55, 1.0, oBody) * oVert;
      }
      // M11n9: the extended slab-unit height (0 = base … 4.3 = tower top),
      // exposed for the march's ambient term
      float cloudHeightE(vec3 p) {
        return (length(p) - uPlanetR - ${CLOUD_BOTTOM.toFixed(1)}) /
               ${((CLOUD_TOP - CLOUD_BOTTOM)).toFixed(1)};
      }
      void main() {
        vec3 pc = -uOrigin;
        vec3 ro = uCamPos - pc;      // ray origin in planet frame
        vec3 rd = normalize(vWorld - uCamPos);
        float camAlt = length(ro) - uPlanetR;
        vec3 up0 = normalize(ro);
        // (M11n3b: hLayer removed — it fed only the dead inBand term; with
        // the near hull camera-following its value was direction-dependent
        // anyway and never meaningful.)

        // M10.9: single-owner handoff — the near hull owns the volumetric
        // march, the far hull owns the texture map. Their radii differ
        // (R+2.6 km vs R+9.8 km), so any altitude band where BOTH paint
        // shows the same cloud system twice with parallax.
        // Single-owner handoff: the volumetric fades out by 25 km and the
        // far map takes over there. The old 30-90 km cross-fade let BOTH
        // hulls paint simultaneously — the far map lives at R+9.8 km and
        // the volumetric deck at R+2.6 km, so in the overlap band the same
        // cloud system appeared TWICE with parallax (the "second, different
        // cloud layer below the far clouds").
        // M11n9b: the handoff moved DOWN to the tower tops. wVol now fades
        // 8-12 km (the tower band) and wShell takes over above 12 km —
        // at 11 km looking at the horizon the old gates left a dead band:
        // the march faded out, the shell hadn't started, and the horizon
        // atmosphere band showed no clouds at all (user report).
        float wVol = (1.0 - smoothstep(8000.0, 12000.0, camAlt)) * step(0.001, uVolSteps);

        // ---------------- weather / coverage ----------------
        vec3 upF = normalize(vWorld - pc);
        // M11n9k: the ray's TRUE distance to the slab-mid crossing — the
        // shell map's clouds live there, NOT at the hull fragment's own
        // position (the near hull follows the camera, so its fragments are
        // only 2.6 km away while the mapped clouds are 100-300 km out).
        // The aerial-perspective fog must use this distance or the far
        // wisps stay crisp (user report: the wisps survived the M11n9k fog).
        float cloudDist = -1.0;
        // Weather anchor: the direction where the ray MEETS THE SLAB, not
        // the fragment's own sky direction. From below/at deck level the
        // shell fragment sits 10-60 km up-sky from the actual march column
        // (a 9° ray crosses the slab ~10 km out but exits the 9.8 km shell
        // ~60 km out); anchoring weather to the FRAGMENT sampled a different
        // weather cell than the deck the camera flies through — the deck
        // overhead could be wet while the anchored cell was dry, so clouds
        // VANISHED as you descended under them. Slab-midpoint intersection
        // along the ray is the fair compromise for both LODs.
        vec3 upW = upF;
        {
          float rMid = uPlanetR + ${((CLOUD_BOTTOM + CLOUD_TOP) / 2).toFixed(1)};
          float b = dot(ro, rd);
          float c = dot(ro, ro) - rMid * rMid;
          float hq = b * b - c;
          if (hq >= 0.0) {
            float sq = sqrt(hq);
            float tMid = (-b - sq) > 0.0 ? (-b - sq) : (-b + sq);
            if (tMid > 0.0) {
              upW = normalize(ro + rd * tMid);
              cloudDist = tMid; // M11n9k: the true mapped-cloud distance
            }
          }
        }
        vec3 q = upW * 2.2; // weather scale ~ R/2.2
        // Wind drift in METERS (same physical speed for every LOD): the old
        // per-use offsets (uTime*0.006 lattice cells) drifted the far map's
        // 70 km cells at ~420 m/s — clouds visibly crawled/spread across the
        // globe. 4.5 m/s (gentle breeze): the far map slides its features at
        // wind*pxPerMeter — at 218 km that was 1.9 px/30s with 18 m/s, and
        // since the deck edge is only 2-3 px wide, every edge pixel repainted
        // → reads as NOISE-like shimmer ("clouds start moving when close,
        // stop when far"). 4.5 m/s puts even the 100-300 km band below the
        // shimmer threshold (~0.5 px/30s) while keeping a slow drift.
        float driftM = uTime * 4.5;
        vec3 wind = vec3(driftM, driftM * 1.3, -driftM * 0.8);
        // Weather drift: wTime was uTime*0.002 q-units/s. One q-unit spans
        // ~4600 km of globe, so the weather gate slid at ~9 km/s — cloud
        // systems visibly crawled and morphed everywhere ("the deck boils
        // when you approach"). 2e-5 q-units/s ≈ 92 m/s: calm jet-stream.
        float wTime = uTime * 2e-5;
        // MACRO WEATHER (2 octaves): the gate and cover thresholds ride
        // this, NOT the 4-octave weather. fbm4's finest octave (~300 km
        // value-noise cells) carved round pinholes into the deck wherever
        // its plateaus dipped below the gate — "round holes in rows".
        // The 2-octave field only shapes ~1000-3000 km systems; the fine
        // octave still modulates the density INSIDE systems (f1 term), so
        // texture is preserved but holes stop being round plateaus.
        float weatherM = fbm2(q + vec3(wTime, wTime * 0.7, -wTime * 0.6));
        float weather = fbm4(q + vec3(wTime, wTime * 0.7, -wTime * 0.6));
        float lat = asin(clamp(upF.y, -1.0, 1.0));
        float bands = 0.55 + 0.45 * cos(lat * 6.0) * 0.5 + 0.25 * exp(-pow((abs(lat) - 0.15) * 3.0, 2.0));
        float cover = clamp(uCover * bands * 1.6 * weatherM + (weatherM - 0.5) * 0.4, 0.0, 1.0);
        cover = pow(cover, 0.7); // bias toward more visible coverage

        // MACRO COVERAGE shared by both LODs: the shell's sys/thr formula
        // evaluated at the weather anchor direction. The far shell defines
        // the deck's macro layout from this; coupling the volumetric march
        // to the same term makes the near view AS DENSE AS THE FAR MAP —
        // before this, the march's own sparse puff threshold meant the
        // deck thinned drastically as the shell faded out (25→8 km), which
        // read as "clouds disappear when you descend / near clouds pale".
        float macroThr = mix(0.60, 0.50, cover);
        float macroGate = max(smoothstep(0.34, 0.55, weatherM),
                              0.12 * smoothstep(0.15, 0.35, weatherM));
        float dMacro = smoothstep(macroThr, macroThr + 0.16,
                                  max(0.62 * smoothstep(0.30, 0.62, weatherM) + 0.38 * weatherM * 0.5,
                                      0.62 * smoothstep(0.30, 0.62, weatherM) - 0.10));
        dMacro = pow(dMacro, 0.45) * macroGate;
        float dMacroFrag = weatherM; // probe: anchor weather visibility

        // ============ M10.7 hybrid: shell + volumetric ==================
        // Complementary handoff: wShell = 1 - wVol (above the near gate), so
        // at EVERY altitude one of the two LODs is at full weight and total
        // cloud coverage never dips (the gap the user saw at 48-60 km).
        // Both evaluate the SAME density field at the slab middle, so the
        // handoff just swaps WHO draws the same clouds.
        // M11n9b: the shell starts at the TOWER TOPS (12 km) — its geometry
        // sits at 12 km now, so painting from 8 km up (overlapping the
        // march's fade) keeps the horizon band populated at every altitude
        float wShell = (1.0 - wVol) * smoothstep(6000.0, 10000.0, camAlt);
        vec3 shellCol = vec3(0.0);
        float shellA = 0.0;
        float shellAraw = 0.0; // M11n9c: shell alpha without the altitude weight — the far-band fill uses it
        // M11n9o: the shell block runs on the FAR HULL ONLY now — the near
        // hull is march-only (its shell contributions are zeroed below).
        // The far hull always computes: above 12 km it owns the full map,
        // below it owns the horizon band (see the in-zone gate inside).
        if (uNearHull < 0.5) {
          // M11n9t2 PERF early-out: in-zone only the horizon band's pixels
          // can show shell clouds (zoneBand gates the density to the band);
          // off-zone wShell owns the full map. When both weights are ~0 the
          // whole block — 6+ noise evals plus the 2-sample fog march —
          // would be pure waste per pixel, and inside the deck / on the
          // ground that is most of the screen. The gates need only rd/ro/
          // camAlt (cheap trig), so they hoist above the block; the inner
          // code reuses them.
          float elevR = asin(clamp(dot(rd, normalize(ro)), -1.0, 1.0));
          float bandGate = 1.0 - smoothstep(0.035, 0.075, elevR);
          float overlap = mix(0.45, 1.0, smoothstep(4000.0, 12000.0, camAlt));
          float zoneBand = (1.0 - smoothstep(11000.0, 13000.0, camAlt)) * bandGate * overlap;
          if (wShell > 0.001 || zoneBand > 0.001) {
          // M11n6c PARALLAX FIX: the shell paints its clouds at the
          // fragment's own direction (upF) on the R+9.8 km shell, but the
          // clouds it depicts live in the 1.8-4.2 km slab. Along a view ray
          // the slab crossing is at direction upW, so evaluating the field
          // at upF displaced the far map outward — when the march took over
          // (18-25 km handoff) the same clouds snapped to their true slab
          // positions, the "near and far clouds at different positions"
          // jump. Anchoring the field at upW paints each cloud where the
          // march will draw it; from orbit (upW ≈ upF) nothing changes.
          vec3 shellP = upW * (uPlanetR + ${((CLOUD_BOTTOM + CLOUD_TOP) / 2).toFixed(1)});
          // M10.9d: octave LOD is keyed to ALTITUDE (see fbm3oLod) — the
          // field no longer depends on kmPerPx, so nothing re-sweeps with
          // camera distance and the deck-edge snow cannot shimmer.
          float camAltKm = camAlt / 1000.0;
          // FAR MAP: the near-view billow lattice (3/1.4/0.64 km cells) is
          // SUB-PIXEL from orbit (5-50 km/px) — thresholding it produces
          // organized moiré dots/rings (the "regular holes" the user saw).
          // The far shell instead thresholds a low-frequency billow whose
          // cells stay above the pixel grid: same weather gate, same cover
          // bias, so the deck's MACRO layout matches the near view while
          // Cell size follows the PIXEL GRID: base cell ≈ 30 px at any
          // altitude (clamped 6-70 km). A fixed 70 km lattice made 140-470 px
          // flat plates at 100-400 km altitudes (the block noise the user
          // flagged); screen-relative sizing keeps features readable and the
          // octave fade (fbm3oLod) kills anything approaching the pixel grid.
          // WORLD-ANCHORED lattice: fixed 70 km base cell (octaves 152/70/
          // 32/15 km inside fbm3oLod), independent of the camera. Wind
          // drift in world meters divided by the 70 km cell — the same
          // physical speed every LOD uses.
          // ANISOTROPY: rotate 30° about the pole axis then stretch y by
          // 0.45 — islands become elongated banded systems (ITCZ/frontal
          // bands) instead of round blobs. World-fixed and continuous.
          const float cosB = 0.8660254, sinB = 0.5;
          vec3 spR = shellP;
          vec3 spX = vec3(spR.x * cosB - spR.z * sinB, spR.y,
                          spR.x * sinB + spR.z * cosB);
          vec3 pwF = vec3(spX.x * 0.62, spX.y * 1.38, spX.z * 0.62) *
                     (1.0 / 70000.0) + wind * (1.0 / 70000.0);
          vec3 pwU = shellP * (1.0 / 70000.0) + wind * (1.0 / 70000.0);
          float ampSum;
          float f1 = fbm3oLod(pwF, pwU, camAltKm, 1.0, ampSum);
          f1 /= max(ampSum, 0.15); // renormalize after octave fade-out
          // PERSISTENT SYSTEMS: threshold a blend of the big weather gate
          // and the fbm detail, not the fbm alone. Pure-fbm decks re-arrange
          // at every zoom (equal-size blobs have no identity as the planet
          // scales — "the clouds all changed" while approaching). Anchoring
          // the density on the ~4600 km weather systems makes the deck's
          // macro layout stable across altitude; fbm only textures the
          // systems' edges.
          // f1 multiplies in only where the weather system is established;
          // sys is FLOORED at (weatherTerm - 0.10): inside a cloudy system
          // the fbm detail can thin the deck but never punch through to a
          // hole. Before the floor, f1's 152-193 km octave minima dipped
          // below thr INSIDE cloudy systems — a quasi-periodic lattice of
          // round ground-showing dots (red class in cloudbg; 8px autocorr
          // peak at 9.45 Mm). Genuine system gaps keep coming from the
          // weather gate (green class), not from fbm dips.
          float wSys = 0.62 * smoothstep(0.30, 0.62, weatherM);
          float sys = max(wSys + 0.38 * f1 * smoothstep(0.30, 0.55, weatherM),
                          wSys - 0.10);
          // NO billow fold here: 1-|2f-1| paints a round fold-hole at every
          // lattice node, and thresholded from above those holes read as an
          // organized lace of circles ("noise when approaching"). The far
          // map thresholds the RAW fbm instead — organic system shapes.
          // sys distribution (gradient-noise fbm, 30k sphere samples):
          // mean 0.499, std 0.180. thr picks q70-q78 → ~33-40% coverage —
          // M11n7: discrete deck with ground gaps (Ace-Combat-style), was
          // q55-q72 (50-60% coverage — a continuous sheet from 20-25 km).
          float thr = mix(0.68, 0.60, cover);
          // DIP FLOOR: cloudbg proved the dot holes are sys dips below thr
          // (red class). Shallow dips (the lattice-minima speckle) close by
          // flooring the input 0.055 below thr — deep system gaps survive.
          float d = smoothstep(thr, thr + 0.06, max(sys, thr - 0.055));
          d = pow(d, 0.45); // saturate interior: translucent gray dots close
          // SINGLE soft gain: the double clamp (1.35 then 1.5) forced the
          // deck to binary alpha — during approach every edge pixel flipped
          // 0↔1 and the pattern read as re-dealt at every zoom. Partial
          // edge alpha keeps features identifiable while the view scales.
          float gate = max(smoothstep(0.34, 0.55, weatherM),
                           0.12 * smoothstep(0.15, 0.35, weatherM));
          // M11n6: the shell gets the same unconditional dry-trough floor
          // as the march's macroM — orbit and ground must agree.
          gate = max(gate, 0.16);
          d *= gate;
          // M11n6b: the shell's sparse fair-weather puffs (sys top peaks) —
          // matches the march's 3 km sparse puffs so dry cells read as
          // scattered cumulus from orbit AND from the ground.
          float sparseS = smoothstep(0.78, 0.92, sys);
          d = max(d, sparseS * 0.35);
          // M11n9 REDESIGN: the SAME region/body/tower structure as the
          // march's cloudDensity, evaluated at the slab-midpoint position —
          // the orbit view shows the same limited cloudy regions with the
          // same discrete bodies, and the 18-25 km handoff stays seamless.
          vec3 spF = upW * (uPlanetR + 3000.0);
          float regionN = noise3(spF * (1.0 / 60000.0) + wind * (1.0 / 60000.0));
          float region = smoothstep(0.50, 0.60, regionN);
          float bodyN = noise3(spF * (1.0 / 4000.0) + wind * (1.0 / 4000.0));
          // M11n9l: the body duty matches cloudDensity's raised threshold
          float body = smoothstep(0.55, 0.66, bodyN);
          float towerN = noise3(spF * (1.0 / 9000.0) + wind * (1.0 / 9000.0));
          float tower = smoothstep(0.60, 0.75, towerN);
          // M11n9o CROSS-FADE: above 12 km the full map; below it the shell
          // keeps painting only the HORIZON BAND (ray elevation < ~4 deg —
          // the distant regions' tower tops), with an opacity that ramps up
          // toward the handoff — near (march) and far (shell band) OVERLAP
          // on the horizon strip and the far band thins gradually while
          // descending, instead of vanishing at 11 km (the cloudless gap).
          // (elevR/bandGate/overlap/zoneBand hoisted to the block's early-
          // out gate — M11n9t2.)
          d *= max(smoothstep(11000.0, 13000.0, camAlt), zoneBand);
          // M11n9q HORIZON WALL: the strip between the deck's visual edge
          // (~244 km from 8.9 km) and the true horizon (~340 km) shows the
          // distant clouds' VERTICAL SIDES — from afar you see cloud walls,
          // not the top-down coverage, so the strip read as empty blue even
          // where regions existed (user report: altitude-dependent). Add a
          // relaxed-coverage wall density in the band (region/body
          // thresholds lowered): the strip reads as a fog-washed distant
          // cloud bank at every altitude. zoneBand keeps it out of the
          // mid-sky (the wisp killer) and above the handoff.
          float wallRegion = smoothstep(0.38, 0.50, regionN);
          float wallBody = smoothstep(0.46, 0.58, bodyN);
          float wall = wallRegion * mix(0.55, 1.0, wallBody);
          d = max(d, wall * zoneBand);
          float shellShade = 0.65 + 0.35 * clamp(dot(upF, uSunDir) * 0.5 + 0.5, 0.0, 1.0);
          // M11n9e: sunset tint on the far map (matches the march's tint)
          vec3 shellTint = mix(vec3(1.0, 0.48, 0.25), vec3(1.0),
                               smoothstep(0.05, 0.45, dot(upF, uSunDir)));
          // ---- DEBUG (?cloudbg=1): color-code what suppresses the deck --
          // RED   = sys below threshold (fbm/anchor dips: lattice holes)
          // GREEN = gate low (macro weather suppressing: system gaps)
          // YELLOW= d mid-range (partial deck)
          // WHITE = fully cloudy
          if (uCloudDbg > 0.5) {
            float sunHd = dot(up0, uSunDir);
            vec3 dbg = vec3(1.0); // white base (cloudy)
            if (gate < 0.5) dbg = vec3(0.1, 0.9, 0.1);          // gate hole
            else if (sys < thr) dbg = vec3(0.9, 0.1, 0.1);      // sys hole
            else if (d < 0.6) dbg = vec3(0.9, 0.9, 0.1);        // partial
            dbg *= (0.4 + 0.6 * shellShade);
            shellCol = dbg;
            shellA = 0.95 * wShell;
            shellA *= smoothstep(-0.25, 0.0, sunHd) * 0.98 + 0.02;
          } else {
          // Density-driven color: thick cores read warm-white, thin edges
          // cool gray-blue — matches how the volumetric deck shades, and
          // keeps the shell from reading as one flat cream sheet.
          shellCol = mix(vec3(0.72, 0.76, 0.82), vec3(1.02, 1.0, 0.97), smoothstep(0.05, 0.6, d)) * shellShade * shellTint;
          // Opacity calibrated to the volumetric march it replaces: marching
          // the full ~2.4 km slab accumulates ~1-exp(-d * 6). The old k=2600
          // saturated EVERY pixel to opaque (uniform cream sheet from orbit).
          // k=12: d=0.3 → alpha 0.97 — mid-density deck reads as CLOUD, not
          // translucent gray dots over dark terrain (the 'gray dot grid':
          // d 0.2-0.5 regions were 30-80% see-through, compositing the dark
          // ground into a warm-gray dot).
          shellA = 1.0 - exp(-d * 12.0);
          // M11n9p HULL-PROXIMITY FADE: the far hull is a 2D surface —
          // as the camera approaches its 12 km radius the polygon edges
          // and the flat sheet become visible (user screenshot at 10.2 km,
          // 1.8 km below the hull). Fade the shell out as the fragment's
          // distance to the camera shrinks: full opacity beyond 4 km,
          // fully transparent within 1.2 km of the surface. Rays that
          // pierce the surface near the camera (looking up from below the
          // hull) fade the same way, so the hull is invisible at the
          // moment of crossing.
          float fragDist = distance(uCamPos, vWorld);
          shellA *= smoothstep(1200.0, 4000.0, fragDist);
          // M11n9o: the shell's weight — full above the handoff; in-zone
          // the horizon band's own overlap weight (wShell is 0 there)
          shellA *= max(wShell, zoneBand);
          // night fade (same terms as the volumetric path)
          float sunHs = dot(up0, uSunDir);
          shellCol *= smoothstep(-0.12, 0.08, sunHs);
          shellA *= smoothstep(-0.25, 0.0, sunHs) * 0.98 + 0.02;
          // M11n9k AERIAL PERSPECTIVE: distant shell clouds wash into the
          // horizon haze. Edge-on from inside the zone the 2D map read as
          // paper-thin wisps floating in the sky (user report) — with the
          // same fog the terrain uses (optical depth over the view
          // distance, 2-sample sun od) the far band fades into the haze
          // instead. Cloud uniforms share the atmosphere's betas.
          {
            // M11n9k fix: the fog distance is the TRUE mapped-cloud distance
            // (the slab crossing, up to 300 km for horizon rays) — the hull
            // fragment's own position is only 2.6 km away (the near hull
            // follows the camera) and using it left the far wisps crisp
            float distS = cloudDist > 0.0 ? cloudDist : distance(uCamPos, vWorld);
            vec3 midS = ro + rd * (distS * 0.5);
            float hgtS = max(length(midS) - uPlanetR, 0.0);
            float dRS = exp(-hgtS / uHR) * distS;
            float dMS = exp(-hgtS / uHM) * distS;
            float sunHs3 = dot(normalize(midS), uSunDir);
            // inscatter toward the camera (2-sample sun od, same as terrain)
            vec3 p1S = midS + uSunDir * (uHR * 2.0);
            float h1S = max(length(p1S) - uPlanetR, 0.0);
            float sLenS = distance(midS, p1S);
            float sdRS = (exp(-hgtS / uHR) + exp(-h1S / uHR)) * 0.5 * sLenS;
            float sdMS = (exp(-hgtS / uHM) + exp(-h1S / uHM)) * 0.5 * sLenS;
            vec3 odSunS = vec3(sdRS * uBetaR.x, sdRS * uBetaR.y, sdRS * uBetaR.z) + sdMS * uBetaM;
            float muS = dot(rd, uSunDir);
            float phRS = 3.0 / (16.0 * 3.14159) * (1.0 + muS * muS);
            float gS = 0.76;
            float phMS = 3.0 / (8.0 * 3.14159) * ((1.0 - gS*gS)*(1.0+muS*muS)) / ((2.0+gS*gS)*pow(1.0+gS*gS-2.0*gS*muS, 1.5));
            vec3 inscS = (vec3(dRS * uBetaR.x, dRS * uBetaR.y, dRS * uBetaR.z) * phRS + dMS * uBetaM * phMS)
                       * exp(-odSunS) * smoothstep(-0.15, 0.1, sunHs3);
            float fogS = clamp(1.0 - exp(-min(dRS * uBetaR.x + dMS * uBetaM.x, 12.0)), 0.0, 1.0);
            shellCol = mix(shellCol, inscS * 1.15, min(fogS, 0.85));
          }
          // M11n9s DISTANT LIGHTNING: the shell's tower band flashes too —
          // far storms on the horizon pulse blue-white (unmistakable at
          // night, a faint white pulse by day). Added AFTER the fog mix
          // because the night fog washes shellCol toward 0 (inscS is dark
          // at night) and would erase the flash. Same cell-hash timing
          // scheme as the march's flashes. The alpha lifts for the flash
          // duration: the night fade leaves shellA at its 0.02 floor, and
          // a flash with no alpha is invisible.
          vec3 tcS = floor(spF / 9000.0 + wind / 9000.0);
          float periodS = mix(2.5, 9.0, fract(sin(dot(tcS, vec3(127.1, 311.7, 74.7))) * 43758.5453));
          float phaseS = fract(sin(dot(tcS, vec3(269.5, 183.3, 246.1))) * 43758.5453);
          float ftS = fract(uTime / periodS + phaseS);
          float flashS = clamp(exp(-pow((ftS - 0.02) / 0.012, 2.0))
                     + 0.6 * exp(-pow((ftS - 0.085) / 0.010, 2.0)), 0.0, 1.0);
          if (flashS > 0.001 && towerN > 0.72) {
            shellCol += vec3(0.85, 0.92, 1.0) * flashS * tower * 2.5;
            shellA = max(shellA, flashS * tower * 0.85 * max(wShell, zoneBand));
          }
          } // M11n9t2 early-out if
          }
        }

        // ============ volumetric march (near view) ====================
        vec3 volCol = vec3(0.0);
        float volT = 1.0; // transmittance
        float dbgT0 = 0.0, dbgSpan = 0.0, dbgSteps = 0.0, dbgMaxD = 0.0;
        float dbgMacro = 0.0, dbgD = 0.0, dbgWeather = weather, dbgSys = 0.0;
        if (wVol > 0.001) {
          // M11n9: the cloud zone now extends to the TOWER tops (~12 km) —
          // cumulonimbus towers rise from the 1.8 km base through the old
          // 4.2 km deck top
          float rB = uPlanetR + 300.0; // M11n9v: extended down from the 1800 m base for rain shafts
          float rT = uPlanetR + 12000.0;
          vec2 tB = raySphere(ro, rd, rB);
          vec2 tT = raySphere(ro, rd, rT);
          float t0, t1;
          if (camAlt < 300.0) {
            // below the base: enter at bottom-sphere far hit, exit at top far hit.
            // HORIZON REJECTION: the near hull has depthTest off (mountains
            // must not erase the deck), so downward rays whose sight line
            // strikes the ground BEFORE the slab would paint cloud onto the
            // terrain (the "cloud texture on the ground" artifact). Such
            // rays dip inside the planet: reject them by closest approach
            // (perigee = |ro × rd|) < surface radius + margin.
            float perigee = length(cross(ro, rd));
            // ...and only if that closest approach lies AHEAD of the camera
            // (t = -dot(ro,rd) > 0). For up-looking rays the line's perigee
            // is behind the camera — the ray climbs away and never dips.
            // M11n9b: the margin was +300 m — at 8-12 km altitude a
            // NEAR-HORIZONTAL ray has perigee ≈ camAlt (the ray still exits
            // the cloud zone far away, above the ground), so the tiny
            // margin rejected every horizon ray and the horizon atmosphere
            // band showed no clouds (user report). The ray only paints
            // cloud ONTO TERRAIN if the perigee actually dips below the
            // SURFACE: margin = camAlt is exactly the surface radius.
            if (-dot(ro, rd) > 0.0 && perigee < uPlanetR + 150.0) {
              t0 = 1.0; t1 = 0.0; // no march
            } else {
              t0 = max(tB.y, 0.0);
              t1 = tT.y;
            }
          } else if (camAlt > 12000.0) {
            // above the tower tops: enter at top near hit, exit at bottom near hit
            t0 = max(tT.x, 0.0);
            t1 = tB.x;
          } else {
            // inside the cloud zone (1.8-12 km)
            t0 = 0.0;
            // M11n9b: INSIDE the zone a downward ray must march to where it
            // EXITS the zone bottom — but tB.x (the bottom sphere's near
            // hit) is BEHIND the camera when the camera is inside the
            // bottom sphere (alt < 1.8 km never happens here, but at 8-11
            // km a downward ray's bottom crossing is the FAR hit tB.y, not
            // tB.x). Use the far hit for downward rays, the near hit for
            // the (rare) upward case; up rays exit at the top far hit.
            t1 = dot(rd, up0) > 0.0 ? tT.y : max(tB.y, 0.0);
          }
          if (tB.x < 0.0 && tB.y < 0.0 && camAlt < 300.0) {
            t1 = -1.0; // grazing ray that never re-enters: no march
          }
          if (t1 > t0) {
            // M11n9c: the march cap is DIRECTION-DEPENDENT now. A
            // near-horizontal ray from inside the cloud zone stays in the
            // zone for 100+ km; the old flat 18 km cap meant the march
            // sampled only the first 18 km — when that stretch sat in a
            // clear region/body the horizon band rendered as a cloudless
            // strip (user report). Horizontal rays march 40 km (crossing
            // several 60 km-scale regions' edges), steep rays keep 18 km.
            const int MAX_STEPS = 32;
            t1 = min(t1, t0 + mix(32000.0, 18000.0,
                                  clamp(abs(dot(rd, up0)) * 10.0, 0.0, 1.0)));
            // step count scales with the marched span so the sample
            // density stays ~1 km/step in every direction
            // M11n9t PERF: 32 -> 24 cap. Inside the deck the whole screen
            // marches and the frame rate fell to ~6 fps; 24 steps at 1.3
            // km spacing is visually indistinguishable in a broad soft
            // deck (the IGN dither hides the coarser quantization).
            int steps = int(clamp(uVolSteps * (t1 - t0) / 18000.0, 12.0, 24.0));
            dbgT0 = t0; dbgSpan = (t1 - t0) / 18000.0; dbgSteps = float(steps) / 48.0;
            // cap the marched path: grazing rays through the slab would
            // accumulate alpha=1 over hundreds of km and read as a gray
            // wall. (M11n9c: the flat 18 km cap was replaced by the
            // direction-dependent cap above — horizontal 40 km / steep 18 km.)
            // M11n3 UNIFIED MACRO: evaluate the SHELL's exact system field
            // at the column midpoint direction (per-fragment is enough: the
            // 152 km system scale varies ~12% of a cell over the 18 km
            // path). The old proxy (single 152 km gnoise3 octave, phase
            // 91.7) was a different field than the far shell's fbm3oLod
            // sys — from orbit a system read solid while the same column
            // from inside marched to aV=0 ("no clouds above the horizon
            // from inside / below the deck"). Same formula, same
            // thresholds, same octave fade as the shell.
            vec3 upMid = normalize(ro + rd * (t0 + (t1 - t0) * 0.5));
            const float cosB2 = 0.8660254, sinB2 = 0.5;
            vec3 spM = upMid * (uPlanetR + 3000.0);
            vec3 spMX = vec3(spM.x * cosB2 - spM.z * sinB2, spM.y,
                             spM.x * sinB2 + spM.z * cosB2);
            vec3 pwFM = vec3(spMX.x * 0.62, spMX.y * 1.38, spMX.z * 0.62) *
                        (1.0 / 70000.0) + wind * (1.0 / 70000.0);
            vec3 pwUM = spM * (1.0 / 70000.0) + wind * (1.0 / 70000.0);
            float ampSumM;
            // 512 km gate: match the SHELL's octave fade as seen from orbit
            // (the deck the user compares against). At the camera altitude
            // (1-25 km) the 15-70 km octaves would enable, and their noise
            // swings sysM across the threshold inside a single system cell
            // — the orbit view (512 km, 15 km octave gated out) shows a
            // solid deck where the march saw holes. System gate must be
            // octave-stable and agree with the orbit view's field.
            float f1m = fbm3oLod(pwFM, pwUM, 512.0, 1.0, ampSumM);
            f1m /= max(ampSumM, 0.15);
            float weatherMm = fbm2(upMid * 2.2 + vec3(wTime, wTime * 0.7, -wTime * 0.6));
            dbgWeather = weatherMm;
            float wSysM = 0.62 * smoothstep(0.30, 0.62, weatherMm);
            float sysM = max(wSysM + 0.38 * f1m * smoothstep(0.30, 0.55, weatherMm),
                             wSysM - 0.10);
            dbgSys = sysM;
            float macroThrM = mix(0.60, 0.50, cover);
            float macroM = pow(smoothstep(macroThrM, macroThrM + 0.16,
                                          max(sysM, macroThrM - 0.055)), 0.45)
                         * max(smoothstep(0.34, 0.55, weatherMm),
                               0.12 * smoothstep(0.15, 0.35, weatherMm));
            // M11n9: macroM is now a SOFT density modulator (0.3+0.7·m at
            // the d multiply below) — the M11n6 hard gate + floor chain is
            // gone; the new region/body/tower structure owns coverage.
            dbgMacro = macroM;
            float dt = (t1 - t0) / float(steps);
            // Static dither (Interleaved Gradient Noise, Jimenez'): breaks
            // the concentric step-quantization bands of a uniform march.
            // Frame-varying jitter was tried and reads as boiling noise;
            // IGN is screen-stable and decorrelates neighbors cleanly.
            float jit = fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x
                        + 0.00583715 * gl_FragCoord.y));
            float t = t0 + dt * jit;
            float phase = 0.35 + 0.65 * pow(clamp(dot(rd, uSunDir) * 0.5 + 0.5, 0.0, 1.0), 2.0);
            for (int i = 0; i < MAX_STEPS; i++) {
              if (i >= steps) break;
              vec3 p = ro + rd * t;
              float r = length(p);
              // M11n9: hE (extended slab units, 0=base … 4.3=tower top) is
              // computed inside cloudDensity; the old h (deck-only units)
              // is no longer needed here
              // Detail LOD: fade the 3rd fbm octave with distance INSTEAD of
              // rescaling the lattice (rescaling planet-frame coords per
              // pixel jumps the noise grid by whole cells -> the concentric
              // "ripple rings" seen when approaching the deck). h grows with
              // sample distance, so the fade is smooth along the ray too.
              float detail = 1.0 / (1.0 + t * (1.0 / 8000.0));
              // ---- unified density: same function the far shell shows ----
              // M11n7: sharper edge (0.06) — discrete puff bodies instead
              // of a soft continuous sheet; the narrow partial-density band
              // also keeps oblique slant paths from veiling over
              float d = cloudDensity(p, cover, weather, wind, 0.06, detail, weatherMm);
              // M11n3 macro coupling: the gate is macroM, evaluated ONCE per
              // fragment at the column midpoint (above). The older schemes
              // both failed: per-fragment constants measured a different
              // weather cell than the column (capped the whole deck at
              // aV≈0.2), and the per-sample proxy field disagreed with the
              // shell's field so the deck vanished from inside while the
              // orbit view showed it solid.
              // M11n3: the macro gate is the shell's own field (macroM
              // above), uniform along the column — the 152 km system scale
              // cannot change across an 18 km path, and using the shell's
              // field/thresholds is what makes the inside view agree with
              // the orbit view.
              // M11n9: macroM is a SOFT modulator (0.3 + 0.7·m) — coverage
              // structure comes from cloudDensity's region/body/tower; the
              // macro field only thickens/thins the density.
              d *= (0.3 + 0.7 * macroM);
              if (i == 0) dbgD = d;
              // M11n9b diagnostic: sample the gate layers at the mid point
              if (i == steps / 2) {
                float dR, dB, dV;
                float gT = cloudGateDebug(p, cover, weather, wind, weatherMm, dR, dB, dV);
                dbgMaxD = dR; dbgSpan = dB; dbgSteps = dV;
              }
              // edge erosion: high-frequency wisps carve the surface (fades
              // out with distance so far samples stay smooth); hf cells are
              // 640 m — same physical wind divided by that cell size
              // M11n9t PERF: skip entirely for far samples (detail < 0.55 =
              // t > 6.6 km) — the erosion amplitude is already faded to
              // < 55% there and the fbm3o is 3 noise evals per step
              if (detail > 0.55) {
                float hf = fbm3o(p * (1.0 / 640.0) + wind * (1.0 / 640.0) * vec3(-1.7, 1.0, 0.8), detail * detail);
                d -= (1.0 - d) * hf * 0.35;
              }
              // M11n9: the vertical shaping now lives in cloudDensity's vert
              // (base + tower profile) — the old deck-only shaping removed
              d = clamp(d * 1.5, 0.0, 1.0);
              if (d > 0.015) {
                // light march: 3 samples toward the sun (cheap 1-octave billow)
                // M11n9t PERF: tail steps (volT < 0.12) skip the light
                // march — their contribution is aStep·volT ≲ 0.06·lit, so
                // a constant mid-shadow replaces 9 noise evals invisibly.
                float od = 0.0;
                float shadow;
                if (volT > 0.12) {
                  // M11n9t PERF: 1 light sample instead of 3 — the old
                  // samples sat 220 m apart inside a 3 km noise lattice,
                  // so they read nearly the same field value: 3x the cost
                  // for ~1 sample of information. One 400 m sample with
                  // the od scale matched (700 = the old 3x220 range) keeps
                  // the same shadow depth.
                  vec3 pl = p + uSunDir * 400.0;
                  float fl = noise3(pl * (1.0 / 3000.0) + wind * (1.0 / 3000.0));
                  float bl = 1.0 - abs(2.0 * fl - 1.0);
                  // same threshold family as cloudDensity (M11n8b: mix(0.41,0.35)):
                  float thrS = mix(0.41, 0.35, cover);
                  od += smoothstep(thrS, thrS + 0.18, bl * (0.72 + 0.28 * cover)) * 700.0;
                  shadow = exp(-od * 0.0008);     // Beer-Lambert, gentler:
                  // 0.0012/0.004 history — interior samples went near-black
                  // from orbit and the near-view deck read PALE GRAY next to
                  // the far shell's bright map (the handoff mismatch).
                  // tops catch the sun: height-based ambient brightening.
                  // Sun term floored higher + whiter ambient: the near-view
                  // deck must match the far shell's white, or the LOD
                  // handoff reads as the clouds fading (user report).
                  // 0.45 floor on shadow + stronger ambient: bases seen from
                  // below were rendering luma ~100 (near-black underbellies).
                  shadow = 0.45 + 0.55 * shadow;
                } else {
                  shadow = 0.70;
                }
                float powder = 1.0 - exp(-d * 4.0);   // dark edges, bright cores
                // M11n9e: sunset tint — near the terminator the direct sun
                // term turns warm orange and the sky ambient turns dusk
                // red-purple (the old fixed white/blue lit the towers the
                // same at noon and sunset)
                float sunHs2 = clamp(dot(normalize(p), uSunDir), -1.0, 1.0);
                vec3 sunCol = mix(vec3(1.0, 0.45, 0.20), vec3(1.0, 0.98, 0.95),
                                  smoothstep(0.05, 0.45, sunHs2));
                vec3 ambCol = mix(vec3(0.50, 0.38, 0.48), vec3(0.62, 0.68, 0.80),
                                  smoothstep(-0.02, 0.30, sunHs2));
                vec3 lit = sunCol * shadow * phase * (0.55 + 0.45 * powder)
                         + ambCol * (0.55 + 0.35 * clamp(cloudHeightE(p), 0.0, 1.2)); // sky ambient
                // M11n9g: city glow on cloud bases — over a night-side city
                // (the SAME 3-octave population field the terrain's city
                // lights use) the base samples catch a faint orange
                // up-glow. EMISSIVE (own small alpha, no volT update):
                // thin bases have aStep≈0 so the lit path can't carry it.
                float hEg = cloudHeightE(p);
                if (hEg < 0.4) {
                  float nightF = smoothstep(0.06, -0.04, dot(normalize(p), uSunDir));
                  if (nightF > 0.001) {
                    vec3 gp = normalize(p) * (uPlanetR + 20.0);
                    float cl1 = fbm2(gp * (1.0 / 900000.0));
                    float cl2 = fbm2(gp * (1.0 / 120000.0) + vec3(7.31, 2.9, 5.13));
                    float latG = asin(clamp(normalize(p).y, -1.0, 1.0)) * 57.29578;
                    float popBand = 0.35 + 0.65 * exp(-pow((latG - 30.0) / 38.0, 2.0));
                    float pop = smoothstep(0.46, 0.68, cl1) * smoothstep(0.40, 0.62, cl2) * popBand;
                    if (pop > 0.001) {
                      float spG = fbm2(gp * (1.0 / 800.0) + vec3(3.7, 9.1, 1.3));
                      float glow = pop * smoothstep(0.46, 0.72, spG);
                      volCol += vec3(1.0, 0.70, 0.40) * glow * nightF * 0.10 * volT;
                    }
                  }
                }
                float aStep = 1.0 - exp(-d * dt * 0.005); // extinction tuned
                // to the far shell's opacity (1-exp(-d*12)): with k=0.0022
                // a full column only reached alpha≈0.4 — the deck stayed
                // see-through from below (dark sky bled through, cloud
                // pixels read luma ~100) and pale at the LOD handoff.
                volCol += lit * aStep * volT;
                // M11n9s LIGHTNING: storm towers flash from inside. The
                // tower lattice cell (9 km, wind-attached) hashes to a
                // random period/phase; the flash is a short double-flicker
                // (two pulses ~90 ms apart, ~0.25 s total — real strokes
                // restrike). Emissive: bypasses the sun's shadow term (a
                // storm interior is self-lit) and tints blue-white, gated
                // to the tower body (hE 0.3-4.5) — the flash only computes
                // its tower noise during a flash window (rare), so the
                // steady-state march cost is one hash.
                vec3 tc = floor(p / 9000.0 + wind / 9000.0);
                float periodL = mix(2.5, 9.0, fract(sin(dot(tc, vec3(127.1, 311.7, 74.7))) * 43758.5453));
                float phaseL = fract(sin(dot(tc, vec3(269.5, 183.3, 246.1))) * 43758.5453);
                float ftL = fract(uTime / periodL + phaseL);
                float f1L = exp(-pow((ftL - 0.02) / 0.012, 2.0));
                float f2L = 0.6 * exp(-pow((ftL - 0.085) / 0.010, 2.0));
                float flashI = clamp(f1L + f2L, 0.0, 1.0);
                if (flashI > 0.001) {
                  float tN = noise3(p * (1.0 / 9000.0) + wind * (1.0 / 9000.0));
                  if (tN > 0.70) {
                    float hEl = cloudHeightE(p);
                    float vProf = smoothstep(0.3, 1.0, hEl) * (1.0 - smoothstep(3.4, 4.5, hEl));
                    volCol += vec3(0.85, 0.92, 1.0) * flashI * vProf * aStep * volT * 3.0;
                  }
                }
                volT *= 1.0 - aStep;
                if (volT < 0.03) break;
              }
              t += dt;
            }
          }
        }

        // ---------------- composite (hull-specific) ---------------------
        // M11n9o: the near hull is MARCH ONLY (its color stays full-bright
        // and the handoff fade is carried by alpha alone — the old
        // wVol-scaled color double-faded dark); the far hull is SHELL ONLY.
        // The handoff overlap: the far hull's horizon band (see the in-zone
        // gate) fills the strip the march's cap can't reach and thins
        // gradually as the march fades in — no cloudless gap.
        float aV = 1.0 - volT; // march coverage (probe/debug channel)
        float cov = uNearHull > 0.5
          ? clamp(aV * wVol, 0.0, 1.0)
          : clamp(shellA, 0.0, 1.0);
        vec3 col = uNearHull > 0.5 ? volCol : shellCol;
        float alpha = cov;
        // altitude-based opacity fade inside the band (flying through).
        // Floor 0.55 (was 0.35): inside/near the slab the deck used to dim
        // so much that entering the clouds read as them DISAPPEARING.
        // (M11n3b cleanup: the old inBand term was dead code —
        // farFade*inBand + farFade*(1-inBand) == farFade — and hLayer fed
        // only that; removed. bandFade is purely the camera-altitude ramp.)
        float farFade = clamp(abs(camAlt - ${((CLOUD_BOTTOM + CLOUD_TOP) / 2).toFixed(1)}) / 6000.0, 0.0, 1.0);
        float bandFade = mix(0.55, 0.95, farFade);
        alpha *= mix(bandFade, 1.0, step(8000.0, camAlt));
        alpha = clamp(alpha, 0.0, 1.0);
        // ?cloudbg=2: march probe — R=aV*4, G=span/18km, B=steps/28,
        // plus dbgMaxD folded into B's fraction. Diagnoses below-deck.
        if (uCloudDbg > 3.5) {
          // probe v9: R=cover, G=macroThrM, B=macroThr* (bands/lat diag)
          float latD = asin(clamp(upF.y, -1.0, 1.0));
          float bandsD = 0.55 + 0.45 * cos(lat * 6.0) * 0.5 + 0.25 * exp(-pow((abs(lat) - 0.15) * 3.0, 2.0));
          gl_FragColor = vec4(clamp(cover, 0.0, 1.0), clamp(macroThr, 0.0, 1.0), clamp(bandsD * 0.5, 0.0, 1.0), 1.0);
          #include <colorspace_fragment>
          return;
        }
        if (uCloudDbg > 2.5) {
          // TEMP M11n6 diagnostic: constant red = near hull coverage map
          gl_FragColor = vec4(0.9, 0.08, 0.05, 1.0);
          #include <colorspace_fragment>
          return;
        }
        if (uCloudDbg > 1.5) {
          // probe v6+diag: R=aV*4, G=region(dR), B=vert(dV) — B<1 = body/vert gates
          gl_FragColor = vec4(clamp(aV * 4.0, 0.0, 1.0), clamp(0.5 + 0.5 * dbgMaxD, 0.0, 1.0), clamp(dbgSteps, 0.0, 1.0), 1.0);
          #include <colorspace_fragment>
          return;
        }
        // night fade
        float sunH = dot(up0, uSunDir);
        col *= smoothstep(-0.12, 0.08, sunH);
        alpha *= smoothstep(-0.25, 0.0, sunH) * 0.98 + 0.02;

        gl_FragColor = vec4(col, clamp(alpha, 0.0, 1.0));
        #include <colorspace_fragment>
      }
    `;
  const nearGeo = new THREE.SphereGeometry(planetR + 2600, 128, 96);
  // M11n9: the far shell sits at 12 km — the cumulonimbus TOWER TOP — so
  // the far map's parallax matches the towers the march draws
  const farGeo = new THREE.SphereGeometry(planetR + 12000, 128, 96);
  // M11m3 OWNERSHIP FIX: the injections were INVERTED — the near hull
  // drew the shell map and the far hull the march, so from orbit the SAME
  // deck painted twice (far hull at R+9.8 km + a copy on the R+2.6 km
  // hull): a second layer of small clouds below the far deck, moving
  // with parallax ("snow-like clouds under the far clouds"). Gate each
  // path by hull identity instead of early-returning (an early return
  // before wShell would also kill the march that shares the shader):
  // near hull (uNearHull=1) → march ONLY; far hull (0) → shell ONLY.
  const nearMat = makeMat(1);
  const farMat = makeMat(0);
  // M11n9c: the shell gate is IN the source now (uNearHull-conditional);
  // only the march gate needs the hull injection
  const gateVol = 'if (wVol > 0.001 && uNearHull > 0.5) {';
  nearMat.fragmentShader = FRAGMENT
    .replace('if (wVol > 0.001) {', gateVol);
  farMat.fragmentShader = FRAGMENT
    .replace('if (wVol > 0.001) {', gateVol);
  const nearMesh = new THREE.Mesh(nearGeo, nearMat);
  const farMesh = new THREE.Mesh(farGeo, farMat);
  // NEAR hull: depth-test OFF. Inside/above the slab the hull fragments
  // sit BEHIND terrain along most downward rays (mountains reach 9.6 km,
  // the hull is at 2.6 km), so depth culling erased ~98% of the near
  // deck (measured: 1.6% of the frame rasterized over land). With the
  // depth test off the near deck always renders; the only artifact is
  // thin deck painted over a peak that is in front of it — minor next
  // to "clouds vanish when descending". The far hull stays depth-tested
  // (it sits above all terrain, so culling is correct there).
  nearMat.depthTest = false;
  nearMesh.frustumCulled = false;
  farMesh.frustumCulled = false;
  nearMesh.renderOrder = 5; // M11n3: after the far hull (4) so the far shell can never overwrite the march; ties the atmosphere (5), sorted nearer-last
  farMesh.renderOrder = 4;
  const group = new THREE.Group();
  group.add(nearMesh);
  group.add(farMesh);
  return group;
}
