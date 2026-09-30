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
}

export function makeCloudUniforms(planetR: number): CloudUniforms {
  return {
    uSunDir: { value: new THREE.Vector3(1, 0.3, 0.35).normalize() },
    uPlanetR: { value: planetR },
    uCamPos: { value: new THREE.Vector3() },
    uOrigin: { value: new THREE.Vector3() },
    uTime: { value: 0 },
    uCover: { value: 0.42 },
    uVolSteps: { value: 18 },
    uTanHalfFov: { value: Math.tan((60 * Math.PI) / 360) },
    uViewportH: { value: 900 },
    uCloudDbg: { value: 0 },
  };
}

export function makeCloudMesh(planetR: number, uniforms: CloudUniforms): THREE.Mesh {
  // MESH RADIUS vs SAMPLE ALTITUDE — decoupled. The shell sits ABOVE the
  // highest terrain (peaks reach ~9.6 km; coarse LOD tiles displace whole
  // mountain-octave regions 3-8 km up). At the slab-middle radius (3 km)
  // those tiles poked THROUGH the shell: depth-culled shell fragments left
  // round tile-sized holes where the ground showed as gray dots in a
  // quasi-regular grid (the mountain lattice's 320 km wavelength). The far
  // map is a flat texture of the deck — raising the MESH to 9.8 km costs
  // nothing visually, while density sampling below still evaluates the
  // slab-middle altitude so the map matches the volumetric near view.
  const geo = new THREE.SphereGeometry(planetR + Math.max(CLOUD_TOP, 9800), 128, 96);
  const mat = new THREE.ShaderMaterial({
    uniforms: uniforms as unknown as { [k: string]: THREE.IUniform },
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
    fragmentShader: /* glsl */ `
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
        float a = 0.5, s = 0.0;
        s += a * noise3(p); p *= 2.17; a *= 0.5;
        s += a * noise3(p); p *= 2.17; a *= 0.5 * detail;
        s += a * noise3(p);
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
      float fbm3oLod(vec3 p, vec3 pU, float kmPerPx, float detail,
                     out float ampSum) {
        // CONVERGING-OCTAVE LOD: every octave keeps FULL amplitude, but as
        // its cell shrinks toward ~10 px on screen the octave's value
        // converges to the NEXT-COARSER octave's value instead of fading
        // out. Amplitude fades create a LONE-OCTAVE regime — one surviving
        // Perlin octave paints one round blob/hole per cell (the halftone
        // grids). Converging values keep two scales alive at every
        // altitude. No renorm needed (ampSum stays 1.0), so threshold
        // statistics never shift.
        // Octave cells (km): 330 anchor, 152, 70, 32.2, 14.8 (detail-gated).
        #define OCT_BLEND(cellKm) clamp((cellKm * 0.5 / kmPerPx - 5.0) / 8.0, 0.0, 1.0)
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
        // octave 3 (152 km): converges to octave 4 below ~10 px
        float b3 = OCT_BLEND(152.0);
        float v3 = mix(v4, 0.5 + gnoise3(pU * 0.461 + 7.7) * 1.2, b3);
        // octave 2 (70 km): converges to octave 3 below 10 px
        float b2 = OCT_BLEND(70.0);
        float v2 = mix(v3, 0.5 + gnoise3(p + w) * 1.2, b2);
        // octave 1 (32 km): converges to octave 2
        float b1 = min(OCT_BLEND(32.2), b2);
        vec3 p2 = vec3(p.y, p.z, p.x);
        float v1 = mix(v2, 0.5 + gnoise3(p2 * 2.17 + w) * 1.2, b1);
        // octave 0 (15 km): converges to octave 1, detail-gated
        float b0 = min(OCT_BLEND(14.8), b1) * detail;
        vec3 p3r = vec3(p.z, p.x, p.y);
        float v0 = mix(v1, 0.5 + gnoise3(p3r * 4.71 + w) * 1.2, b0);
        ampSum = 1.0;
        return 0.20 * v4 + 0.26 * v3 + 0.30 * v2 + 0.14 * v1 + 0.10 * v0;
        #undef OCT_BLEND
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
      float cloudDensity(vec3 p, float cover, float weather, vec3 windOff, float edge, float detail) {
        vec3 pw = p * (1.0 / 3000.0) + windOff * (1.0 / 3000.0); // puff cells ~3 km
        float f1 = fbm3o(pw, detail);
        float billow = 1.0 - abs(2.0 * f1 - 1.0); // rounded blobs [0,1]
        // Billow field statistics (200k samples): mean 0.19, median 0.23,
        // q70 0.52, q85 0.76. The old thr range (0.30 -> -0.10 with cover)
        // sat BELOW the mean: with cover ~0.4+ the smoothstep fired over
        // most of the field, every grazing ray saturated alpha within its
        // 40 km budget, and the deck read as a flat gray "water" sheet.
        // thr is anchored WELL above the median so the deck stays SPARSE:
        // low cover picks the top ~10% of puffs, high cover (~0.9) still
        // only claims the top ~40%.
        float thr = mix(0.70, 0.36, cover);
        float d = smoothstep(thr, thr + edge, billow * (0.55 + 0.45 * cover));
        d = clamp(d * 1.35, 0.0, 1.0);
        // Cluster gate: the old smoothstep(0.42,0.62) was so tight that only
        // isolated weather-field speckles passed — clouds read as mottled
        // fuzz instead of coherent systems. The wide gate grows proper
        // clusters; below it a thin sparse haze (0.12x) keeps clear skies
        // limited to the genuinely dry weather troughs.
        float gate = max(smoothstep(0.40, 0.58, weather),
                         0.12 * smoothstep(0.15, 0.35, weather));
        d *= gate;
        return d;
      }

      void main() {
        vec3 pc = -uOrigin;
        vec3 ro = uCamPos - pc;      // ray origin in planet frame
        vec3 rd = normalize(vWorld - uCamPos);
        float camAlt = length(ro) - uPlanetR;
        vec3 up0 = normalize(ro);
        // fragment's own altitude within the layer [0..1]
        float hLayer = clamp((length(vWorld - pc) - uPlanetR - ${CLOUD_BOTTOM.toFixed(1)}) /
                             ${((CLOUD_TOP - CLOUD_BOTTOM)).toFixed(1)}, 0.0, 1.0);

        // M10.7: the far-view TEXTURE SHELL owns the far field, so nothing is
        // discarded here any more. wVol still cross-fades the volumetric
        // march out (30-90 km) while the shell fades in (45-110 km); both
        // evaluate the same density field (see shell branch below).
        float wVol = (1.0 - smoothstep(30000.0, 90000.0, camAlt)) * step(0.001, uVolSteps);

        // ---------------- weather / coverage ----------------
        vec3 upF = normalize(vWorld - pc);
        vec3 q = upF * 2.2; // weather scale ~ R/2.2
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

        // ============ M10.7 hybrid: shell + volumetric ==================
        // Complementary handoff: wShell = 1 - wVol (above the near gate), so
        // at EVERY altitude one of the two LODs is at full weight and total
        // cloud coverage never dips (the gap the user saw at 48-60 km).
        // Both evaluate the SAME density field at the slab middle, so the
        // handoff just swaps WHO draws the same clouds.
        float wShell = (1.0 - wVol) * smoothstep(8000.0, 25000.0, camAlt);
        vec3 shellCol = vec3(0.0);
        float shellA = 0.0;
        if (wShell > 0.0001) {
          vec3 shellP = upF * (uPlanetR + ${((CLOUD_BOTTOM + CLOUD_TOP) / 2).toFixed(1)});
          // Detail LOD for the shell: the billow cells (3/1.4/0.64 km) and
          // the hf erosion (640 m) are SUB-PIXEL from orbit (km/px grows
          // ~linearly with camAlt) — sampling them anyway produces regular
          // moiré dots and, near the poles where meridians converge, the
          // organized dot rings + seam cuts seen on orbit. Fade the 3rd
          // octave (detail) and kill hf erosion as cells approach ~2px.
          float kmPerPx = 2.0 * camAlt * uTanHalfFov / uViewportH / 1000.0;
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
          float f1 = fbm3oLod(pwF, pwU, kmPerPx, 1.0, ampSum);
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
          // mean 0.499, std 0.180. thr picks q55-q72 → 28-45% pre-gate.
          float thr = mix(0.60, 0.50, cover);
          // DIP FLOOR: cloudbg proved the dot holes are sys dips below thr
          // (red class). Shallow dips (the lattice-minima speckle) close by
          // flooring the input 0.055 below thr — deep system gaps survive.
          float d = smoothstep(thr, thr + 0.16, max(sys, thr - 0.055));
          d = pow(d, 0.45); // saturate interior: translucent gray dots close
          // SINGLE soft gain: the double clamp (1.35 then 1.5) forced the
          // deck to binary alpha — during approach every edge pixel flipped
          // 0↔1 and the pattern read as re-dealt at every zoom. Partial
          // edge alpha keeps features identifiable while the view scales.
          float gate = max(smoothstep(0.34, 0.55, weatherM),
                           0.12 * smoothstep(0.15, 0.35, weatherM));
          d *= gate;
          float shellShade = 0.65 + 0.35 * clamp(dot(upF, uSunDir) * 0.5 + 0.5, 0.0, 1.0);
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
          shellCol = mix(vec3(0.72, 0.76, 0.82), vec3(1.02, 1.0, 0.97), smoothstep(0.05, 0.6, d)) * shellShade;
          // Opacity calibrated to the volumetric march it replaces: marching
          // the full ~2.4 km slab accumulates ~1-exp(-d * 6). The old k=2600
          // saturated EVERY pixel to opaque (uniform cream sheet from orbit).
          // k=12: d=0.3 → alpha 0.97 — mid-density deck reads as CLOUD, not
          // translucent gray dots over dark terrain (the 'gray dot grid':
          // d 0.2-0.5 regions were 30-80% see-through, compositing the dark
          // ground into a warm-gray dot).
          shellA = 1.0 - exp(-d * 12.0);
          shellA *= wShell;
          // night fade (same terms as the volumetric path)
          float sunHs = dot(up0, uSunDir);
          shellCol *= smoothstep(-0.12, 0.08, sunHs);
          shellA *= smoothstep(-0.25, 0.0, sunHs) * 0.98 + 0.02;
          }
        }

        // ============ volumetric march (near view) ====================
        vec3 volCol = vec3(0.0);
        float volT = 1.0; // transmittance
        if (wVol > 0.001) {
          float rB = uPlanetR + ${CLOUD_BOTTOM.toFixed(1)};
          float rT = uPlanetR + ${CLOUD_TOP.toFixed(1)};
          vec2 tB = raySphere(ro, rd, rB);
          vec2 tT = raySphere(ro, rd, rT);
          float t0, t1;
          if (camAlt < ${CLOUD_BOTTOM.toFixed(1)}) {
            // below the slab: enter at bottom-sphere far hit, exit at top far hit
            t0 = max(tB.y, 0.0);
            t1 = tT.y;
          } else if (camAlt > ${CLOUD_TOP.toFixed(1)}) {
            // above: enter at top near hit, exit at bottom near hit
            t0 = max(tT.x, 0.0);
            t1 = tB.x;
          } else {
            // inside the slab
            t0 = 0.0;
            t1 = tT.y > 0.0 ? tT.y : tB.y;
          }
          if (tB.x < 0.0 && tB.y < 0.0 && camAlt < ${CLOUD_BOTTOM.toFixed(1)}) {
            t1 = -1.0; // grazing ray that never re-enters: no march
          }
          if (t1 > t0) {
            const int MAX_STEPS = 28;
            int steps = int(clamp(uVolSteps, 4.0, 28.0));
            // cap the marched path: grazing rays through the slab would
            // accumulate alpha=1 over hundreds of km and read as a gray
            // wall. 18 km keeps distant air hazy instead of solid, and the
            // shorter path doubles the effective step density (fewer
            // aliasing artifacts on grazing views).
            t1 = min(t1, t0 + 18000.0);
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
              float h = clamp((r - uPlanetR - ${CLOUD_BOTTOM.toFixed(1)}) /
                              ${((CLOUD_TOP - CLOUD_BOTTOM)).toFixed(1)}, 0.0, 1.0);
              // Detail LOD: fade the 3rd fbm octave with distance INSTEAD of
              // rescaling the lattice (rescaling planet-frame coords per
              // pixel jumps the noise grid by whole cells -> the concentric
              // "ripple rings" seen when approaching the deck). h grows with
              // sample distance, so the fade is smooth along the ray too.
              float detail = 1.0 / (1.0 + t * (1.0 / 8000.0));
              // ---- unified density: same function the far shell shows ----
              float d = cloudDensity(p, cover, weather, wind, 0.18, detail);
              // edge erosion: high-frequency wisps carve the surface (fades
              // out with distance so far samples stay smooth); hf cells are
              // 640 m — same physical wind divided by that cell size
              float hf = fbm3o(p * (1.0 / 640.0) + wind * (1.0 / 640.0) * vec3(-1.7, 1.0, 0.8), detail * detail);
              d -= (1.0 - d) * hf * 0.35;
              // vertical shaping: rounded bases, domed tops
              d *= smoothstep(0.0, 0.18, h) * (0.55 + 0.45 * smoothstep(1.0, 0.55, h));
              d = clamp(d * 1.5, 0.0, 1.0);
              if (d > 0.015) {
                // light march: 3 samples toward the sun (cheap 1-octave billow)
                float od = 0.0;
                for (int j = 1; j <= 3; j++) {
                  vec3 pl = p + uSunDir * (float(j) * 220.0);
                  float fl = fbm3o(pl * (1.0 / 3000.0) + wind * (1.0 / 3000.0), 1.0);
                  float bl = 1.0 - abs(2.0 * fl - 1.0);
                  // same threshold family as cloudDensity (thr = mix(0.70,0.36)):
                  // the stale low threshold here made shadowing fire almost
                  // everywhere, flattening the deck's shading
                  float thrS = mix(0.70, 0.36, cover);
                  od += smoothstep(thrS, thrS + 0.18, bl * (0.55 + 0.45 * cover)) * 220.0;
                }
                float shadow = exp(-od * 0.0012);     // Beer-Lambert (gentler:
                // the old 0.004 killed the sun term for every interior sample
                // — top-down views of the deck read as a flat dark-gray sheet)
                float powder = 1.0 - exp(-d * 4.0);   // dark edges, bright cores
                // tops catch the sun: height-based ambient brightening
                vec3 lit = vec3(1.0, 0.98, 0.95) * shadow * phase * (0.35 + 0.65 * powder)
                         + vec3(0.55, 0.63, 0.78) * (0.40 + 0.35 * h); // sky ambient
                float aStep = 1.0 - exp(-d * dt * 0.0022); // lower extinction:
                // puffs stay translucent instead of piling into an opaque sheet
                volCol += lit * aStep * volT;
                volT *= 1.0 - aStep;
                if (volT < 0.03) break;
              }
              t += dt;
            }
          }
        }

        // ---------------- composite (shell + volumetric) ----------------
        // Complementary weights (wShell = 1 - wVol above 25 km): coverage
        // alpha = aV*wVol + aS_shell*wShell sums to full coverage at every
        // altitude; color is the opacity-weighted mean of the two layers'
        // lit colors, so puffs stay bright-white through the handoff.
        float aV = 1.0 - volT;            // march coverage (already wVol-scaled below)
        float aS = shellA;                // shell coverage (wShell-scaled in shellA)
        float cov = clamp(aV * wVol + aS * (1.0 - wVol), 0.0, 1.0);
        vec3 col = cov > 0.0001
          ? (volCol * wVol + shellCol * (1.0 - wVol)) / max(wVol + (1.0 - wVol), 0.0001)
          : vec3(0.0);
        float alpha = cov;
        // altitude-based opacity fade inside the band (flying through)
        float inBand = smoothstep(0.0, 0.25, hLayer) * (1.0 - smoothstep(0.75, 1.0, hLayer));
        float farFade = clamp(abs(camAlt - ${((CLOUD_BOTTOM + CLOUD_TOP) / 2).toFixed(1)}) / 6000.0, 0.0, 1.0);
        float bandFade = mix(0.35, 0.95, farFade * inBand + farFade * (1.0 - inBand));
        alpha *= mix(bandFade, 1.0, step(8000.0, camAlt));
        alpha = clamp(alpha, 0.0, 1.0);
        // night fade
        float sunH = dot(up0, uSunDir);
        col *= smoothstep(-0.12, 0.08, sunH);
        alpha *= smoothstep(-0.25, 0.0, sunH) * 0.98 + 0.02;

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
