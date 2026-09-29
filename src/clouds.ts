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
  };
}

export function makeCloudMesh(planetR: number, uniforms: CloudUniforms): THREE.Mesh {
  // shell at the middle of the layer
  const geo = new THREE.SphereGeometry(planetR + (CLOUD_BOTTOM + CLOUD_TOP) / 2, 128, 96);
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
        x = mod(x, 2048.0);
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
      float fbm4(vec3 p) {
        float a = 0.5, s = 0.0;
        for (int i = 0; i < 4; i++) { s += a * noise3(p); p *= 2.13; a *= 0.5; }
        return s;
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
      float cloudDensity(vec3 p, float cover, float weather, float t2, float edge, float detail) {
        vec3 pw = p * (1.0 / 3000.0); // puff cells ~3 km
        float f1 = fbm3o(pw + vec3(t2, t2 * 1.3, -t2), detail);
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
        float wTime = uTime * 0.002;
        float weather = fbm4(q + vec3(wTime, wTime * 0.7, -wTime * 0.6));
        float lat = asin(clamp(upF.y, -1.0, 1.0));
        float bands = 0.55 + 0.45 * cos(lat * 6.0) * 0.5 + 0.25 * exp(-pow((abs(lat) - 0.15) * 3.0, 2.0));
        float cover = clamp(uCover * bands * 1.6 * weather + (weather - 0.5) * 0.4, 0.0, 1.0);
        cover = pow(cover, 0.7); // bias toward more visible coverage
        float t2 = uTime * 0.006;

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
          float detail = 1.0; // far view: full detail is fine (no marching)
          float d = cloudDensity(shellP, cover, weather, t2, 0.18, detail);
          float hf = fbm3o(shellP * (1.0 / 640.0) + vec3(-t2 * 1.7, t2, t2 * 0.8), detail * detail);
          d -= (1.0 - d) * hf * 0.35;
          d = clamp(d * 1.5, 0.0, 1.0);
          float shellShade = 0.65 + 0.35 * clamp(dot(upF, uSunDir) * 0.5 + 0.5, 0.0, 1.0);
          // Density-driven color: thick cores read warm-white, thin edges
          // cool gray-blue — matches how the volumetric deck shades, and
          // keeps the shell from reading as one flat cream sheet.
          shellCol = mix(vec3(0.72, 0.76, 0.82), vec3(1.02, 1.0, 0.97), smoothstep(0.05, 0.6, d)) * shellShade;
          // Opacity calibrated to the volumetric march it replaces: marching
          // the full ~2.4 km slab accumulates ~1-exp(-d * 6). The old k=2600
          // saturated EVERY pixel to opaque (uniform cream sheet from orbit).
          shellA = 1.0 - exp(-d * 6.0);
          shellA *= wShell;
          // night fade (same terms as the volumetric path)
          float sunHs = dot(up0, uSunDir);
          shellCol *= smoothstep(-0.12, 0.08, sunHs);
          shellA *= smoothstep(-0.25, 0.0, sunHs) * 0.98 + 0.02;
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
              float d = cloudDensity(p, cover, weather, t2, 0.18, detail);
              // edge erosion: high-frequency wisps carve the surface (fades
              // out with distance so far samples stay smooth)
              float hf = fbm3o(p * (1.0 / 640.0) + vec3(-t2 * 1.7, t2, t2 * 0.8), detail * detail);
              d -= (1.0 - d) * hf * 0.35;
              // vertical shaping: rounded bases, domed tops
              d *= smoothstep(0.0, 0.18, h) * (0.55 + 0.45 * smoothstep(1.0, 0.55, h));
              d = clamp(d * 1.5, 0.0, 1.0);
              if (d > 0.015) {
                // light march: 3 samples toward the sun (cheap 1-octave billow)
                float od = 0.0;
                for (int j = 1; j <= 3; j++) {
                  vec3 pl = p + uSunDir * (float(j) * 220.0);
                  float fl = fbm3o(pl * (1.0 / 3000.0) + vec3(t2, t2 * 1.3, -t2), 1.0);
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
