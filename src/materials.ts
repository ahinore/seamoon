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
