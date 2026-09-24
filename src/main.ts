import * as THREE from 'three';
import { CameraRig } from './cameraRig';
import { PlanetView } from './cubeSphereLod';
import { makePlanetMaterial, makeStars } from './materials';
import { Hud } from './hud';
import { AutoPilot } from './testAuto';
import { WorldOrigin } from './world';

const R = 6_371_000; // Earth radius, meters

// Surface errors in the HUD so they are observable via DOM text reads.
const errors: string[] = [];
window.addEventListener('error', (e) => {
  errors.push('error: ' + ((e as ErrorEvent).error?.stack ?? e.message));
});
window.addEventListener('unhandledrejection', (e) => {
  errors.push('rejection: ' + String((e as PromiseRejectionEvent).reason));
});
// Surface three.js shader/asset errors in the HUD (they only hit console.*).
const origConsoleError = console.error.bind(console);
console.error = (...args: unknown[]) => {
  errors.push('console: ' + args.map((a) => String(a)).join(' ').slice(0, 200));
  origConsoleError(...args);
};

const urlParams = new URLSearchParams(location.search);

const renderer = new THREE.WebGLRenderer({
  // MSAA switchable from URL for A/B diagnosis (?aa=0).
  antialias: urlParams.get('aa') !== '0',
  // Reversed-Z depth is OPT-IN for now (?revz=1): three.js 0.170's
  // reverseDepthBuffer path has a bug (WebGLState.setReversed tests the OLD
  // `reversed` value, so it sets NEGATIVE_ONE_TO_ONE instead of ZERO_TO_ONE
  // and skips the clear-depth flip), which black-screens every fragment.
  // Our app-side workaround is not yet sufficient; forward 24-bit depth with
  // the altitude-adaptive near plane stays the default until verified.
  reverseDepthBuffer: urlParams.get('revz') === '1',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// WORKAROUND for a three.js 0.170 bug (WebGLState.setReversed):
// the clip-control branch tests the OLD `reversed` value, so enabling
// reversed depth sets NEGATIVE_ONE_TO_ONE instead of ZERO_TO_ONE, and the
// clear-depth flip (1 -> 0) is skipped on the first setClear. Every
// fragment then fails the GEQUAL test -> black frame.
// Fix from app side: set the correct clip mapping, correct the tracked
// depth clear (three thinks it cleared 1; with ZERO_TO_ONE mapping the
// window-space clear must be 0), and prime the depth buffer now.
if ((renderer as unknown as { capabilities: { reverseDepthBuffer: boolean } }).capabilities.reverseDepthBuffer) {
  const glc = renderer.getContext() as WebGL2RenderingContext;
  const clip = glc.getExtension('EXT_clip_control');
  if (clip) {
    clip.clipControlEXT(clip.LOWER_LEFT, clip.ZERO_TO_ONE);
    glc.clearDepth(0);
    glc.clear(glc.DEPTH_BUFFER_BIT);
    // keep three's internal tracker from re-clearing with 1 later
    (
      renderer as unknown as {
        state: { buffers: { depth: { setClear: (d: number) => void } } };
      }
    ).state.buffers.depth.setClear(0);
  } else {
    errors.push('EXT_clip_control missing — reversed-Z unavailable');
  }
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// Floating origin (Phase 2): absolute space lives in `world`; the camera and
// all rendered meshes live in frame-relative space that stays small.
const world = new WorldOrigin();
const ORIGIN_REBASE_M = 2_000; // recenters whenever the camera drifts >2 km

const material = makePlanetMaterial();
const planet = new PlanetView(scene, R, material, {
  maxLevel: 20,
  tauPx: 2,
  res: 65,
  cacheSize: 300,
});
const stars = makeStars(2500, 6e8);
scene.add(stars);

const hud = new Hud('hud');
const absCam = new THREE.Vector3(0, 0, R * 4); // absolute camera position
let rigRef: CameraRig | null = null;
const rig = new CameraRig(
  renderer.domElement,
  new THREE.Vector3(0, 0, R * 4), // start frame-relative == absolute (origin 0)
  new THREE.Vector3(0, 0, 0),
  // Altitude uses the ABSOLUTE camera position (origin-aware).
  () => (rigRef ? Math.max(absCam.length() - R, 0) : Number.POSITIVE_INFINITY),
  // Local zenith in frame-relative space: with a floating origin the planet
  // center sits at -origin, so "up" (away from the center) is +origin
  // normalized. (Getting the sign wrong here tumbles the auto-leveler.)
  () => _up.copy(world.origin).normalize(),
);
rigRef = rig;
const _up = new THREE.Vector3();

window.addEventListener('keydown', (e) => {
  if (e.code === 'KeyG') {
    // Wireframe overlay is drawn in the fragment shader (grid lines on the
    // visible surface only — no back-face edges, no diagonal clutter).
    material.uniforms.uWire.value = material.uniforms.uWire.value > 0.5 ? 0 : 1;
  }
});

// test hook: force wireframe from URL for automated runs
if (urlParams.get('wire') === '1') {
  material.uniforms.uWire.value = 1;
}

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  rig.camera.aspect = window.innerWidth / window.innerHeight;
  rig.camera.updateProjectionMatrix();
});

const fmtDist = (m: number): string =>
  m >= 1e6 ? (m / 1e6).toFixed(2) + ' Mm' : m >= 1e4 ? (m / 1e3).toFixed(1) + ' km' : m.toFixed(1) + ' m';

const auto = new AutoPilot(rig, world);

// Demo-only pixel probe: samples rendered colors so automated verification can
// confirm actual pixels (e.g. planet lit vs. sky), not just stats. null = off.
type ProbeFn = () => string;
let probe: ProbeFn | null = null;
{
  const q = new URLSearchParams(location.search);
  if (q.get('demo')) {
    const gl = renderer.getContext() as WebGL2RenderingContext;
    const w = () => renderer.domElement.width;
    const h = () => renderer.domElement.height;
    const buf = new Uint8Array(4);
    // persistent state for the 'j' jitter probe
    let prevRow: Uint8Array | null = null;
    let prevW = 0;
    let prevH = 0;
    const sample = (px: number, py: number): [number, number, number] => {
      gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      return [buf[0], buf[1], buf[2]];
    };
    const preset = q.get('probe') ?? 'd';
    const dpr = renderer.getPixelRatio();
    const cssW = () => renderer.domElement.clientWidth;
    const cssH = () => renderer.domElement.clientHeight;
    probe = () => {
      const W = w(), H = h();
      let pts: [number, number][];
      if (preset === 'd') {
        // d: center-ish diagonal — planet occupies it in hover/drop views
        pts = [
          [Math.floor(W * 0.5), Math.floor(H * 0.5)],
          [Math.floor(W * 0.25), Math.floor(H * 0.25)],
          [Math.floor(W * 0.5), Math.floor(H * 0.8)],
        ];
      } else if (preset === 'e') {
        // e: left 20% column x 3 rows — planet edge on the limb test
        pts = [
          [Math.floor(W * 0.2), Math.floor(H * 0.5)],
          [Math.floor(W * 0.2), Math.floor(H * 0.25)],
          [Math.floor(W * 0.2), Math.floor(H * 0.75)],
        ];
      } else if (preset === 'g') {
        // g: 6x6 grid over the central 60% — counts anomalies (sky/black or
        // suspiciously dark pixels = cracks) in steady state
        pts = [];
        for (let gy = 0; gy < 6; gy++) {
          for (let gx = 0; gx < 6; gx++) {
            pts.push([
              Math.floor(W * (0.2 + (0.6 * gx) / 5)),
              Math.floor(H * (0.2 + (0.6 * gy) / 5)),
            ]);
          }
        }
      } else if (preset === 'm') {
        // m: 12x8 color matrix over the central 80% — poor man's screenshot
        pts = [];
        for (let my = 0; my < 8; my++) {
          for (let mx = 0; mx < 12; mx++) {
            pts.push([
              Math.floor(W * (0.1 + (0.8 * mx) / 11)),
              Math.floor(H * (0.1 + (0.8 * my) / 7)),
            ]);
          }
        }
      } else if (preset === 'j') {
        // j: jitter meter — framebuffer row diff between frames while the
        // camera is stationary. Persistent state in closure.
        pts = [];
      } else {
        pts = [[Math.floor(W / 2), Math.floor(H / 2)]];
      }
      const rgb = pts.map(([x, y]) => sample(x, y));
      const isSky = (c: [number, number, number]) => c[0] < 15 && c[1] < 15 && c[2] < 15;
      const planetCount = rgb.filter((c) => !isSky(c)).length;
      // anomaly: not sky but abnormally dark relative to the median surface
      // color — indicates a crack, skirt wall, or shadowed seam pixel
      const bright = rgb.filter((c) => !isSky(c));
      let anom = 0;
      if (bright.length > 0) {
        const med = [...bright].sort((a, b) => a[1] - b[1])[Math.floor(bright.length / 2)][1];
        anom = bright.filter((c) => c[1] < med * 0.55).length;
      }
      const show =
        preset === 'g'
          ? `g anomalies=${anom}/${rgb.length}`
          : preset === 'm'
            ? `m\n` + rgb
                .reduce<string[][]>((rows, c, i) => {
                  const r = Math.floor(i / 12);
                  (rows[r] ??= []).push(
                    `${c[0].toString(16).padStart(2, '0')}${c[1].toString(16).padStart(2, '0')}${c[2].toString(16).padStart(2, '0')}`,
                  );
                  return rows;
                }, [])
                .map((r) => r.join(' '))
                .join('\n')
            : `rgb=${rgb.map((c) => c.join(',')).join(' | ')} planet=${planetCount}/${rgb.length}`;
      if (preset === 'j') {
        // Read one horizontal row through the planet center and diff it
        // against the previous frame. Stationary camera => any delta is
        // temporal jitter (vertex snapping, z-fighting shimmer).
        const y = Math.floor(H * 0.5);
        const n = Math.min(Math.floor(W * 0.6), 4096);
        const x0 = Math.floor((W - n) / 2);
        const row = new Uint8Array(n * 4);
        gl.readPixels(x0, y, n, 1, gl.RGBA, gl.UNSIGNED_BYTE, row);
        let d = 0;
        if (prevRow && prevW === n && prevH === H) {
          for (let i = 0; i < n * 4; i += 4) {
            const dd =
              Math.abs(row[i] - prevRow[i]) +
              Math.abs(row[i + 1] - prevRow[i + 1]) +
              Math.abs(row[i + 2] - prevRow[i + 2]);
            if (dd > 2) d++; // ignore 1-LSB noise
          }
        }
        prevRow = row;
        prevW = n;
        prevH = H;
        return `j changed=${d}/${n} rebases=${world.rebaseCount}`;
      }
      return `${show} dpr=${dpr} ${cssW()}x${cssH()}`;
    };
  }
}

let last = performance.now();

// Test hook: run N simulation sub-steps per rendered frame so background
// (rAF-throttled) tabs still advance the simulation at test speed.
const speedup = Math.min(
  Math.max(Number(new URLSearchParams(location.search).get('speedup') ?? '1') || 1, 1),
  60,
);

renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dt = Math.min((now - last) / 1000, 0.25);
  last = now;

  let autoLine: string | null = null;
  let rebased = 0;
  for (let i = 0; i < speedup; i++) {
    // Keep the absolute camera position in sync with the frame-relative one
    // (the rig moves the camera; the autopilot may also teleport it).
    world.abs(rig.camera.position, absCam);

    rig.update(dt);
    autoLine = auto.update(rig, dt);
    world.abs(rig.camera.position, absCam);

    // Floating-origin rebase: recenters the frame origin onto the camera.
    // Only frame-relative values shift; absolute bookkeeping is untouched.
    const shift = world.rebase(rig.camera.position, ORIGIN_REBASE_M);
    if (shift) {
      rebased++;
      planet.forceReposition(world.origin);
    }

    planet.update(rig.camera, world.origin, window.innerHeight);
  }
  // Keep distant scenery centered on the camera (stars are only directions —
  // recenter them each frame so they never sit behind the far plane).
  stars.position.copy(rig.camera.position);

  renderer.render(scene, rig.camera);
  hud.frame(dt);

  const s = planet.stats;
  const alt = Math.max(absCam.length() - R, 0);
  const look = rig.getLookAngles();
  hud.update([
    ...errors.slice(-3),
    ...(autoLine ? [autoLine] : []),
    ...(speedup > 1 ? [`speedup x${speedup}`] : []),
    ...(rebased > 0 ? [`rebase x${rebased} (total ${world.rebaseCount})`] : []),
    ...(probe ? [`probe ${probe()}`] : []),
    `alt ${fmtDist(alt)}  speed ${fmtDist(rig.currentSpeed)}/s  x${rig.speedMultiplier}`,
    `pitch ${look.pitch.toFixed(1)}°  bank ${look.bank.toFixed(1)}°  hdg ${look.heading.toFixed(0)}°  level ${rig.autoLevel ? 'on(R)' : 'off(R)'}`,
    `revz ${(renderer as unknown as { capabilities: { reverseDepthBuffer: boolean } }).capabilities.reverseDepthBuffer ? 'ON' : 'off'}  aa ${urlParams.get('aa') === '0' ? 'off' : 'on'}`,
    `dbg cam=(${rig.camera.position.x.toFixed(0)},${rig.camera.position.y.toFixed(0)},${rig.camera.position.z.toFixed(0)}) org=(${world.origin.x.toFixed(0)},${world.origin.y.toFixed(0)},${world.origin.z.toFixed(0)})`,
    `tiles ${s.visibleTiles}  tris ${(s.triangles / 1000).toFixed(1)}k  maxLvl ${s.maxVisibleLevel}`,
    `queue ${s.pending}  cache ${s.cached}  built ${s.built}  evicted ${s.evicted}  hits ${s.cacheHits}`,
    `drawcalls ${renderer.info.render.calls}`,
    rig.mouseLocked ? 'mouse locked' : 'click = mouse look (arrows also work)',
  ]);
});
