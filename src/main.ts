import * as THREE from 'three';
import { CameraRig } from './cameraRig';
import { PlanetView } from './cubeSphereLod';
import { makePlanetMaterial, makeSeaMaterial, makeStars, makeSunDisc } from './materials';
import { makeAtmosphereMesh, makeAtmosphereUniforms } from './atmosphere';
import { makeCloudMesh, makeCloudUniforms } from './clouds';
import { Hud } from './hud';
import { AutoPilot } from './testAuto';
import { FlightModel } from './flight';
import { WorldOrigin } from './world';
import { makeMoonMaterial } from './moonMaterial';
import { moonPosition, moonPositionAtAngle } from './moonOrbit';
import { MOON_BODY } from './moonBody';
import { EARTH, MOON, nearestFrame } from './frames';
import { FlightAudio, updateFlightAudio } from './audio';

const R = EARTH.radius;

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
// M10.9 procedural flight audio (?audio=0 disables; starts on first gesture)
const audio = new FlightAudio(urlParams.get('audio') !== '0');
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
// M10.3: HDR tone mapping. Terrain/sea shaders output HDR values (sun glint
// 1-exp(-d*fres*0.12) caps at 1, but inscatter * 1.15 and vertex colors on
// the sunlit limb exceed 1.0); without tone mapping those clip to flat white.
// ACES gives filmic rolloff for the glint hotspot and the atmosphere's
// horizon band. ?tonemap=0 keeps the old pass-through for A/B.
// NOTE: our custom ShaderMaterials call <colorspace_fragment> themselves, so
// three's output tonemap chunk never runs for them — we apply the same ACES
// curve inline in each shader (uToneMap flag) instead. Built-in materials
// (Lambert for vegetation) DO get renderer.toneMapping automatically.
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
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

// Sun direction (unit). Slight tilt so day/night and phases are visible.
// Shared by terrain/sea/atmosphere materials (same uniform object).
const sunDir = new THREE.Vector3(1, 0.3, 0.35).normalize();

// ?fov=<deg> (25..120) for wide-FoV testing. The screenshot of the
// screen-edge hole was likely taken with a wide/zoomed-out view — FoV
// widens the frustum but pxPerUnit shrinks proportionally, so SSE splits
// stay correct; only the far-plane reach per angle changes.
const fovDeg = Math.min(Math.max(Number(urlParams.get('fov') ?? '60') || 60, 25), 120);

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

// Phase 5 atmosphere uniforms FIRST (single source of truth): terrain and
// sea materials reference these same objects.
const atmoUniforms = makeAtmosphereUniforms(R);
atmoUniforms.uSunDir.value.copy(sunDir);

// Phase 7: ocean shell — a second quadtree LOD at sea level with per-vertex
// water depth. Coarser than terrain (smooth sphere; waves are shader-side),
// so it gets a smaller cache and slightly laxer error threshold. renderOrder
// after terrain; depth test keeps it hidden under land automatically.
const seaMaterial = makeSeaMaterial({
  uSunDir: atmoUniforms.uSunDir,
  uCamPos: atmoUniforms.uCamPos,
  uOrigin: atmoUniforms.uOrigin,
  uPlanetR: atmoUniforms.uPlanetR,
  uAtmoR: atmoUniforms.uAtmoR,
  uBetaR: atmoUniforms.uBetaR,
  uBetaM: atmoUniforms.uBetaM,
  uHR: atmoUniforms.uHR,
  uHM: atmoUniforms.uHM,
});
const sea = new PlanetView(scene, R, seaMaterial, {
  maxLevel: 19,
  tauPx: 2,
  res: 33,
  cacheSize: 120,
  seaMode: true,
});
if (urlParams.get('sea') === '0') {
  sea.root.visible = false;
}
// M10.3: ?tonemap=0 disables the in-shader ACES curve (A/B diagnosis).
const tonemapOn = urlParams.get('tonemap') !== '0';
if (!tonemapOn) {
  material.uniforms.uToneMap.value = 0;
  seaMaterial.uniforms.uToneMap.value = 0;
  renderer.toneMapping = THREE.NoToneMapping;
}
// M10.4: ?detail=0 disables per-pixel procedural detail splatting (A/B).
if (urlParams.get('detail') === '0') {
  material.uniforms.uDetail.value = 0;
}
const stars = makeStars(6000, 6e8);
scene.add(stars);
// Sun disc (M9.4): placed along sunDir at a fixed camera distance each frame
// (inside the far plane; stars are at 6e8 so the sun sits well inside them).
// ?sun=0 hides it (A/B diagnosis).
const SUN_DIST = 5e8;
const sunDisc = makeSunDisc(SUN_DIST);
scene.add(sunDisc);
if (urlParams.get('sun') === '0') sunDisc.visible = false;
const _sunPos = new THREE.Vector3();

// Phase 5: atmosphere shell. The mesh sits at the frame-relative planet
// center (-origin each frame) and its shader takes the frame-relative
// camera position and the floating origin as uniforms.
const atmosphere = makeAtmosphereMesh(R, atmoUniforms);
scene.add(atmosphere);
// Terrain and sea uniforms are SHARED with the atmosphere: one source of
// truth per frame (uSunDir/uCamPos/uOrigin/betas/scale heights).
material.uniforms.uSunDir = atmoUniforms.uSunDir;
material.uniforms.uCamPos = atmoUniforms.uCamPos;
material.uniforms.uOrigin = atmoUniforms.uOrigin;
material.uniforms.uPlanetR = atmoUniforms.uPlanetR;
material.uniforms.uAtmoR = atmoUniforms.uAtmoR;
material.uniforms.uBetaR = atmoUniforms.uBetaR;
material.uniforms.uBetaM = atmoUniforms.uBetaM;
material.uniforms.uHR = atmoUniforms.uHR;
material.uniforms.uHM = atmoUniforms.uHM;
// ?atmo=0 disables the shell (A/B for diagnosis)
if (urlParams.get('atmo') === '0') atmosphere.visible = false;

// Phase 8: cloud shell (same inverted-hull pattern; ?clouds=0 disables).
const cloudUniforms = makeCloudUniforms(R);
cloudUniforms.uSunDir = atmoUniforms.uSunDir; // share the sun object
// M11n4 cloud shadows: terrain/sea evaluate the deck's MACRO weather field —
// share the coverage uniform and the (wt-freezable) weather clock so the
// shadows always agree with the visible deck's weather phase.
material.uniforms.uCover = cloudUniforms.uCover;
material.uniforms.uTime = cloudUniforms.uTime;
seaMaterial.uniforms.uCover = cloudUniforms.uCover;
seaMaterial.uniforms.uWTime = cloudUniforms.uTime;
if (urlParams.get('cloudshadow') === '0') {
  material.uniforms.uCloudShadow.value = 0;
  seaMaterial.uniforms.uCloudShadow.value = 0;
}
const clouds = makeCloudMesh(R, cloudUniforms);
scene.add(clouds);
if (urlParams.get('clouds') === '0') clouds.visible = false;
// A/B for the dual-hull work: hide one hull at a time (?nearhull=0 /
// ?farhull=0). clouds.children = [nearMesh, farMesh].
if (urlParams.get('nearhull') === '0') clouds.children[0].visible = false;
if (urlParams.get('farhull') === '0') clouds.children[1].visible = false;
// Same-session A/B (tools/abshot.mjs): expose the group for toggling.
if (urlParams.has('abshot')) {
  (window as unknown as { __clouds: unknown }).__clouds = clouds;
  // M11n: expose the cloud uniforms too (uCover sweeps for deck tests)
  (window as unknown as { __cloudU: unknown }).__cloudU = cloudUniforms;
}
// Probe access for tools/reentry.mjs: the flight model (heat, note,
// phase). Assigned after the FlightModel exists (it's declared at line
// ~301) — registered via a lazy getter on first frame instead.
let __flightExposed = false;
// ?cloudbg=1: color-code which term suppresses the far deck (red = fbm
// below threshold, green = weather gate, yellow = partial). 2 = march
// probe for the below-deck view.
if (urlParams.get('cloudbg') === '1') cloudUniforms.uCloudDbg.value = 1;
if (urlParams.get('cloudbg') === '2') cloudUniforms.uCloudDbg.value = 2;
if (urlParams.get('cloudbg') === '3') cloudUniforms.uCloudDbg.value = 3;
if (urlParams.get('cloudbg') === '4') cloudUniforms.uCloudDbg.value = 4;

// M10.5: vegetation lighting shares the same sun-direction object as the
// terrain/atmosphere — trees and ground can never disagree on the light.
planet.vegMaterial.uniforms.uSunDir = atmoUniforms.uSunDir;

// ---- Phase 9: the moon -------------------------------------------------
// Reuses the ENTIRE cube-sphere LOD pipeline via the BodySurface interface:
// same quadtree, same tile builder, same material pattern — a different
// radius, height function (craters), and a Lambert no-atmosphere material.
// ?moon=0 hides it (A/B diagnosis).
const R_MOON = MOON.radius;
const moonMaterial = makeMoonMaterial(atmoUniforms.uSunDir);
const moonView = new PlanetView(scene, R_MOON, moonMaterial, {
  maxLevel: 20,
  tauPx: 2,
  res: 65,
  // M11n3b: 300 starved the SURFACE-view horizon ring — from 2 km the
  // quadtree wants ~700 visible tiles (center L12 + the grazing horizon
  // ring to L13+), so the far tiles lost the build/evict race and the
  // horizon band rendered as bare background (the moon black band).
  cacheSize: 900,
  // M11n9h: the moon has no haze — boost the horizon-ring split so the
  // grazing band fills with terrain instead of void (earth keeps 0: its
  // band is haze-covered and the boost costs draw calls)
  grazingBoost: 5,
  }, MOON_BODY);
if (urlParams.get('moon') === '0') moonView.root.visible = false;
// sim clock for the orbit (performance.now-based; deterministic per session)
const t0Sim = performance.now() / 1000;

const hud = new Hud('hud');
const absCam = new THREE.Vector3(0, 0, R * 4); // absolute camera position
let rigRef: CameraRig | null = null;
// Nearest-body reference (Phase 9 M9.2, registry-driven since M9.6): all
// single-body references (altitude for near-plane/speed, zenith for
// auto-level) resolve against the CURRENT nearest body's frame.
let nearBody: 'earth' | 'moon' = 'earth';
const _bodyUp = new THREE.Vector3();
const rig = new CameraRig(
  renderer.domElement,
  new THREE.Vector3(0, 0, R * 4), // start frame-relative == absolute (origin 0)
  new THREE.Vector3(0, 0, 0),
  // Altitude uses the ABSOLUTE camera position relative to the NEAREST body
  // (origin-aware, moon-aware).
  () => {
    if (!rigRef) return Number.POSITIVE_INFINITY;
    if (nearBody === 'moon') {
      return Math.max(absCam.distanceTo(MOON.center) - R_MOON, 0);
    }
    return Math.max(absCam.length() - R, 0);
  },
  // Local zenith: radial away from the nearest body's center. With a floating
  // origin the planet center sits at -origin (earth) or MOON.center-origin
  // (moon) in frame space; "up" points away from that center.
  () => {
    if (nearBody === 'moon') {
      _bodyUp.copy(rigRef ? rigRef.camera.position : _up.set(0, 0, 0));
      _bodyUp.add(world.origin).sub(MOON.center);
      return _bodyUp.normalize();
    }
    return _up.copy(world.origin).normalize();
  },
);
rig.camera.fov = fovDeg;
// Initial aspect: the rig's camera is constructed with aspect=1 and only the
// 'resize' event updated it — so on load the scene rendered square-projected
// (planet visibly squashed into an ellipse) until the user resized the window.
// Match the actual viewport once at startup; the listener handles the rest.
rig.camera.aspect = window.innerWidth / window.innerHeight;
rig.camera.updateProjectionMatrix();
rigRef = rig;
const _up = new THREE.Vector3();

window.addEventListener('keydown', (e) => {
  // key repeat fires toggle handlers many times per second — a held G
  // would leave the wireframe in a random state ("wires appear for no
  // reason"). First press only.
  if (e.repeat) return;
  if (e.code === 'KeyG') {
    // Wireframe overlay is drawn in the fragment shader (grid lines on the
    // visible surface only — no back-face edges, no diagonal clutter).
    // Toggle BOTH bodies: G on the moon used to do nothing (only the
    // earth material was flipped).
    const on = material.uniforms.uWire.value > 0.5 ? 0 : 1;
    material.uniforms.uWire.value = on;
    moonMaterial.uniforms.uWire.value = on;
    hud.setNote('wire ' + (on ? 'ON' : 'OFF'));
  }
  if (e.code === 'KeyH') {
    // Home: teleport back to the session spawn position/orientation. The
    // spawn pose is stored in ABSOLUTE coordinates, so this is exact no
    // matter how many floating-origin rebases happened since.
    auto.goHome(rig);
    world.abs(rig.camera.position, absCam);
  }
});

// test hook: force wireframe from URL for automated runs
if (urlParams.get('wire') === '1') {
  material.uniforms.uWire.value = 1;
}
// M11j: ?lvl=1 distance-band debug on the moon material
if (urlParams.get('lvl') === '1') {
  moonMaterial.uniforms.uLevelDebug.value = 1;
}

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  rig.camera.aspect = window.innerWidth / window.innerHeight;
  rig.camera.updateProjectionMatrix();
});

const fmtDist = (m: number): string =>
  m >= 1e6 ? (m / 1e6).toFixed(2) + ' Mm' : m >= 1e4 ? (m / 1e3).toFixed(1) + ' km' : m.toFixed(1) + ' m';

// M11n9k: visible version tag — bump on every cloud/renderer change so a
// stale cached module is instantly obvious in screenshots
const SIM_VERSION = 'sim v11.9k-aerial';

const auto = new AutoPilot(rig, world);
world.abs(rig.camera.position, absCam); // autopilot placed the camera

// Phase 6: flight model. F toggles between the free camera and the aircraft
// (?demo=fly starts in the aircraft with the scripted takeoff->landing
// mission). In flight mode the autopilot test driver is disabled.
const flight = new FlightModel(rig, world);
if (flight.mode === 'fly' || flight.mode === 'lunar' || flight.mode === 'orbital') {
  rig.stickMode = true;
  flight.reset(); // aircraft owns the camera from frame 1 in flight modes
  auto.suspend();
  world.abs(rig.camera.position, absCam);
}

/** Switch free-camera <-> aircraft (F key). */
function toggleFlight(): void {
  if (rig.stickMode) {
    // exit to free camera at the current pose
    rig.stickMode = false;
    rig.ctl.pitch = rig.ctl.roll = rig.ctl.yaw = 0;
    auto.resume();
  } else {
    flight.reset();
    rig.stickMode = true;
    auto.suspend();
  }
}

window.addEventListener('keydown', (e) => {
  if (e.code === 'KeyF' && !e.repeat) toggleFlight();
  if (e.code === 'KeyR' && rig.stickMode && !e.repeat) flight.reset();
  audio.resume(); // any key: browsers need a gesture to start sound
});
document.addEventListener('pointerdown', () => audio.resume());
// In flight mode the rig's own R handler must not fire (R = respawn there).
// The rig listens on window too; guard it via stickMode inside the rig.

// Demo-only pixel probe: samples rendered colors so automated verification can
// confirm actual pixels (e.g. planet lit vs. sky), not just stats. null = off.
// M11h: readPixels stalls the GPU pipeline — throttle to 10 Hz (probe
// consumers poll the HUD text; sub-100 ms latency is plenty).
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
    let lastProbe = 0;
    let cached = '';
    const sample = (px: number, py: number): [number, number, number] => {
      gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      return [buf[0], buf[1], buf[2]];
    };
    const preset = q.get('probe') ?? 'd';
    const dpr = renderer.getPixelRatio();
    const cssW = () => renderer.domElement.clientWidth;
    const cssH = () => renderer.domElement.clientHeight;
    probe = () => {
      // M11h: serve the cached readout at >10 Hz — readPixels is a
      // synchronous GPU stall, and the probe runs every frame otherwise
      const now = performance.now();
      if (now - lastProbe < 100 && cached) return cached;
      lastProbe = now;
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
      cached = `${show} dpr=${dpr} ${cssW()}x${cssH()}`;
      return cached;
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
// M11i: chute staging → one-shot audio crack (chuteCrack was wired nowhere)
let lastChuteEvents = 0;
// M11j: moonshot stage-separation events
let lastMsStageEvents = 0;
for (let i = 0; i < speedup; i++) {
  // Keep the absolute camera position in sync with the frame-relative one
  // (the rig moves the camera; the autopilot may also teleport it).
  world.abs(rig.camera.position, absCam);

  if (rig.stickMode) {
    rig.update(dt);
    // M11n9j: fixed-timestep substepping — the flight physics (reentry G,
    // chute deploy) is dt-sensitive, and heavy cloud views can drop the
    // frame rate to 20-30 fps, coarsening the integration and spiking the
    // sampled peakG (54 -> 150 run-to-run). Substep at 16 ms granularity
    // so the physics is frame-rate independent.
    const nSteps = Math.max(1, Math.ceil(dt / 0.016));
    for (let s = 0; s < nSteps; s++) flight.step(dt / nSteps);
    world.abs(rig.camera.position, absCam);
    // M10.9 audio: booster on during the lob boost phase
    updateFlightAudio(audio, flight, flight.boostPhase);
    // M11i: chute crack on each staging event (drogue, main)
    if (flight.chuteEvents !== lastChuteEvents) {
      lastChuteEvents = flight.chuteEvents;
      audio.chuteCrack();
    }
    // M11j: moonshot stage separation uses the same percussive crack
    if (flight.msStageEvents !== lastMsStageEvents) {
      lastMsStageEvents = flight.msStageEvents;
      audio.chuteCrack();
    }
  } else {
    rig.update(dt);
    autoLine = auto.update(rig, dt);
    world.abs(rig.camera.position, absCam);
  }

  // Floating-origin rebase: recenters the frame origin onto the camera.
  // Only frame-relative values shift; absolute bookkeeping is untouched.
  const shift = world.rebase(rig.camera.position, ORIGIN_REBASE_M);
  if (shift) {
    rebased++;
    planet.forceReposition(world.origin);
    sea.forceReposition(world.origin);
    moonView.forceReposition(world.origin); // M11c: the moon tiles went stale too
  }

  planet.update(rig.camera, world.origin, window.innerHeight);
  sea.update(rig.camera, world.origin, window.innerHeight);
  // Moon: place tiles at center + bodyCenter - origin (all double).
  moonView.update(rig.camera, world.origin, window.innerHeight);
}
  // Keep distant scenery centered on the camera (stars are only directions —
  // recenter them each frame so they never sit behind the far plane).
  stars.position.copy(rig.camera.position);
  // Sun disc: camera-centered quad along the fixed sun direction, oriented
  // to face the camera (billboard). far = 2e9 keeps 5e8 inside the frustum.
  _sunPos.copy(rig.camera.position).addScaledVector(sunDir, SUN_DIST);
  sunDisc.position.copy(_sunPos);
  sunDisc.quaternion.copy(rig.camera.quaternion);

  // Atmosphere shell follows the planet center (-origin in frame space) and
  // feeds the shader the camera pose in the same frame. The terrain material
  // shares the aerial-perspective uniforms.
  atmosphere.position.copy(world.origin).negate();
  atmoUniforms.uCamPos.value.copy(rig.camera.position);
  atmoUniforms.uOrigin.value.copy(world.origin);
  clouds.position.copy(world.origin).negate();
  // M11n3: the NEAR hull follows the camera. A planet-centered near hull
  // would not rasterize at all from inside (its fragments vanished while
  // the same geometry centered on the camera rendered fine — probe aV=4),
  // which blanked every cloud above the horizon when inside/below the
  // deck. The shader math is position-independent: ro comes from uCamPos,
  // and rd = normalize(vWorld - uCamPos) is the fragment direction either
  // way, so moving the mesh only moves the rasterization footprint.
  clouds.children[0].position.copy(absCam);
  // M11n3: draw the near hull AFTER the far hull. Both share renderOrder 4
  // and the transparent sort can order far-then-near; the far hull's own
  // alpha=1 debug pixels (and its shell color) would then overwrite the
  // marched volume and clouds vanished from inside/below the deck.
  clouds.children[0].renderOrder = 5;
  cloudUniforms.uCamPos.value.copy(rig.camera.position);
  cloudUniforms.uOrigin.value.copy(world.origin);
  // ?wt=SECONDS: freeze the weather-clock for deterministic cloud A/B
  // (the field drifts with wall clock; tests must pin it to compare
  // march vs far shell at the same weather phase).
  {
    const wt = urlParams.get('wt');
    cloudUniforms.uTime.value = wt !== null ? parseFloat(wt) : performance.now() / 1000;
  }
  cloudUniforms.uTanHalfFov.value = Math.tan(THREE.MathUtils.degToRad(rig.camera.fov) * 0.5);
  cloudUniforms.uViewportH.value = window.innerHeight;
  // Moon orbit: the absolute position is written into the MOON frame center
  // (M9.6: the registry entry is the single source of truth — PlanetView,
  // flight physics and the nearest-body rule all read this one vector).
  // ?moonangle=<deg> (test hook) freezes the orbit at a fixed angle.
  // M11j: the moonshot demo OWNS the moon's position (placed at the TLI
  // antipode by the flight model at spawn) — the live clock must not
  // overwrite it or the transfer misses the moon by the drift.
  if (auto.moonAngle !== null) {
    moonPositionAtAngle(auto.moonAngle, MOON.center);
  } else if (flight.moonshotActive) {
    // keep MOON.center = flight.moonC (the demo's frozen placement)
    MOON.center.copy(flight.moonCenter);
  } else {
    moonPosition(performance.now() / 1000 - t0Sim, MOON.center);
  }
  moonView.bodyCenter.copy(MOON.center);
  moonView.root.position.copy(moonView.bodyCenter).sub(world.origin);
  // Nearest-body selection (M9.2): whichever surface the camera is closest
  // to (SOI handoff reference; steers the rig's altitude/zenith). The rule
  // lives in frames.ts (M9.6) — one definition for the whole app.
  nearBody = nearestFrame(absCam);
  // M11n3: the near cloud hull (R+2.6 km, camera-following) exists around
  // the EARTH only. The rig caps its near plane at 40% of the hull
  // clearance so the hull can never fall behind the near plane.
  rig.nearCapAlt = nearBody === 'moon' ? Infinity : 2600;
  // M11c/M11k lander floodlight: on EITHER body's night side below 60 km
  // AGL the camera carries a warm point light that pools on the terrain
  // ahead — the final approach is otherwise pitch black (the M11c moon
  // landing and the M11g return's earth night-side touchdown both need
  // it; the same uniform pair now lives in both materials).
  {
    let lampOn = 0;
    const center = nearBody === 'moon' ? MOON.center : _up.set(0, 0, 0);
    const bodyR = nearBody === 'moon' ? R_MOON : R;
    const altN = absCam.distanceTo(center) - bodyR;
    if (altN < 60000) {
      _bodyUp.copy(absCam).sub(center).normalize();
      const sunDot = _bodyUp.dot(atmoUniforms.uSunDir.value);
      // night side: sunDot < -0.1; ramp the lamp in as sun falls away
      lampOn = THREE.MathUtils.clamp(-sunDot * 3.0, 0, 1);
    }
    const lamMoon = moonMaterial.uniforms;
    const lamEarth = material.uniforms;
    lamMoon.uLampOn.value = lampOn;
    lamEarth.uLampOn.value = lampOn;
    if (lampOn > 0.001) {
      // The shaders compare uLampPos against vWorld, which is FRAME-relative
      // (modelMatrix positions). absCam is ABSOLUTE — subtract the floating
      // origin or the pool is displaced by |origin| (up to the 2 km rebase
      // threshold) and its inverse-square atten collapses to zero. This was
      // the flaky "night approach is pitch black" bug: it worked right
      // after spawn (origin 0) and died after any rebase.
      _bodyUp.copy(absCam).sub(world.origin);
      lamMoon.uLampPos.value.copy(_bodyUp);
      lamEarth.uLampPos.value.copy(_bodyUp);
    }
  }
  // sea material shares the terrain's uniform objects (updated above); only
  // its own time / viewport uniforms need ticking here.
  seaMaterial.uniforms.uTime.value = performance.now() / 1000;
  seaMaterial.uniforms.uFovTan.value = Math.tan(THREE.MathUtils.degToRad(rig.camera.fov) * 0.5);
  seaMaterial.uniforms.uViewportH.value = window.innerHeight;

  renderer.render(scene, rig.camera);
  hud.frame(dt);

  // Probe access for tools/reentry.mjs: expose the flight model once it
  // exists (declared below this point in module scope).
  if (!__flightExposed) {
    __flightExposed = true;
    (window as unknown as { __flight: unknown }).__flight = flight;
    (window as unknown as { __audio: unknown }).__audio = audio; // M10.9 probe
    (window as unknown as { __moonmat: unknown }).__moonmat = moonMaterial; // M11c probe
    (window as unknown as { __moonview: unknown }).__moonview = moonView; // M11c probe
    (window as unknown as { __world: unknown }).__world = world; // M11c probe
    // M11h perf probes: scene traversal + renderer memory counters
    (window as unknown as { __scene: unknown }).__scene = scene;
    (window as unknown as { __renderer: unknown }).__renderer = renderer;
    (window as unknown as { __views: unknown }).__views = { planet, sea, moonView };
    (window as unknown as { __rig: unknown }).__rig = rig; // M11i probe (camera pose)
  }

  // M10.8 reentry plasma overlay: brightness follows the flight model's
  // normalized stagnation heat (orbital mode, earth atmosphere only).
  {
    const plasma = document.getElementById('plasma') as HTMLElement | null;
    if (plasma) {
      const h = flight.heat;
      // heat^0.65: the glow ramps early (a 0.3 heat = 1MW/m² already looks
      // hot) instead of hiding behind the linear curve until the peak
      plasma.style.opacity = (0.95 * Math.pow(h, 0.65)).toFixed(3);
    }
  }

  const s = planet.stats;
  // HUD altitude is relative to the NEAREST body (lunar hover shows lunar alt)
  const alt = nearBody === 'moon'
    ? Math.max(absCam.distanceTo(MOON.center) - R_MOON, 0)
    : Math.max(absCam.length() - R, 0);
  const look = rig.getLookAngles();
  const moonDist = MOON.center.length() - R_MOON;
  hud.update([
    ...errors.slice(-3),
    SIM_VERSION,
    ...(autoLine ? [autoLine] : []),
    ...(rig.stickMode ? [flight.statusLine()] : []),
    ...flight.missionSummary(),
    ...(speedup > 1 ? [`speedup x${speedup}`] : []),
    ...(rebased > 0 ? [`rebase x${rebased} (total ${world.rebaseCount})`] : []),
    ...(probe ? [`probe ${probe()}`] : []),
    `alt ${fmtDist(alt)} (${nearBody})  speed ${fmtDist(rig.stickMode && !flight.frozen ? flight.tGs : rig.currentSpeed)}/s  x${rig.speedMultiplier}`,
    `moon dist ${fmtDist(moonDist)}  tiles ${moonView.stats.visibleTiles} L${moonView.stats.maxVisibleLevel}`,
    `pitch ${look.pitch.toFixed(1)}°  bank ${look.bank.toFixed(1)}°  hdg ${look.heading.toFixed(0)}°  level ${rig.autoLevel ? 'on(R)' : 'off(R)'}`,
    `revz ${(renderer as unknown as { capabilities: { reverseDepthBuffer: boolean } }).capabilities.reverseDepthBuffer ? 'ON' : 'off'}  aa ${urlParams.get('aa') === '0' ? 'off' : 'on'}`,
    `dbg cam=(${rig.camera.position.x.toFixed(0)},${rig.camera.position.y.toFixed(0)},${rig.camera.position.z.toFixed(0)}) org=(${world.origin.x.toFixed(0)},${world.origin.y.toFixed(0)},${world.origin.z.toFixed(0)})`,
    `tiles ${s.visibleTiles}  tris ${(s.triangles / 1000).toFixed(1)}k  maxLvl ${s.maxVisibleLevel}`,
    `sea tiles ${sea.stats.visibleTiles}  tris ${(sea.stats.triangles / 1000).toFixed(1)}k  maxLvl ${sea.stats.maxVisibleLevel}`,
    `queue ${s.pending}  cache ${s.cached}  built ${s.built} (wk ${s.workerBuilt})  evicted ${s.evicted}  hits ${s.cacheHits}  veg ${planet.vegTiles}`,
    `drawcalls ${renderer.info.render.calls}`,
    rig.mouseLocked ? 'mouse locked' : 'click = mouse look (arrows also work)',
  ]);
});
