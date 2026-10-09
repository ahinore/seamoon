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
import { buildMoonFallbackGeometry } from './moon';
import { moonPosition, moonPositionAtAngle, MOON_ORBIT_R, INCLINATION } from './moonOrbit';
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

// M11w20g: per-frame camera pose log (?camlog=1) — absolute position +
// world quaternion + autopilot status each rendered frame, exposed as
// window.__camLog for jump/teleport analysis (e.g. tour hold->travel).
const CAMLOG = urlParams.get('camlog') === '1';
if (CAMLOG) {
  (window as any).__camLog = [];
}

const renderer = new THREE.WebGLRenderer({
  // MSAA switchable from URL for A/B diagnosis (?aa=0).
  antialias: urlParams.get('aa') !== '0',
  // Reversed-Z depth is the DEFAULT (M11w): the three.js 0.170
  // reverseDepthBuffer path has a bug (WebGLState.setReversed tests the OLD
  // `reversed` value, so enabling it leaves NEGATIVE_ONE_TO_ONE clip control
  // and skips the clear-depth flip), but the app-side workaround below now
  // fully corrects both — A/B across the 7 smoke views matches forward-Z
  // pixel-stat for pixel-stat. ?revz=0 falls back to forward 24-bit depth
  // with the altitude-adaptive near plane for diagnosis.
  reverseDepthBuffer: urlParams.get('revz') !== '0',
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

// WORKAROUND for a three.js 0.170 bug (WebGLState.setReversed) — now the
// reason reversed-Z can be the DEFAULT:
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
    // three's setReversed(true) fails to update the clear depth when its
    // tracker is still null (oldDepth === null -> setClear no-ops), so GL
    // clearDepth stays at the default 1 — which in reversed depth is the
    // NEAR plane, and the GREATER test then rejects every fragment.
    // setClear takes USER-space depth (default 1) and maps it itself
    // (1 - 1 = 0): this fixes both the GL state and the tracker in one
    // call. The previous workaround called setClear(0), which mapped to
    // gl.clearDepth(1) and re-broke every frame's clear (black frame).
    (
      renderer as unknown as {
        state: { buffers: { depth: { setClear: (d: number) => void } } };
      }
    ).state.buffers.depth.setClear(1);
    glc.clearDepth(0);
    glc.clear(glc.DEPTH_BUFFER_BIT);
  } else {
    errors.push('EXT_clip_control missing — reversed-Z unavailable');
  }
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// Sun direction (unit). M11w20c: tilted toward +Z so that from the moon's
// tour position (orbit angle 180, anti-Earth X) the Earth shows ~half lit
// (illuminated fraction = (1 - sun.x)/2; the old x=0.91 gave a 4% crescent).
// Shared by terrain/sea/atmosphere materials (same uniform object).
const sunDir = new THREE.Vector3(0.1, 0.35, 0.93).normalize();

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
// ?nocull=1: test hook — disable the tile frustum cull (mis-cull diagnosis)
if (urlParams.get('nocull') === '1') {
  (planet as unknown as { o: { noFrustumCull?: boolean } }).o.noFrustumCull = true;
  (sea as unknown as { o: { noFrustumCull?: boolean } }).o.noFrustumCull = true;
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
// M11w20f: the milky way band was removed per user feedback (it read as an
// unnatural white stripe — only the star points remain).
scene.add(stars);
// Sun disc (M9.4): placed along sunDir at a fixed camera distance each frame
// (inside the far plane; stars are at 6e8 so the sun sits well inside them).
// ?sun=0 hides it (A/B diagnosis).
const SUN_DIST = 5e8;
const sunDisc = makeSunDisc(SUN_DIST);
scene.add(sunDisc);
if (urlParams.get('sun') === '0') sunDisc.visible = false;
const _sunPos = new THREE.Vector3();
const _moonNdc = new THREE.Vector3(); // loddbg: moon screen position (NDC)

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
// probe for the below-deck view. Floats pass straight through for the
// shader-side probes (2.5 / 6 / 7.5 ...).
{
  const cdb = parseFloat(urlParams.get('cloudbg') ?? '0');
  if (cdb > 0) cloudUniforms.uCloudDbg.value = cdb;
}

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
// M11n9r: global fallback sphere under the moon's tile quadtree — the
// grazing horizon band renders guaranteed terrain instead of void when
// the build budget can't fill the ring (the black band). The material is
// cloned so the fallback can carry its own polygon offset; the shared
// uniforms the clone deep-copied are re-pointed at the originals so the
// sun direction and the floodlight stay in sync.
const moonFallbackMat = moonMaterial.clone();
moonFallbackMat.uniforms.uSunDir = atmoUniforms.uSunDir;
moonFallbackMat.uniforms.uLampPos = moonMaterial.uniforms.uLampPos;
moonFallbackMat.uniforms.uLampOn = moonMaterial.uniforms.uLampOn;
moonFallbackMat.side = THREE.FrontSide; // a closed sphere seen from outside
moonFallbackMat.polygonOffset = true; // extra depth push: tiles always win
moonFallbackMat.polygonOffsetFactor = 2;
const moonFallback = new THREE.Mesh(buildMoonFallbackGeometry(), moonFallbackMat);
moonFallback.renderOrder = -3; // after sun disc (-5), before tiles (0)
// M11w13: the fallback lives in the SCENE, not under moonView.root. The
// tile meshes are root children whose positions already carry
// `node.center + bodyCenter - origin` (the absolute->frame map baked in at
// show()/reposition time), so root itself must stay at (0,0,0) — main used
// to ALSO translate root by `bodyCenter - origin`, double-counting the
// offset: the moon's tiles rendered at 2x (bodyCenter - origin), i.e. a
// displaced ghost sphere floating in front of the correctly-placed
// fallback (the "two nested moons" report; Earth never showed it because
// its bodyCenter is 0). The fallback is positioned per-frame below.
scene.add(moonFallback);
// sim clock for the orbit (performance.now-based; deterministic per session)
const t0Sim = performance.now() / 1000;

const hud = new Hud('hud');

// M11w16 NAV MAP v2 + HUD BEARING ARROWS: the user could not tell where
// they were ("マップが動かない") because v1 auto-framed Earth+Moon+ship,
// which renormalizes the layout and looks static. v2 is a ship-centered
// TOP-DOWN plane map ("自分の上から見た平面"):
//   - the ship is always at the panel center; Earth and Moon are drawn at
//     their true planar offsets (world XZ, the ~orbital plane) so the dots
//     visibly slide as you travel
//   - screen up = the camera's forward direction projected on the map
//     plane, so the map rotates as you turn and "what you face" is up
//   - zoom keys to the FARTHER body (rmax = 0.85 x max(dE,dM), clamped):
//     near a body its dot sits at the center under you and slides outward
//     as you leave; the other body stays visible at the panel rim
//   - radii still exaggerated x15 (capped to 45% of the panel so a close
//     body does not swallow the map)
// Plus a fullscreen HUD arrow layer (#dirs): a colored chevron + label for
// Earth and Moon, always clamped to a border rectangle around the screen
// center along the shortest on-screen direction — never on the body itself
// (it overlapped the planet) and never lost when the body goes off-screen.
const mapDiv = document.getElementById('map3d') as HTMLDivElement | null;
const mapCanvas = document.getElementById('map3dgl') as HTMLCanvasElement | null;
const radarCanvas = document.getElementById('radar') as HTMLCanvasElement | null;
const radarCtx = radarCanvas ? radarCanvas.getContext('2d') : null;
const dirsCanvas = document.getElementById('dirs') as HTMLCanvasElement | null;
const dirsCtx = dirsCanvas ? dirsCanvas.getContext('2d') : null;
const radarEnabled = urlParams.get('radar') !== '0';
let radarAcc = 1; // draw on the first frame
const _radarEarth = new THREE.Vector3(0, 0, 0); // Earth center, absolute

const MAP_S = 200; // css px square
const mapRenderer = mapCanvas && radarCtx
  ? new THREE.WebGLRenderer({ canvas: mapCanvas, alpha: true, antialias: true })
  : null;
let mapScene: THREE.Scene | null = null;
let mapCam: THREE.PerspectiveCamera | null = null;
let mapEarth: THREE.Mesh | null = null;
let mapMoon: THREE.Mesh | null = null;
let mapRing: THREE.LineLoop | null = null;
let mapShip: THREE.Group | null = null;
let mapShipRay: THREE.Line | null = null;
let mapU = 1 / 2.1e8;              // meters -> map units (lerped)
let mapUT = mapU;                  // 10 Hz zoom target
const mapUp = new THREE.Vector3(0, 0, -1); // persistent screen-up (world XZ)
const _mapF = new THREE.Vector3();
const _mapUp = new THREE.Vector3();
const _mapV = new THREE.Vector3();
const _dirsQ = new THREE.Quaternion();
const _dirsD = new THREE.Vector3();

if (mapRenderer) {
  mapRenderer.setSize(MAP_S, MAP_S, false); // CSS size comes from #map3d
  mapRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  mapRenderer.setClearColor(0x000000, 0);
  mapScene = new THREE.Scene();
  mapCam = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(0.3, 1, 0.4); // near-vertical: the map is viewed top-down
  mapScene.add(sun, new THREE.AmbientLight(0xffffff, 0.5));
  const eMat = new THREE.MeshLambertMaterial(); eMat.color.setHex(0x4d8fdb);
  const mMat = new THREE.MeshLambertMaterial(); mMat.color.setHex(0xc9c9d6);
  mapEarth = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), eMat);
  mapMoon = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), mMat);
  mapScene.add(mapEarth, mapMoon);
  // the moon's true orbit ring (radius MOON_ORBIT_R, XZ circle inclined
  // around X exactly like moonPositionAtAngle) — the map's reference plane
  const rp: number[] = [];
  const ci = Math.cos(INCLINATION), si = Math.sin(INCLINATION);
  for (let k = 0; k < 128; k++) {
    const a = (k / 128) * Math.PI * 2;
    rp.push(Math.cos(a), Math.sin(a) * si, Math.sin(a) * ci);
  }
  const rg = new THREE.BufferGeometry();
  rg.setAttribute('position', new THREE.Float32BufferAttribute(rp, 3));
  mapRing = new THREE.LineLoop(rg, new THREE.LineBasicMaterial({ color: 0x88aadd, transparent: true, opacity: 0.3 }));
  mapScene.add(mapRing);
  // ship: octahedron marker + a ray along the camera's forward (-Z)
  mapShip = new THREE.Group();
  mapShip.add(new THREE.Mesh(new THREE.OctahedronGeometry(1), new THREE.MeshBasicMaterial({ color: 0xffd27f })));
  mapShipRay = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1)]),
    new THREE.LineBasicMaterial({ color: 0xffd27f, transparent: true, opacity: 0.9 }));
  mapShip.add(mapShipRay);
  mapScene.add(mapShip);
}

// M11w16 fullscreen HUD bearing arrows (drawn under the map panel)
function drawDirs(show: boolean): void {
  if (!dirsCtx || !dirsCanvas) return;
  if (!show) { dirsCanvas.style.display = 'none'; return; }
  const W = window.innerWidth, H = window.innerHeight;
  if (dirsCanvas.width !== W || dirsCanvas.height !== H) {
    dirsCanvas.width = W; dirsCanvas.height = H;
  }
  dirsCanvas.style.display = 'block';
  const ctx = dirsCtx;
  ctx.clearRect(0, 0, W, H);
  _dirsQ.copy(rig.camera.quaternion).invert();
  const tf = Math.tan(rig.camera.fov * Math.PI / 360);
  const aspect = rig.camera.aspect || 1;
  const cx = W / 2, cy = H / 2;
  const placed: { x: number; y: number }[] = [];
  // pass 1: project both bodies (camera space: -Z front, +X right, +Y up)
  const bodies = [
    { bodyAbs: _radarEarth, color: '#6fb5ff', label: 'Earth', bodyR: R },
    { bodyAbs: MOON.center, color: '#d0d0dc', label: 'Moon', bodyR: R_MOON },
  ].map((b) => {
    _dirsD.copy(b.bodyAbs).sub(absCam).normalize().applyQuaternion(_dirsQ);
    const front = _dirsD.z < 0;
    let sx: number, sy: number;
    if (front) {
      sx = (_dirsD.x / -_dirsD.z) / (tf * aspect);
      sy = (-_dirsD.y / -_dirsD.z) / tf;
    } else {
      // behind the camera: the raw camera-space direction acts as a
      // screen-space pointer (y flipped to the screen convention)
      sx = _dirsD.x; sy = -_dirsD.y;
    }
    const dist = b.bodyAbs.distanceTo(absCam);
    return {
      ...b, front, sx, sy, l: Math.hypot(sx, sy), dist, occluded: false, labelOff: 21,
      rpx: (Math.tan(Math.asin(Math.min(1, b.bodyR / dist))) / tf) * (H / 2),
      px0: cx + (sx * W) / 2, py0: cy + (sy * H) / 2,
    };
  });
  // pass 2: place each chevron
  _dirsPlaced.length = 0;
  for (const b of bodies) {
    const other = bodies[0] === b ? bodies[1] : bodies[0];
    // a body hidden BEHIND the other body's disc (collinear view) must not
    // hug there: its chevron would mark the wrong sphere — send it to the
    // border instead so only the visible body gets the hug
    const occluded = b.front && other.front && other.dist < b.dist
      && Math.hypot(b.px0 - other.px0, b.py0 - other.py0) < other.rpx + 12;
    let px: number, py: number;
    let atBody = false; // body visible: chevron hugs its disc, pointing at it
    let flipped = false;
    if (b.front && !occluded && Math.abs(b.sx) < 0.95 && Math.abs(b.sy) < 0.9 && b.rpx < 110) {
      // body IS on screen (center inside the frustum, apparent disc small
      // enough): park the chevron just outside its apparent radius along
      // the center->body direction, rotated to point AT the body center —
      // it marks the body without covering it, exactly where it is.
      // Near the view center the radial direction is ill-defined, so the
      // chevron sits straight above the disc instead (stable, still hugs).
      const nx = b.l > 0.15 ? b.sx / b.l : 0;
      const ny = b.l > 0.15 ? b.sy / b.l : -1;
      let px2 = cx + (b.sx * W) / 2 + nx * (b.rpx + 26);
      let py2 = cy + (b.sy * H) / 2 + ny * (b.rpx + 26);
      // if the hug spot falls off-screen or on the help bar, hug from the
      // OPPOSITE side of the disc instead (always on-screen for a visible
      // disc) — never drag the chevron away from the body
      if (py2 > H - 150 || py2 < 36 || px2 < 44 || px2 > W - 44) {
        px2 = cx + (b.sx * W) / 2 - nx * (b.rpx + 26);
        py2 = cy + (b.sy * H) / 2 - ny * (b.rpx + 26);
        flipped = true;
      }
      b.sx = nx; b.sy = ny;
      atBody = true;
      px = Math.min(Math.max(px2, 30), W - 30);
      py = Math.min(Math.max(py2, 30), H - 30);
      // keep the two arrows from stacking on each other
      for (const p of placed) {
        const d = Math.hypot(px - p.x, py - p.y);
        if (d < 110) px += (px >= p.x ? 1 : -1) * (110 - d);
      }
      px = Math.min(Math.max(px, 30), W - 30);
      placed.push({ x: px, y: py });
      // label on the side of the chevron that faces AWAY from the disc
      b.labelOff = Math.abs(ny) >= 0.3 ? (flipped ? -ny : ny) * 12 : -12;
    } else if (!b.front && b.l < 0.15) {
      // dead behind: fixed top-center slot pointing down
      px = cx; py = 84; b.sx = 0; b.sy = 1;
      b.labelOff = 21;
    } else {
      // off-screen / behind / disc fills the view / occluded by the other
      // body: clamp to a border rectangle around the screen center along
      // the shortest on-screen direction — the arrow never disappears
      // M11w20e: when the body's center IS on screen (huge visible disc,
      // e.g. the whole-Earth tour stop) the clamped chevron must point
      // BACK at the body's center — the outward bearing would point away
      // from the disc it is standing on. Off-screen bodies keep the
      // outward bearing (shortest-way semantics).
      const onScreen = b.front && !occluded
        && Math.abs(b.sx) < 0.95 && Math.abs(b.sy) < 0.9;
      const nx = onScreen && b.l < 0.15 ? 0 : b.sx / b.l;
      const ny = onScreen && b.l < 0.15 ? -1 : b.sy / b.l;
      const t = Math.min((W / 2 - 56) / Math.max(Math.abs(nx), 1e-6),
        (H / 2 - 100) / Math.max(Math.abs(ny), 1e-6));
      px = cx + nx * t;
      py = cy + ny * t;
      b.sx = nx; b.sy = ny;
      // an occluded body's border arrow must not touch the occluder's
      // disc (it would look like it marks that sphere): slide it outward
      // along the same bearing until it clears the disc
      if (occluded) {
        const ddx = px - other.px0, ddy = py - other.py0;
        const dd = Math.hypot(ddx, ddy);
        if (dd < other.rpx + 20) {
          px = other.px0 + (ddx / (dd || 1)) * (other.rpx + 20);
          py = other.py0 + (ddy / (dd || 1)) * (other.rpx + 20);
        }
      }
      // dodge the fixed panels: nav map (top-right), debug HUD (top-left)
      if (px > W - 240 && py < 240) px = W - 240;
      if (px < 540 && py < 290) px = 540;
      // keep the two arrows from stacking on each other near the border
      for (const p of placed) {
        const d = Math.hypot(px - p.x, py - p.y);
        if (d < 110) px += (px >= p.x ? 1 : -1) * (110 - d);
      }
      px = Math.min(Math.max(px, 30), W - 30);
      placed.push({ x: px, y: py });
      b.labelOff = 21;
      atBody = onScreen; // flip the chevron to point at the visible disc
    }
    b.occluded = occluded;
    _dirsPlaced.push({ ...b, px, py });
    // screen-space bearing (y grows down); when hugging a visible body the
    // chevron is flipped to point back at the body's center (from whichever
    // side of the disc it hugs)
    const ang = Math.atan2(b.sy, b.sx) + (atBody && !flipped ? Math.PI : 0);
    ctx.save();
    // dark halo so the chevron stays readable against a bright planet limb
    ctx.shadowColor = 'rgba(0,0,0,0.9)';
    ctx.shadowBlur = 4;
    ctx.translate(px, py);
    ctx.rotate(ang);
    ctx.fillStyle = b.color;
    ctx.beginPath();
    ctx.moveTo(10, 0); ctx.lineTo(-6, 6); ctx.lineTo(-2, 0); ctx.lineTo(-6, -6);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    ctx.fillStyle = b.color;
    ctx.font = '10px ui-monospace,Consolas,monospace';
    ctx.textAlign = 'center';
    ctx.shadowColor = 'rgba(0,0,0,0.9)';
    ctx.shadowBlur = 4;
    ctx.fillText(b.label, px, py + (b.labelOff ?? (atBody ? -12 : 21)));
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
  }
  if (urlParams.has('dirdbg')) {
    // temporary: expose the arrow math for the moon vs the render camera
    _dirsD.copy(MOON.center).sub(absCam).normalize().applyQuaternion(_dirsQ);
    const mx = (_dirsD.x / -_dirsD.z) / (tf * aspect);
    const my = (-_dirsD.y / -_dirsD.z) / tf;
    (_dirsDbg ?? (_dirsDbg = []))[0] = `dirdbg moon d=(${_dirsD.x.toFixed(3)},${_dirsD.y.toFixed(3)},${_dirsD.z.toFixed(3)}) ndc=(${mx.toFixed(3)},${my.toFixed(3)}) camQ=(${rig.camera.quaternion.x.toFixed(3)},${rig.camera.quaternion.y.toFixed(3)},${rig.camera.quaternion.z.toFixed(3)},${rig.camera.quaternion.w.toFixed(3)}) absCam=(${absCam.x.toFixed(0)},${absCam.y.toFixed(0)},${absCam.z.toFixed(0)}) moonAbs=(${MOON.center.x.toFixed(0)},${MOON.center.y.toFixed(0)},${MOON.center.z.toFixed(0)})`;
    (_dirsDbg ?? (_dirsDbg = []))[1] = `dirdbg placed ${_dirsPlaced.map((p) => `${p.label}@(${p.px.toFixed(0)},${p.py.toFixed(0)}) proj=(${p.px0.toFixed(0)},${p.py0.toFixed(0)}) s=(${p.sx.toFixed(3)},${p.sy.toFixed(3)}) rpx=${p.rpx.toFixed(0)} occl=${p.occluded}`).join(' | ')}`;
  }
}

let _dirsDbg: string[] | null = null;
type DirPlaced = {
  label: string; px: number; py: number; px0: number; py0: number;
  sx: number; sy: number; rpx: number; occluded: boolean;
};
let _dirsPlaced: DirPlaced[] = [];
export function dirsDebugLines(): string[] | null { return _dirsDbg; }

function drawRadar(dt: number): void {
  if (!mapRenderer || !mapScene || !mapCam || !mapDiv || !mapShip || !mapEarth ||
      !mapMoon || !mapRing || !mapShipRay || !radarCtx || !radarCanvas || !radarEnabled) return;
  const alt = nearBody === 'moon'
    ? absCam.distanceTo(MOON.center) - R_MOON
    : absCam.length() - R;
  const show = alt > 40000;
  mapDiv.style.display = show ? 'block' : 'none';
  drawDirs(show);
  if (!show) return;
  // 10 Hz: re-zoom around the SHIP (the map is ship-centered)
  radarAcc += dt;
  if (radarAcc >= 0.1) {
    radarAcc = 0;
    const dE = absCam.length();
    const dM = absCam.distanceTo(MOON.center);
    // zoom keys to the farther body (held at the panel rim): near a body
    // its dot starts at the center under you and slides outward as you
    // leave, while the other body stays visible at the far edge
    const rmax = THREE.MathUtils.clamp(0.85 * Math.max(dE, dM), 3e7, 3.5e8);
    mapUT = 1 / rmax;
  }
  // smooth the zoom/rotation so warp bursts don't jump-cut the map
  const kf = 1 - Math.exp(-dt * 4);
  mapU += (mapUT - mapU) * kf;
  const place = (obj: THREE.Object3D, abs: THREE.Vector3) =>
    obj.position.copy(abs).sub(absCam).multiplyScalar(mapU);
  mapShip.position.set(0, 0, 0); // ship always at the map center
  place(mapEarth, _radarEarth);
  place(mapMoon, MOON.center);
  mapEarth.scale.setScalar(Math.min(R * 15 * mapU, 0.45));  // radii x15, capped
  mapMoon.scale.setScalar(Math.min(R_MOON * 15 * mapU, 0.45));
  mapRing.scale.setScalar(MOON_ORBIT_R * mapU);
  mapRing.position.copy(mapEarth.position);
  const marker = mapShip.children[0];
  if (marker) marker.scale.setScalar(0.05);   // map units: framed sphere is r=1
  mapShipRay.scale.setScalar(0.25);           // forward ray, map units
  mapShip.quaternion.copy(rig.camera.quaternion); // ray = camera forward
  // top-down: the mini camera sits directly above the ship looking down on
  // the world XZ plane; screen up = camera forward projected on that plane,
  // so the map rotates as you turn ("what you face" is always up)
  _mapF.set(0, 0, -1).applyQuaternion(rig.camera.quaternion);
  if (Math.hypot(_mapF.x, _mapF.z) > 0.05) {
    _mapUp.set(_mapF.x, 0, _mapF.z).normalize();
  }
  mapUp.lerp(_mapUp, kf).normalize();
  mapCam.position.set(0, 2.9, 0);
  mapCam.up.copy(mapUp);
  mapCam.lookAt(0, 0, 0);
  mapRenderer.render(mapScene, mapCam);
  // label overlay: fixed slots + leader lines to the projected positions
  const W = radarCanvas.width, H = radarCanvas.height;
  const ctx = radarCtx;
  ctx.clearRect(0, 0, W, H);
  const proj = (abs: THREE.Vector3): [number, number] => {
    _mapV.copy(abs).sub(absCam).multiplyScalar(mapU).project(mapCam);
    return [(_mapV.x * 0.5 + 0.5) * W, (-_mapV.y * 0.5 + 0.5) * H];
  };
  const [ex, ey] = proj(_radarEarth);
  const [mx, my] = proj(MOON.center);
  const distE = Math.max(absCam.length() - R, 0);
  const distM = Math.max(absCam.distanceTo(MOON.center) - R_MOON, 0);
  ctx.font = '9px ui-monospace,Consolas,monospace';
  const slot = (lx: number, ly: number, px: number, py: number, color: string, label: string, dist: string) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(30, ly - 3);
    ctx.lineTo(px, py);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.textAlign = 'left';
    ctx.fillText(label, 6, ly);
    ctx.fillText(dist, 6, ly + 9);
  };
  slot(6, 14, ex, ey, '#6fb5ff', 'Earth', fmtDist(distE));
  slot(6, H - 22, mx, my, '#d0d0dc', 'Moon', fmtDist(distM));
  ctx.fillStyle = '#ffd27f';
  ctx.fillText('YOU', W / 2 + 6, H / 2 - 6);
}

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
const SIM_VERSION = 'sim v11.9w20-tour11';

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

/** Switch free-camera <-> aircraft (F key). M11w20e: F during the tour
 * aborts the scripted tour and hands over the free camera at the current
 * pose (no aircraft respawn — the user keeps the tour location). */
function toggleFlight(): void {
  if (rig.stickMode) {
    // exit to free camera at the current pose
    rig.stickMode = false;
    rig.ctl.pitch = rig.ctl.roll = rig.ctl.yaw = 0;
    auto.resume();
  } else if (auto.exitTourIfRunning()) {
    // tour aborted: stay in the free camera right here
    rig.ctl.pitch = rig.ctl.roll = rig.ctl.yaw = 0;
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
    rig.autoLevelFrozen = auto.active; // the autopilot owns the pose — no fight
    rig.update(dt);
    autoLine = auto.update(rig, dt);
    world.abs(rig.camera.position, absCam);
    if (CAMLOG) {
      const log = (window as any).__camLog as Record<string, unknown>[];
      const cq = rig.camera.quaternion;
      log.push({
        t: performance.now(),
        x: absCam.x, y: absCam.y, z: absCam.z,
        qx: cq.x, qy: cq.y, qz: cq.z, qw: cq.w,
        s: autoLine ?? '',
      });
      if (log.length > 60000) log.shift();
    }
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

  // M11w20i: the moon's absolute position MUST be finalized BEFORE the LOD
  // updates below place tiles — a tile mesh bakes node.center + bodyCenter -
  // origin at show/reposition time. This block used to run AFTER
  // moonView.update, so while the moon was moving (tour orbit sweep, live
  // clock, ?moonangle) the tiles were placed at LAST frame's moon center
  // while the fallback and the HUD arrows used the current one — two moon
  // images offset by one frame of the moon's motion (86-1400 km/frame during
  // the sweep = a 3-5 px "double moon", reported during the fast crossing).
  // Moon orbit: the absolute position is written into the MOON frame center
  // (M9.6: the registry entry is the single source of truth — PlanetView,
  // flight physics and the nearest-body rule all read this one vector).
  // ?moonangle=<deg> (test hook) freezes the orbit at a fixed angle.
  // M11j: the moonshot demo OWNS the moon's position (placed at the TLI
  // antipode by the flight model at spawn) — the live clock must not
  // overwrite it or the transfer misses the moon by the drift.
  if (auto.moonAngle !== null) {
    moonPositionAtAngle(auto.moonAngle, MOON.center);
  } else if (flight.moonshotActive || flight.returnMission) {
    // M11j/M11w: the moonshot AND the return mission own the moon's
    // placement (the flight model freezes it at spawn — return uses the
    // anti-solar phase so the splashdown antipode is in daylight); the
    // live clock must not overwrite it or the parked craft drifts off
    // the rendered moon.
    MOON.center.copy(flight.moonCenter);
  } else {
    moonPosition(performance.now() / 1000 - t0Sim, MOON.center);
  }
  moonView.bodyCenter.copy(MOON.center);
  // M11w13: root stays at (0,0,0) — the tile meshes already carry the
  // absolute->frame map (node.center + bodyCenter - origin) individually,
  // so translating root as well double-counted the offset (the ghost
  // second moon). The fallback (scene child) is placed here instead.
  moonFallback.position.copy(MOON.center).sub(world.origin);
  if (urlParams.get('moon') === '0') moonFallback.visible = false;
  if (urlParams.get('moonfb') === '0') moonFallback.visible = false; // M11w13 diagnosis

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
  // M11w7: camera altitude for the terrain/sea cloud-shadow LOD key — the
  // shadow's sys field is octave-gated on camAlt/1000 exactly like the
  // march's (ro = uCamPos + uOrigin in the cloud shader, so the same sum
  // here; hypot avoids a per-frame vector allocation)
  {
    const p = rig.camera.position, o = world.origin;
    const camAltM = Math.hypot(p.x + o.x, p.y + o.y, p.z + o.z) - R;
    material.uniforms.uCamAlt.value = camAltM;
    seaMaterial.uniforms.uCamAlt.value = camAltM;
  }
  // ?wt=SECONDS: freeze the weather-clock for deterministic cloud A/B
  // (the field drifts with wall clock; tests must pin it to compare
  // march vs far shell at the same weather phase).
  {
    const wt = urlParams.get('wt');
    cloudUniforms.uTime.value = wt !== null ? parseFloat(wt) : performance.now() / 1000;
  }
  cloudUniforms.uTanHalfFov.value = Math.tan(THREE.MathUtils.degToRad(rig.camera.fov) * 0.5);
  cloudUniforms.uViewportH.value = window.innerHeight;
  // (M11w20i: the moon-orbit block moved ABOVE the planet/sea/moonView
  // updates — see the comment there — so tiles never lag the fallback.)
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
  drawRadar(dt); // M11w16 ship-centered 3D map + HUD bearing arrows

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
  const dbg = dirsDebugLines();
  hud.update([
    ...errors.slice(-3),
    SIM_VERSION,
    ...(autoLine ? [autoLine] : []),
    ...(rig.stickMode ? [flight.statusLine()] : []),
    ...flight.missionSummary(),
    ...(speedup > 1 ? [`speedup x${speedup}`] : []),
    ...(rebased > 0 ? [`rebase x${rebased} (total ${world.rebaseCount})`] : []),
    ...(probe ? [`probe ${probe()}`] : []),
    ...(dbg ?? []),
    `alt ${fmtDist(alt)} (${nearBody})  speed ${fmtDist(rig.stickMode && !flight.frozen ? flight.tGs : rig.currentSpeed)}/s  x${rig.speedMultiplier}`,
    `moon dist ${fmtDist(moonDist)}  tiles ${moonView.stats.visibleTiles} L${moonView.stats.maxVisibleLevel}`,
    `pitch ${look.pitch.toFixed(1)}°  bank ${look.bank.toFixed(1)}°  hdg ${look.heading.toFixed(0)}°  level ${rig.autoLevel ? 'on(R)' : 'off(R)'}`,
    `revz ${(renderer as unknown as { capabilities: { reverseDepthBuffer: boolean } }).capabilities.reverseDepthBuffer ? 'ON' : 'off'}  aa ${urlParams.get('aa') === '0' ? 'off' : 'on'}`,
    `dbg cam=(${rig.camera.position.x.toFixed(0)},${rig.camera.position.y.toFixed(0)},${rig.camera.position.z.toFixed(0)}) org=(${world.origin.x.toFixed(0)},${world.origin.y.toFixed(0)},${world.origin.z.toFixed(0)})`,
    `tiles ${s.visibleTiles}  tris ${(s.triangles / 1000).toFixed(1)}k  maxLvl ${s.maxVisibleLevel}`,
    ...(urlParams.has('loddbg') ? [`perL ${(s.perLevel ?? []).map((v, i) => v ? `${i}:${v}` : null).filter(Boolean).join(' ')} fwd(${(() => { const f = new THREE.Vector3(); rig.camera.getWorldDirection(f); return `${f.x.toFixed(2)},${f.y.toFixed(2)},${f.z.toFixed(2)}`; })()}) cam(${rig.camera.position.x.toFixed(0)},${rig.camera.position.y.toFixed(0)},${rig.camera.position.z.toFixed(0)}) near${rig.camera.near.toFixed(1)} far${rig.camera.far}`, `moonNdc ${((_moonNdc.copy(MOON.center).sub(world.origin).project(rig.camera))) ? `${_moonNdc.x.toFixed(3)},${_moonNdc.y.toFixed(3)}` : 'n/a'}`] : []),
    `sea tiles ${sea.stats.visibleTiles}  tris ${(sea.stats.triangles / 1000).toFixed(1)}k  maxLvl ${sea.stats.maxVisibleLevel}`,
    `queue ${s.pending}  cache ${s.cached}  built ${s.built} (wk ${s.workerBuilt})  evicted ${s.evicted}  hits ${s.cacheHits}  veg ${planet.vegTiles}`,
    `drawcalls ${renderer.info.render.calls}`,
    rig.mouseLocked ? 'mouse locked' : 'click = mouse look (arrows also work)',
  ]);
});
