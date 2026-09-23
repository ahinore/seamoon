import * as THREE from 'three';
import { CameraRig } from './cameraRig';
import { PlanetView } from './cubeSphereLod';
import { makePlanetMaterial, makeStars } from './materials';
import { Hud } from './hud';
import { AutoPilot } from './testAuto';

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

const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const material = makePlanetMaterial();
const planet = new PlanetView(scene, R, material, {
  maxLevel: 20,
  tauPx: 2,
  res: 65,
  cacheSize: 300,
});
scene.add(makeStars(2500, 6e8));

const hud = new Hud('hud');
let rigRef: CameraRig | null = null;
const rig = new CameraRig(
  renderer.domElement,
  new THREE.Vector3(0, 0, R * 4),
  new THREE.Vector3(0, 0, 0),
  () => (rigRef ? rigRef.camera.position.length() - R : Number.POSITIVE_INFINITY),
);
rigRef = rig;

window.addEventListener('keydown', (e) => {
  if (e.code === 'KeyG') {
    material.uniforms.uWire.value = material.uniforms.uWire.value > 0.5 ? 0 : 1;
  }
});

// test hook: force wireframe from URL for automated runs
if (new URLSearchParams(location.search).get('wire') === '1') {
  material.uniforms.uWire.value = 1;
}

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  rig.camera.aspect = window.innerWidth / window.innerHeight;
  rig.camera.updateProjectionMatrix();
});

const fmtDist = (m: number): string =>
  m >= 1e6 ? (m / 1e6).toFixed(2) + ' Mm' : m >= 1e4 ? (m / 1e3).toFixed(1) + ' km' : m.toFixed(1) + ' m';

const auto = new AutoPilot(rig);

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
      const show = preset === 'g' ? `g anomalies=${anom}/${rgb.length}` : `rgb=${rgb.map((c) => c.join(',')).join(' | ')} planet=${planetCount}/${rgb.length}`;
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
  for (let i = 0; i < speedup; i++) {
    rig.update(dt);
    autoLine = auto.update(rig, dt);
    planet.update(rig.camera, window.innerHeight);
  }
  renderer.render(scene, rig.camera);
  hud.frame(dt);

  const s = planet.stats;
  const alt = Math.max(rig.camera.position.length() - R, 0);
  const look = rig.getLookAngles();
  hud.update([
    ...errors.slice(-3),
    ...(autoLine ? [autoLine] : []),
    ...(speedup > 1 ? [`speedup x${speedup}`] : []),
    ...(probe ? [`probe ${probe()}`] : []),
    `alt ${fmtDist(alt)}  speed ${fmtDist(rig.currentSpeed)}/s  x${rig.speedMultiplier}`,
    `pitch ${look.pitch.toFixed(1)}°  bank ${look.bank.toFixed(1)}°  hdg ${look.heading.toFixed(0)}°  level ${rig.autoLevel ? 'on(R)' : 'off(R)'}`,
    `tiles ${s.visibleTiles}  tris ${(s.triangles / 1000).toFixed(1)}k  maxLvl ${s.maxVisibleLevel}`,
    `queue ${s.pending}  cache ${s.cached}  built ${s.built}  evicted ${s.evicted}  hits ${s.cacheHits}`,
    `drawcalls ${renderer.info.render.calls}`,
    rig.mouseLocked ? 'mouse locked' : 'click = mouse look (arrows also work)',
  ]);
});
