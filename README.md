# SEAMOON

**Seamless flight from a beach on Earth to the surface of the Moon — one continuous ride, at true scale.**

[日本語版 README はこちら](README.ja.md)

Seamoon is a WebGL space flight simulator built around one idea: the whole
Earth–Moon system as a single, continuous space you can fly through. You start
standing on a sunlit beach, rise through the clouds, watch the whole Earth
shrink behind you, cross 384,000 km of cislunar space, and touch down on the
craters of the Moon — with no loading screens, no scene switches, and no
teleports. The camera never leaves the same world.

## Overview

- **True-scale solar system neighborhood.** Earth radius 6,371 km, Moon radius
  1,737 km, the real 384,400 km orbit. The camera flies from 2 m above the
  sand to 400,000 km out in one coordinate system.
- **One guided tour, zero UI drama.** `?demo=tour` flies the whole story for
  you: a 3-second countdown on the beach, a slow backward-drifting climb
  through the cloud deck, a graceful pan to the whole-Earth view with the Moon
  above the limb, a cinematic swing around the Moon, and a final landing with
  the Earth hanging in the sky.
- **Free flight when you take over.** Press F during the tour to grab the
  camera, or start in free-flight mode and fly the ship yourself — aircraft
  mode near the ground, Newtonian orbital mechanics in space, a lander for
  the Moon.
- **Everything procedural.** Terrain, craters, clouds, ocean, stars — no
  texture downloads, the whole world is generated from noise functions on
  load.

Seamoon was developed as a **test of local-LLM-driven software development**:
the entire codebase was written and iteratively debugged by
**GLM-5.3-Flash**, running locally on **two NVIDIA DGX Spark** machines that
operated autonomously for about one week. A human supervised the runs and
occasionally left bug reports and design feedback as comments; every line of
implementation, testing, and fixing was produced by the local model.

## Quick start

Prerequisites: **Node.js 18+** and a WebGL2-capable browser (Chrome/Edge
recommended).

```bash
git clone https://github.com/ahinore/seamoon.git
cd seamoon
npm install
npm run dev
```

Open **http://localhost:5173** and click the canvas to grab the mouse.

For a production build:

```bash
npm run build     # outputs to dist/
npm run preview   # serves the build locally
```

No GPU is required for development — any machine that runs a modern browser
with WebGL2 works. The renderer uses a reversed-float depth buffer for the
2 m → 2,000,000 km depth range.

## Controls

**Free flight (default mode)**

| Key | Action |
|---|---|
| Click | Capture / release mouse look |
| `W` `A` `S` `D` | Move |
| `Q` / `E` | Down / up (along the planet's zenith direction) |
| `Shift` | Boost |
| `1`–`5` | Speed multiplier |
| `G` | Wireframe toggle |
| `H` | Return to initial position |
| `R` | Auto-level toggle (stabilizes the horizon) |
| `Esc` | Release the mouse |

Speed auto-adapts to altitude: meters per second near the ground, kilometers
per second in space.

**Aircraft mode (`F`)** — near a planet's surface the rig switches to a plane
style model: mouse/arrows as the control stick, `W`/`S` as throttle, `A`/`D`
as the rudder. Press `F` again to return to free flight.

**Guided tour (`?demo=tour`)** — sit back and watch the full Earth→Moon
journey (about two and a half minutes). Pressing `F` at any hold point hands
the camera over to you.

**Moon landing demo (`?demo=lunar`)** — pilot the lander: `W`/`S` throttle,
`A`/`D` rudder, mouse/arrows to tilt. Touch down softly on the HUD `AGL 0 m`.

## Demo modes (URL parameters)

| URL | What it does |
|---|---|
| (none) | Free flight starting on the beach |
| `?demo=tour` | The full guided Earth→Moon cinematic tour |
| `?demo=hover&alt=200000` | Hover at a fixed altitude (meters) |
| `?demo=drop` | Automatic dive from 25,000 km to 100 m (LOD stress test) |
| `?demo=look` | Camera-rotation stress test through the zenith |
| `?demo=orbital` | Newtonian orbital flight around Earth (`&moonshot=1` adds a full TLI→landing mission) |
| `?demo=lunar` | Moon lander |
| `?lat=..&lon=..&alt=..` | Custom spawn point |

## How it works

The central problem Seamoon solves is **rendering a 6,371 km planet and a
2 m rock from the same camera without clipping or jitter**. The building
blocks:

**Double-precision floating origin.** All positions live in JavaScript
`float64` absolute meters, referenced to the Earth's center. Each frame the
origin is re-centered onto the camera (`src/world.ts`), so the GPU only ever
sees small float32 frame-relative values — no vertex jitter at any scale.

**Cube-sphere quadtree LOD (`src/cubeSphereLod.ts`).** Each body is a cube
projected onto a sphere, split by screen-space error (ρ > 2 px splits,
< 0.2 px merges) down to level 15 — about 2 m tiles. Tiles carry skirts to
hide cracks, an LRU geometry cache keeps rebuilds near zero, and a
frame-budgeted build queue (plus a web worker, `src/tileWorker.ts`) keeps
tile construction off the main thread.

**Custom shaders everywhere (`src/materials.ts`, `src/clouds.ts`,
`src/seaGeometry.ts`, `src/atmosphere.ts`, `src/moonMaterial.ts`).** The
atmosphere, cloud deck (with a raymarched shadow pass), animated ocean,
terrain lighting, and lunar surface are all hand-written GLSL tuned for the
reversed-float depth layout. Vegetation (`src/scatter.ts`,
`src/vegMaterial.ts`) is GPU-instanced per tile.

**One physics sandbox (`src/flight.ts`, `src/orbit.ts`,
`src/moonOrbit.ts`).** Near the surface you fly an arcade aircraft model; in
space the same craft follows real Keplerian orbital mechanics (patched
conics), including a scripted TLI→translunar→capture→landing mission. The
Moon orbits on its real 384,400 km, 5.14°-inclination path and both bodies
share one absolute-frame registry (`src/frames.ts`).

**Quaternion-only camera (`src/cameraRig.ts`).** All rotation is a single
quaternion applied in view space, so the camera can pitch through the zenith
without gimbal problems; an auto-level assist gently relaxes roll toward the
local horizon.

**Self-verifying autopilot (`src/testAuto.ts`).** The guided tour and the
demo modes are driven by an autopilot that doubles as a test harness — it
reports its state on the HUD and pairs with pixel-probe URL hooks, which is
how the LLM agent regression-tested its own work during the one-week
autonomous run.

| Module | Role |
|---|---|
| `src/main.ts` | Bootstrap, frame loop, HUD, nav map, body arrows |
| `src/world.ts` | Floating origin / rebase |
| `src/frames.ts` | Body frames, lat/lon ↔ absolute helpers |
| `src/cubeSphereLod.ts` | Quadtree LOD core |
| `src/tile*.ts` | Tile geometry, mesh pool, web worker |
| `src/terrain.ts` / `src/noise.ts` | Procedural Earth terrain |
| `src/moon*.ts` | Procedural Moon (craters, material, orbit) |
| `src/clouds.ts` / `src/seaGeometry.ts` / `src/atmosphere.ts` | Weather, ocean, sky |
| `src/scatter.ts` / `src/vegMaterial.ts` | Instanced vegetation |
| `src/cameraRig.ts` | Camera, auto-level |
| `src/flight.ts` / `src/orbit.ts` | Flight models & Keplerian mechanics |
| `src/testAuto.ts` | Autopilot, guided tour, probes |
| `src/hud.ts` / `src/audio.ts` | HUD & sound |

## License

[MIT](LICENSE)
