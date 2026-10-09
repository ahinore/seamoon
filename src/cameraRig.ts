import * as THREE from 'three';

const clamp = (x: number, a: number, b: number) => Math.min(Math.max(x, a), b);

/**
 * Free-flight camera rig with quaternion attitude.
 *
 * Orientation is a single quaternion (no Euler yaw/pitch): turning is applied
 * in the camera's LOCAL frame — yaw around your own head-up axis, pitch around
 * your own right axis — so looking around works identically at the nadir, at
 * the poles, and in deep space (no gimbal degeneration, pitch is unclamped).
 *
 * Side effect of local-frame turning: combined yaw+pitch accumulates an
 * apparent roll relative to the horizon. When autoLevel is on (R toggles),
 * each frame slerps the roll component out against the local zenith
 * (radial direction), preserving the forward direction exactly — the horizon
 * stays level while looking direction stays fully free.
 */
export class CameraRig {
  readonly camera: THREE.PerspectiveCamera;
  mouseLocked = false;
  currentSpeed = 0;
  /**
   * M11n3: altitude of the NEAR CLOUD HULL above the surface (earth only,
   * Infinity elsewhere). The rig's altitude is SEA-LEVEL (terrain included),
   * so hovering over high terrain put the hull surface (2.6 km - sea-level
   * altitude) BEHIND the near plane — the whole near-hull cloud layer
   * clipped away and "no clouds rendered above the horizon when inside or
   * below the deck". near is capped at 40% of the hull clearance.
   */
  nearCapAlt = Infinity;
  speedMultiplier = 1;
  autoLevel = true;
  /** Flight-mode stick deflections (-1..1, spring-centered). FlightModel reads. */
  readonly ctl = { pitch: 0, roll: 0, yaw: 0 };
  /** When true, update() drives the flight stick instead of the free camera. */
  stickMode = false;
  /** M11w20f: the scripted autopilot owns the pose while this is true —
   * auto-level must not fight it. Exiting the nadir skip cone with a scripted
   * orientation up to a half-roll away from level read as a sudden clunk
   * (LEVEL_RATE yanks it back at ~3 rad/s mid-pan). main.ts sets this from
   * AutoPilot.active every frame. */
  autoLevelFrozen = false;
  /** Max auto-level correction speed, rad/s (a 180° half-roll takes ~1 s). */
  private readonly LEVEL_RATE = 3;

  private boost = false;
  private pendingYaw = 0;
  private pendingPitch = 0;
  /** Unconsumed mouse deltas feeding the flight stick. */
  private stickDx = 0;
  private stickDy = 0;
  private readonly keys = new Set<string>();
  private readonly el: HTMLElement;
  private readonly getAltitude: () => number;
  private readonly getUp: () => THREE.Vector3;

  private readonly tmpQ = new THREE.Quaternion();
  private readonly levelQ = new THREE.Quaternion();
  private readonly basisM = new THREE.Matrix4();
  private readonly AXIS_X = new THREE.Vector3(1, 0, 0);
  private readonly AXIS_Y = new THREE.Vector3(0, 1, 0);
  private readonly dir = new THREE.Vector3();
  private readonly fwd = new THREE.Vector3();
  private readonly right = new THREE.Vector3();
  private readonly zenith = new THREE.Vector3();
  private readonly rightH = new THREE.Vector3();
  private readonly upH = new THREE.Vector3();
  private readonly back = new THREE.Vector3();
  private readonly camUp = new THREE.Vector3();
  private readonly rollAxis = new THREE.Vector3();
  private readonly eastV = new THREE.Vector3();
  private readonly northV = new THREE.Vector3();
  private readonly WORLD_Y = new THREE.Vector3(0, 1, 0);

  constructor(
    el: HTMLElement,
    position: THREE.Vector3,
    lookAt: THREE.Vector3,
    getAltitude: () => number,
    getUp: () => THREE.Vector3,
  ) {
    this.el = el;
    this.getAltitude = getAltitude;
    this.getUp = getUp;
    this.camera = new THREE.PerspectiveCamera(60, 1, 1, 2e9);
    this.camera.position.copy(position);
    this.camera.lookAt(lookAt);

    el.addEventListener('click', () => {
      if (!this.mouseLocked) this.el.requestPointerLock?.();
    });
    document.addEventListener('pointerlockchange', () => {
      this.mouseLocked = document.pointerLockElement === this.el;
    });
    document.addEventListener('mousemove', (ev) => {
      if (!this.mouseLocked) return;
      if (this.stickMode) {
        this.stickDx += ev.movementX;
        this.stickDy += ev.movementY;
        return;
      }
      this.pendingYaw -= ev.movementX * 0.0022;
      this.pendingPitch -= ev.movementY * 0.0022;
    });
    window.addEventListener('keydown', (ev) => {
      if (ev.repeat) return;
      this.keys.add(ev.code);
      if (ev.code.startsWith('Digit')) {
        const d = Number(ev.code.slice(5));
        if (d >= 1 && d <= 5) this.speedMultiplier = d;
      }
      if (ev.code === 'ShiftLeft' || ev.code === 'ShiftRight') this.boost = true;
      // R toggles auto-level in free-cam mode; in flight mode R respawns the
      // aircraft (handled by main.ts), so the toggle must not fire.
      if (ev.code === 'KeyR' && !this.stickMode) this.autoLevel = !this.autoLevel;
      if (ev.code.startsWith('Arrow')) ev.preventDefault();
    });
    window.addEventListener('keyup', (ev) => {
      this.keys.delete(ev.code);
      if (ev.code === 'ShiftLeft' || ev.code === 'ShiftRight') this.boost = false;
    });
  }

  /**
   * Pitch (fwd vs. local horizon plane, + toward zenith), bank (roll vs. the
   * leveled horizon), and heading (compass angle of fwd on the local tangent
   * plane, 0 = north, deg) — HUD/diagnostic readout.
   */
  getLookAngles(): { pitch: number; bank: number; heading: number } {
    const q = this.camera.quaternion;
    this.fwd.set(0, 0, -1).applyQuaternion(q);
    this.zenith.copy(this.getUp());
    const pitch = THREE.MathUtils.radToDeg(Math.asin(clamp(this.fwd.dot(this.zenith), -1, 1)));
    let bank = 0;
    let heading = 0;
    this.eastV.crossVectors(this.WORLD_Y, this.zenith);
    if (this.eastV.lengthSq() > 1e-8) {
      this.eastV.normalize();
      this.northV.crossVectors(this.zenith, this.eastV); // unit
      heading = THREE.MathUtils.radToDeg(
        Math.atan2(this.fwd.dot(this.eastV), this.fwd.dot(this.northV)),
      );
      if (heading < 0) heading += 360;
    }
    this.rightH.crossVectors(this.fwd, this.zenith);
    if (this.rightH.lengthSq() > 0.05) {
      this.rightH.normalize();
      this.upH.crossVectors(this.rightH, this.fwd).normalize();
      this.camUp.set(0, 1, 0).applyQuaternion(q);
      this.rollAxis.crossVectors(this.upH, this.camUp);
      bank = THREE.MathUtils.radToDeg(
        Math.atan2(this.rollAxis.dot(this.fwd), this.upH.dot(this.camUp)),
      );
    }
    return { pitch, bank, heading };
  }

  /**
   * Accumulate a view-relative rotation (radians). Feeds the same path as
   * mouse/arrow input; used by the test autopilot.
   */
  turn(yaw: number, pitch: number): void {
    this.pendingYaw += yaw;
    this.pendingPitch += pitch;
  }

  /** Drop accumulated (not yet applied) view-relative turns. */
  clearPendingTurn(): void {
    this.pendingYaw = 0;
    this.pendingPitch = 0;
  }

  /** Key-down test for the flight model (throttle/brake reads). */
  isDown(code: string): boolean {
    return this.keys.has(code);
  }

  update(dt: number): void {
    const k = this.keys;
    if (this.stickMode) {
      // Flight mode: mouse/arrows deflect the stick (spring-centered), no
      // camera motion here — FlightModel.step() owns the pose.
      const rate = 2.2 * dt;
      const d = (v: number, inc: boolean, dec: boolean) =>
        clamp(v + (inc ? rate : 0) - (dec ? rate : 0), -1, 1);
      if (this.mouseLocked) {
        this.ctl.pitch = clamp(this.ctl.pitch - this.stickDy * 0.06, -1, 1);
        this.ctl.roll = clamp(this.ctl.roll - this.stickDx * 0.06, -1, 1);
        this.stickDx = 0;
        this.stickDy = 0;
      }
      this.ctl.pitch = d(this.ctl.pitch, k.has('ArrowUp'), k.has('ArrowDown'));
      this.ctl.roll = d(this.ctl.roll, k.has('ArrowLeft'), k.has('ArrowRight'));
      this.ctl.yaw = d(this.ctl.yaw, k.has('KeyA'), k.has('KeyD'));
      this.currentSpeed = 0;
      return;
    }
    // Arrow keys feed the same accumulator as the mouse.
    const turn = 1.5 * dt;
    if (k.has('ArrowLeft')) this.pendingYaw += turn;
    if (k.has('ArrowRight')) this.pendingYaw -= turn;
    if (k.has('ArrowUp')) this.pendingPitch += turn;
    if (k.has('ArrowDown')) this.pendingPitch -= turn;

    // View-relative rotation: post-multiplying rotates in the camera LOCAL
    // frame — "left" is around your own head-up axis, "up" around your own
    // right axis. Degenerates nowhere (quaternion composition).
    const q = this.camera.quaternion;
    if (this.pendingYaw !== 0) {
      this.tmpQ.setFromAxisAngle(this.AXIS_Y, this.pendingYaw);
      q.multiply(this.tmpQ);
    }
    if (this.pendingPitch !== 0) {
      this.tmpQ.setFromAxisAngle(this.AXIS_X, this.pendingPitch);
      q.multiply(this.tmpQ);
    }
    this.pendingYaw = 0;
    this.pendingPitch = 0;
    q.normalize();

    // Auto-level: remove accumulated roll against the local zenith while
    // keeping the forward direction exactly fixed. Skipped when fwd is within
    // ~13° of the zenith/nadir: the leveled frame is near-degenerate there
    // (rightH -> 0) and the slerp would orbit the camera around the zenith
    // instead of letting it pass through. The zenith comes from a provider
    // because with a floating origin camera.position is frame-relative.
    //
    // Rate-limited: pitching THROUGH the zenith/nadir inherently leaves the
    // camera 180° inverted relative to the horizon, so the level target right
    // after exiting the skip cone can be a full half-roll away. Applying that
    // with a plain exponential slerp reads as a sudden "clunk" flip; capping
    // the correction speed turns it into a smooth deliberate roll while small
    // errors still settle quickly.
    this.fwd.set(0, 0, -1).applyQuaternion(q);
    this.zenith.copy(this.getUp()).normalize();
    const hasZenith = this.zenith.lengthSq() > 0.5;
    if (hasZenith && this.autoLevel && !this.autoLevelFrozen) {
      this.rightH.crossVectors(this.fwd, this.zenith);
      if (this.rightH.lengthSq() > 0.05) {
        this.rightH.normalize();
        this.upH.crossVectors(this.rightH, this.fwd); // unit, orthonormal
        this.back.copy(this.fwd).negate();
        this.basisM.makeBasis(this.rightH, this.upH, this.back);
        this.levelQ.setFromRotationMatrix(this.basisM);
        const err = 2 * Math.acos(clamp(Math.abs(q.dot(this.levelQ)), 0, 1));
        if (err > 1e-4) {
          const t = Math.min(1, (this.LEVEL_RATE * dt) / err);
          q.slerp(this.levelQ, t);
        }
      }
    }

    // Movement basis: forward/right from the (leveled) attitude, vertical
    // (Q/E) from the planet's radial direction — "up" means away from center.
    this.fwd.set(0, 0, -1).applyQuaternion(q);
    this.right.set(1, 0, 0).applyQuaternion(q);

    this.dir.set(0, 0, 0);
    if (k.has('KeyW')) this.dir.add(this.fwd);
    if (k.has('KeyS')) this.dir.sub(this.fwd);
    if (k.has('KeyD')) this.dir.add(this.right);
    if (k.has('KeyA')) this.dir.sub(this.right);
    if (hasZenith) {
      if (k.has('KeyE')) this.dir.add(this.zenith);
      if (k.has('KeyQ')) this.dir.sub(this.zenith);
    }

    const alt = Math.max(this.getAltitude(), 0);
    const speed = clamp(alt * 0.4, 15, 1e6) * this.speedMultiplier * (this.boost ? 8 : 1);
    this.currentSpeed = this.dir.lengthSq() > 0 ? speed : 0;
    if (this.dir.lengthSq() > 0) {
      this.camera.position.addScaledVector(this.dir.normalize(), speed * dt);
    }

    // M11n3: near must stay well INSIDE the near cloud hull while the hull
    // paints (the march lives below ~25 km altitude; the hull follows the
    // camera so its FOV-edge fragments sit ~1.8 km out and a larger near
    // plane would clip them). ABOVE the march band the hull paints nothing
    // (alpha 0), so the cap lifts and the altitude-adaptive near keeps the
    // old depth precision at altitude — a fixed 1 km near at orbital
    // distances degrades sea/terrain depth separation ~50x (z-fight risk
    // on the co-planar limb at spawn).
    const nearCap = alt < 26000 ? this.nearCapAlt * 0.4 : Infinity;
    const near = clamp(Math.min(alt * 0.2, nearCap), 0.5, 5e4);
    if (Math.abs(near - this.camera.near) / near > 0.3) {
      this.camera.near = near;
      this.camera.updateProjectionMatrix();
    }
  }
}
