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
  speedMultiplier = 1;
  autoLevel = true;

  private boost = false;
  private pendingYaw = 0;
  private pendingPitch = 0;
  private readonly keys = new Set<string>();
  private readonly el: HTMLElement;
  private readonly getAltitude: () => number;

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
  ) {
    this.el = el;
    this.getAltitude = getAltitude;
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
      if (ev.code === 'KeyR') this.autoLevel = !this.autoLevel;
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
    this.zenith.copy(this.camera.position).normalize();
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

  update(dt: number): void {
    const k = this.keys;
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
    // instead of letting it pass through.
    this.fwd.set(0, 0, -1).applyQuaternion(q);
    this.zenith.copy(this.camera.position);
    const r2 = this.zenith.lengthSq();
    if (r2 > 1e-6) {
      this.zenith.multiplyScalar(1 / Math.sqrt(r2));
      if (this.autoLevel) {
        this.rightH.crossVectors(this.fwd, this.zenith);
        if (this.rightH.lengthSq() > 0.05) {
          this.rightH.normalize();
          this.upH.crossVectors(this.rightH, this.fwd); // unit, orthonormal
          this.back.copy(this.fwd).negate();
          this.basisM.makeBasis(this.rightH, this.upH, this.back);
          this.levelQ.setFromRotationMatrix(this.basisM);
          q.slerp(this.levelQ, 1 - Math.exp(-6 * dt));
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
    if (r2 > 1e-6) {
      if (k.has('KeyE')) this.dir.add(this.zenith);
      if (k.has('KeyQ')) this.dir.sub(this.zenith);
    }

    const alt = Math.max(this.getAltitude(), 0);
    const speed = clamp(alt * 0.4, 15, 1e6) * this.speedMultiplier * (this.boost ? 8 : 1);
    this.currentSpeed = this.dir.lengthSq() > 0 ? speed : 0;
    if (this.dir.lengthSq() > 0) {
      this.camera.position.addScaledVector(this.dir.normalize(), speed * dt);
    }

    const near = clamp(alt * 0.2, 0.5, 5e4);
    if (Math.abs(near - this.camera.near) / near > 0.3) {
      this.camera.near = near;
      this.camera.updateProjectionMatrix();
    }
  }
}
