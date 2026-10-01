export class Hud {
  private el: HTMLElement;
  private acc = 0;
  private n = 0;
  private fps = 0;
  // M11h: throttle DOM writes — textContent assignment forces layout,
  // and doing it every frame for an always-visible overlay adds up
  private pending: string[] | null = null;
  private sinceWrite = 0;
  // M11k: transient note line (e.g. "wire ON") — visible for 3 s so
  // toggle keys give feedback without permanently cluttering the HUD.
  private note: string | null = null;
  private noteUntil = 0;

  constructor(id: string) {
    this.el = document.getElementById(id)!;
  }

  frame(dt: number): void {
    this.acc += dt;
    this.n++;
    if (this.acc >= 0.5) {
      this.fps = this.n / this.acc;
      this.acc = 0;
      this.n = 0;
    }
    // flush pending lines at ~10 Hz
    if (this.pending && this.sinceWrite >= 0.1) {
      const lines = this.note && performance.now() < this.noteUntil
        ? ['[' + this.note + ']', ...this.pending]
        : this.pending;
      this.el.textContent = lines.join('\n');
      this.pending = null;
      this.sinceWrite = 0;
    }
    this.sinceWrite += dt;
  }

  update(lines: string[]): void {
    this.pending = [`fps ${this.fps.toFixed(0)}`, ...lines];
  }

  /** M11k: show a transient note line for ~3 s (toggle feedback). */
  setNote(text: string): void {
    this.note = text;
    this.noteUntil = performance.now() + 3000;
    // force an immediate flush so the note appears on the next frame
    this.pending = this.pending ?? [`fps ${this.fps.toFixed(0)}`];
    this.sinceWrite = 1;
  }
}
