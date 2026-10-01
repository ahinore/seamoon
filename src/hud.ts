export class Hud {
  private el: HTMLElement;
  private acc = 0;
  private n = 0;
  private fps = 0;
  // M11h: throttle DOM writes — textContent assignment forces layout,
  // and doing it every frame for an always-visible overlay adds up
  private pending: string[] | null = null;
  private sinceWrite = 0;

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
      this.el.textContent = this.pending.join('\n');
      this.pending = null;
      this.sinceWrite = 0;
    }
    this.sinceWrite += dt;
  }

  update(lines: string[]): void {
    this.pending = [`fps ${this.fps.toFixed(0)}`, ...lines];
  }
}
