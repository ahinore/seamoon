export class Hud {
  private el: HTMLElement;
  private acc = 0;
  private n = 0;
  private fps = 0;

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
  }

  update(lines: string[]): void {
    this.el.textContent = [`fps ${this.fps.toFixed(0)}`, ...lines].join('\n');
  }
}
