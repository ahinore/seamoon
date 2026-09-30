/**
 * M10.9: procedural flight audio (no assets — pure WebAudio synthesis).
 *
 * Layers, all driven from the flight model each frame:
 * - booster: brown noise through a lowpass, gain follows booster thrust
 *   (lob boost phase) — deep launch rumble
 * - wind:    pink-ish noise bandpass, gain follows dynamic pressure
 *   (rho * v²) while in atmosphere — airflow rush
 * - plasma:  white noise through a bandpass sweep, gain follows the
 *   normalized entry heat — the reentry roar
 * - chute:   a short filtered crack when the parachute stages
 *
 * Browsers suspend AudioContext until a user gesture: the first
 * click/keydown resumes it. `?audio=0` disables the engine entirely.
 */

import type { FlightModel } from './flight';

export class FlightAudio {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private boosterGain: GainNode | null = null;
  private boosterFilter: BiquadFilterNode | null = null;
  private windGain: GainNode | null = null;
  private windFilter: BiquadFilterNode | null = null;
  private plasmaGain: GainNode | null = null;
  private plasmaFilter: BiquadFilterNode | null = null;
  private enabled: boolean;
  private armed = false;

  constructor(enabled: boolean) {
    this.enabled = enabled;
  }

  /** Lazily build the graph on the first user gesture. */
  private arm(): void {
    if (this.armed || !this.enabled) return;
    try {
      this.ctx = new AudioContext();
    } catch {
      this.enabled = false;
      return;
    }
    this.armed = true;
    const ctx = this.ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 0.8;
    this.master.connect(ctx.destination);

    // noise sources: a looping buffer of pre-generated noise per flavor
    const mk = (gen: (i: number, prev: number) => number) => {
      const buf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
      const d = buf.getChannelData(0);
      let prev = 0;
      for (let i = 0; i < d.length; i++) {
        prev = gen(i, prev);
        d[i] = prev;
      }
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.loop = true;
      src.start();
      return src;
    };
    // brown: integrated white
    const brown = mk((_i, p) => {
      const w = Math.random() * 2 - 1;
      return (p + 0.02 * w) / 1.02;
    });
    // pink-ish: Paul Kellet's cheap filter
    const pink = mk((_i, p) => {
      const w = Math.random() * 2 - 1;
      return 0.96 * p + 0.04 * w;
    });
    const white = mk(() => Math.random() * 2 - 1);

    const chain = (src: AudioBufferSourceNode, type: BiquadFilterType, freq: number) => {
      const f = ctx.createBiquadFilter();
      f.type = type;
      f.frequency.value = freq;
      const g = ctx.createGain();
      g.gain.value = 0;
      src.connect(f); f.connect(g); g.connect(this.master!);
      return { f, g };
    };
    const bo = chain(brown, 'lowpass', 220);
    this.boosterFilter = bo.f;
    this.boosterGain = bo.g;
    const wi = chain(pink, 'bandpass', 700);
    this.windFilter = wi.f;
    this.windGain = wi.g;
    const pl = chain(white, 'bandpass', 1200);
    this.plasmaFilter = pl.f;
    this.plasmaGain = pl.g;
  }

  /** Resume on user gesture (browsers require it for sound). */
  resume(): void {
    if (!this.enabled) return;
    this.arm();
    this.ctx?.resume();
  }

  /** One frame of layer updates. g = 0..1 intensities. */
  frame(booster: number, windQ: number, plasma: number): void {
    if (!this.ctx || !this.armed || this.ctx.state !== 'running') return;
    const t = this.ctx.currentTime;
    const set = (g: GainNode | null, v: number) =>
      g && g.gain.setTargetAtTime(Math.min(v, 1), t, 0.08);
    set(this.boosterGain, booster * 0.9);
    set(this.windGain, windQ * 0.5);
    set(this.plasmaGain, plasma * 0.7);
    // filters breathe with intensity
    if (this.windFilter) this.windFilter.frequency.setTargetAtTime(400 + windQ * 1400, t, 0.1);
    if (this.plasmaFilter) this.plasmaFilter.frequency.setTargetAtTime(900 + plasma * 2400, t, 0.1);
  }

  /** Percussive chute-open crack. */
  chuteCrack(): void {
    if (!this.ctx || !this.armed || this.ctx.state !== 'running') return;
    const ctx = this.ctx;
    const t = ctx.currentTime;
    const buf = ctx.createBuffer(1, ctx.sampleRate * 0.25, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) {
      d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (ctx.sampleRate * 0.04));
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const f = ctx.createBiquadFilter();
    f.type = 'lowpass'; f.frequency.value = 900;
    const g = ctx.createGain(); g.gain.value = 0.9;
    src.connect(f); f.connect(g); g.connect(this.master!);
    src.start(t);
  }

  get active(): boolean {
    return this.enabled && this.armed && this.ctx?.state === 'running';
  }
}

/** Wire the audio engine to the flight model state. Called each frame. */
export function updateFlightAudio(a: FlightAudio, f: FlightModel, boostOn: boolean): void {
  if (!a.active) return;
  // booster intensity: thrust firing (lob boost phase)
  const booster = boostOn ? 1 : 0;
  // wind intensity: dynamic pressure, normalized by a generous 40 kPa
  const windQ = Math.min((f.airDensity * f.tGs * f.tGs) / 40000, 1);
  a.frame(booster, windQ, f.heat);
}
