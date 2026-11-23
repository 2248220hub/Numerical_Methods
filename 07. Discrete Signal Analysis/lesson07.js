/* Lesson 7 — discretisation, sampling, DTFT & DFT: symbols and widgets */
window.SYMBOLS = {
  xn: { n: "x[n] — the sample sequence", t: "data", s: "x(nT): the value of the continuous signal at the n-th sampling instant. What the computer stores.", py: "x = signal(np.arange(N) / fs)", f: "x = signal([(n, n = 0, N-1)]/fs)" },
  n: { n: "n — sample index", t: "integer", s: "0, 1, …, N−1 (Python) or 1 … N (Fortran arrays).", py: "n = np.arange(N)", f: "n = [(i, i = 0, N-1)]" },
  T: { n: "T — sampling interval", t: "seconds", s: "Time between two samples; f<sub>s</sub> = 1/T.", py: "T = 1 / fs", f: "T = 1.0_dp/fs" },
  fs: { n: "f<sub>s</sub> — sampling rate", t: "Hz", s: "Samples per second. Frequencies above f<sub>s</sub>/2 (Nyquist) fold back.", py: "fs = 44100", f: "fs = 44100.0_dp" },
  Qx: { n: "Q(x) — quantiser output", t: "level", s: "The nearest of the 2<sup>B</sup> levels (mid-tread), clipped at full scale.", py: "xq = Delta*np.clip(np.floor(x/Delta+0.5), -2**(B-1), 2**(B-1)-1)", f: "xq = delta*max(-2.0_dp**(B-1), min(2.0_dp**(B-1)-1, floor(x/delta+0.5_dp)))" },
  Delta: { n: "Δ — quantisation step (LSB)", t: "volts / ADU", s: "V<sub>FS</sub>/2<sup>B</sup>: the size of one digital level. The error is within ±Δ/2.", py: "Delta = (Vmax - Vmin) / 2**B", f: "delta = (vmax - vmin)/2.0_dp**B" },
  VFS: { n: "V<sub>FS</sub> — full-scale range", t: "volts", s: "V<sub>max</sub> − V<sub>min</sub> of the converter. Signals should use most of it, never exceed it.", py: "VFS = Vmax - Vmin", f: "vfs = vmax - vmin" },
  B: { n: "B — number of bits", t: "integer", s: "The ADC resolution: 2<sup>B</sup> codes. Each bit adds ≈ 6 dB of SQNR.", py: "B = 16", f: "integer, parameter :: B = 16" },
  Pe: { n: "P<sub>e</sub> — quantisation noise power", t: "variance", s: "Δ²/12 for an error uniform in ±Δ/2: white noise added by rounding.", py: "Pe = Delta**2 / 12", f: "pe = delta**2/12" },
  sigmae: { n: "σ²<sub>e</sub> — variance of the error", t: "variance", s: "Same as P<sub>e</sub>: computed from the uniform density 1/Δ.", py: "(x - xq).var()", f: "sum((x-xq)**2)/size(x)" },
  comb: { n: "Ш<sub>T</sub>(t) — Dirac comb", t: "distribution", s: "Infinite train of delta functions every T: ideal sampling multiplies the signal by it. Its transform is another comb (spacing 2π/T).", py: "# implicit: x[n] = x(n*T)", f: "! implicit: x(n) = x((n-1)*T)" },
  xs: { n: "x<sub>s</sub>(t) — sampled signal", t: "continuous model", s: "Impulses of weight x[n] at t = nT: the continuous-time view of a sequence.", py: "# the list x[n] itself", f: "! the array x" },
  Xs: { n: "X<sub>s</sub>(jΩ) — spectrum of the sampled signal", t: "periodic", s: "Copies of X(jΩ)/T at every multiple of Ω<sub>s</sub>. Overlapping copies = aliasing.", py: "# replicas: X(f - m*fs)", f: "! replicas at m*fs" },
  Oms: { n: "Ω<sub>s</sub> — angular sampling frequency", t: "rad/s", s: "2π/T = 2πf<sub>s</sub>.", py: "Om_s = 2*np.pi*fs", f: "om_s = 2*pi*fs" },
  fmax: { n: "f<sub>max</sub> — highest frequency in the signal", t: "Hz", s: "The band limit. Sampling needs f<sub>s</sub> &gt; 2 f<sub>max</sub> (Nyquist rate).", py: "fs_min = 2 * fmax", f: "fs_min = 2*fmax" },
  falias: { n: "f<sub>alias</sub> — apparent frequency", t: "Hz", s: "|f − k f<sub>s</sub>| with k = round(f/f<sub>s</sub>): where an undersampled tone appears.", py: "abs(f - round(f/fs)*fs)", f: "abs(f - anint(f/fs)*fs)" },
  sinc: { n: "sinc(u) = sin(πu)/(πu)", t: "function", s: "Impulse response of the ideal low-pass filter: 1 at u = 0, 0 at every other integer. The interpolation kernel.", py: "np.sinc(u)   # normalised: sin(pi u)/(pi u)", f: "merge(1.0_dp, sin(pi*u)/(pi*u), abs(u) < 1e-12_dp)" },
  omega: { n: "ω — digital frequency", t: "rad/sample", s: "2πf/f<sub>s</sub> ∈ [−π, π]: frequency measured per sample, independent of the units of time.", py: "omega = 2*np.pi*f/fs", f: "omega = 2*pi*f/fs" },
  Xw: { n: "X(e<sup>jω</sup>) — DTFT", t: "continuous, 2π-periodic", s: "Transform of an infinite sequence: Σ x[n] e<sup>−jωn</sup>. Cannot be computed exactly (infinite sum, continuous ω).", py: "Xw = np.sum(x * np.exp(-1j*w*np.arange(x.size)))", f: "Xw = sum(x*exp(cmplx(0.0_dp, -w*idx, dp)))" },
  Xk: { n: "X[k] — DFT coefficient", t: "complex", s: "The DTFT of the N samples evaluated at ω<sub>k</sub> = 2πk/N. |X[k]|: amplitude, arg X[k]: phase at f<sub>k</sub>.", py: "X = np.fft.fft(x)", f: "call dft(z, Zk, -1)" },
  N: { n: "N — number of samples", t: "integer", s: "Length of the record (and of the DFT). Resolution 1/(NT), cost N² (direct) or N log N (FFT).", py: "N = x.size", f: "N = size(x)" },
  WN: { n: "W<sub>N</sub> = e<sup>−j2π/N</sup> — twiddle factor", t: "complex root of unity", s: "The N-th root of unity: the DFT matrix has entries W<sub>N</sub><sup>kn</sup>. Its symmetries make the FFT possible.", py: "WN = np.exp(-2j*np.pi/N)", f: "wn = exp(cmplx(0.0_dp, -2*pi/N, dp))" },
  fk: { n: "f<sub>k</sub> — frequency of bin k", t: "Hz", s: "k f<sub>s</sub>/N.", py: "f = np.fft.rfftfreq(N, 1/fs)", f: "f = [(k, k = 0, N/2)]*fs/N" },
  df: { n: "Δf — bin spacing", t: "Hz", s: "f<sub>s</sub>/N = 1/T<sub>obs</sub>: set by the length of the observation. Zero padding does not change the true resolution.", py: "df = fs / N", f: "df = fs/N" },
  Xwin: { n: "X<sub>win</sub> — spectrum of the windowed data", t: "complex", s: "True spectrum convolved with the window's transform: lines get the window's main lobe and sidelobes.", py: "np.fft.fft(x * w)", f: "call dft(x*w, Xw, -1)" },
  Ww: { n: "W(e<sup>jω</sup>) — transform of the window", t: "kernel", s: "Main lobe width = resolution; sidelobe level and decay = leakage.", py: "np.fft.fft(w, 8*N)", f: "call dft(w_padded, Ww, -1)" }
};

(function () {
  const W = (window.WIDGETS = {});

  /* 1. sampling + quantisation of a signal */
  W.discretise = (root) => {
    UI.head(root, "DISCRETISE A SIGNAL · SAMPLING RATE AND BITS");
    const k = UI.kpis(root, [["fs", "sampling rate"], ["d", "levels 2^B"], ["e", "rms error"], ["ny", "Nyquist f_s/2"]]);
    const P = UI.canvas(root, 290); const c = UI.ctrls(root);
    let fs = 40, B = 3;
    UI.slider(c, { label: "sampling rate f_s [Hz]", min: 4, max: 100, step: 1, value: fs, on: (v) => { fs = v; draw(); } });
    UI.slider(c, { label: "bits B", min: 1, max: 10, step: 1, value: B, on: (v) => { B = v; draw(); } });
    const sig = (t) => 0.7 * Math.sin(2 * Math.PI * 2 * t) + 0.25 * Math.sin(2 * Math.PI * 5.3 * t + 1);
    function draw() {
      const tt = M.linspace(0, 1, 800), D = 2 / Math.pow(2, B), q = (v) => Math.max(-1, Math.min(1 - D, D * Math.floor(v / D + 0.5)));
      P.begin({ xlim: [0, 1], ylim: [-1.1, 1.1], xlabel: "time [s]", ylabel: "amplitude [V]" });
      for (let l = -1; l <= 1 + 1e-9; l += D) if (B <= 6) P.hline(l, { c: "grid", dash: [], w: 0.8 });
      P.line(tt, tt.map(sig), { c: "ink", w: 1.5 });
      const ts = M.linspace(0, Math.floor(fs) / fs, Math.floor(fs) + 1), vs = ts.map(sig), qs = vs.map(q);
      P.line(ts.flatMap((t, i) => [t, ts[i + 1] ?? t + 1 / fs]), qs.flatMap((v) => [v, v]), { c: "orange", w: 2 });
      P.pts(ts, qs, { c: "red", r: 3.5 });
      const err = M.std(tt.map((t) => sig(t) - q(sig(Math.floor(t * fs) / fs))), 0);
      k.set("fs", fs + " Hz"); k.set("d", Math.pow(2, B)); k.set("e", err.toFixed(3) + " V"); k.set("ny", (fs / 2).toFixed(1) + " Hz");
    }
    root._redraw = draw; draw();
    UI.note(root, "Black: x(t), with components at 2 and 5.3 Hz. Red dots: the stored numbers. Orange: what a zero-order hold would play back. Below f_s ≈ 11 Hz the 5.3 Hz component aliases.");
  };

  /* 2. quantiser + SQNR */
  W.quant = (root) => {
    UI.head(root, "B-BIT QUANTISER · ERROR, SQNR, CLIPPING");
    const k = UI.kpis(root, [["sq", "SQNR measured"], ["th", "6.02B + 1.76"], ["ve", "var(e) / (Δ²/12)"], ["cl", "samples clipped"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 250), Bc = UI.canvas(box, 250); const c = UI.ctrls(root);
    let B = 4, rho = 1;
    UI.slider(c, { label: "bits B", min: 2, max: 16, step: 1, value: B, on: (v) => { B = v; draw(); } });
    UI.slider(c, { label: "amplitude / half full scale ρ", min: 0.1, max: 1.4, step: 0.01, value: rho, fmt: (v) => v.toFixed(2), on: (v) => { rho = v; draw(); } });
    function draw() {
      const n = 4000, D = 2 / Math.pow(2, B), t = M.linspace(0, 1, n);
      const x = t.map((tt) => 0.9999 * rho * Math.sin(2 * Math.PI * 7.123 * tt));
      const q = x.map((v) => D * Math.max(-Math.pow(2, B - 1), Math.min(Math.pow(2, B - 1) - 1, Math.floor(v / D + 0.5))));
      const e = x.map((v, i) => v - q[i]);
      const sq = 10 * Math.log10(M.std(x, 0) ** 2 / Math.max(1e-30, M.std(e, 0) ** 2));
      A.begin({ xlim: [0, 0.3], ylim: [-1.5, 1.5], xlabel: "time", ylabel: "x and Q(x)" });
      A.hline(1, { c: "red", w: 1 }); A.hline(-1, { c: "red", w: 1 });
      A.line(t, x, { c: "ink", w: 1.5 }); A.line(t, q, { c: "orange", w: 1.5 });
      Bc.begin({ xlim: [0, 0.3], ylim: [-Math.max(D, Math.max(...e.map(Math.abs))) * 1.2, Math.max(D, Math.max(...e.map(Math.abs))) * 1.2], xlabel: "time", ylabel: "error e = x − Q(x)" });
      Bc.hline(D / 2, { c: "green", label: "+Δ/2" }); Bc.hline(-D / 2, { c: "green" }); Bc.line(t, e, { c: "blue", w: 1 });
      k.set("sq", sq.toFixed(1) + " dB"); k.set("th", (6.02 * B + 1.76).toFixed(1) + " dB"); k.set("ve", ((M.std(e, 0) ** 2) / (D * D / 12)).toFixed(2));
      k.set("cl", (100 * x.filter((v) => v > 1 - D / 2 || v < -1 - D / 2).length / n).toFixed(1) + "%");   // beyond the last levels
    }
    root._redraw = draw; draw();
    UI.note(root, "ρ = 0.25 loses two bits (≈ 12 dB). ρ > 1 clips: the error is no longer small and white — it becomes the clipped tops, a source of fake harmonics.");
  };

  /* 3. aliasing */
  W.alias = (root) => {
    UI.head(root, "ALIASING · WHAT THE SAMPLES SAY");
    const k = UI.kpis(root, [["f", "true frequency"], ["a", "alias |f − k f_s|"], ["ok", "Nyquist satisfied?"]]);
    const A = UI.canvas(root, 230), Bc = UI.canvas(root, 200); const c = UI.ctrls(root);
    let f = 9, fs = 10;
    UI.slider(c, { label: "signal frequency f [Hz]", min: 0.5, max: 30, step: 0.1, value: f, fmt: (v) => v.toFixed(1), on: (v) => { f = v; draw(); } });
    UI.slider(c, { label: "sampling rate f_s [Hz]", min: 2, max: 60, step: 0.5, value: fs, fmt: (v) => v.toFixed(1), on: (v) => { fs = v; draw(); } });
    function draw() {
      const kk = Math.round(f / fs), fa = Math.abs(f - kk * fs), tt = M.linspace(0, 1, 1200);
      A.begin({ xlim: [0, 1], ylim: [-1.3, 1.3], xlabel: "time [s]", ylabel: "amplitude" });
      A.line(tt, tt.map((t) => Math.cos(2 * Math.PI * f * t)), { c: "ink", w: 1.2 });
      A.line(tt, tt.map((t) => Math.cos(2 * Math.PI * fa * t)), { c: "red", w: 2, dash: [6, 4] });
      const ts = []; for (let t = 0; t <= 1 + 1e-9; t += 1 / fs) ts.push(t);
      A.pts(ts, ts.map((t) => Math.cos(2 * Math.PI * f * t)), { c: "orange", r: 4.5, stroke: "ink" });
      Bc.begin({ xlim: [-60, 60], ylim: [0, 1.25], xlabel: "frequency [Hz] — the spectrum of the samples repeats every f_s", ylabel: "|X_s|" });
      Bc.rect(-fs / 2, 0, fs / 2, 1.25, { c: "green", a: 0.12 });
      for (let m = -12; m <= 12; m++) for (const s of [f, -f]) { const x = s + m * fs; if (Math.abs(x) < 60) Bc.seg(x, 0, x, 1, { c: m === 0 ? "ink" : "faint", w: m === 0 ? 3 : 1.5 }); }
      Bc.seg(fa, 0, fa, 1.1, { c: "red", w: 2, dash: [4, 3] });
      k.set("f", f.toFixed(1) + " Hz"); k.set("a", fa.toFixed(2) + " Hz"); k.set("ok", fs > 2 * f ? "yes" : "no — aliased");
    }
    root._redraw = draw; draw();
    UI.note(root, "Black: the true cosine; orange: the samples; red dashed: the lowest-frequency cosine through the same samples. Bottom: replicas at m·f_s; the green band is [−f_s/2, f_s/2].");
  };

  /* 4. sinc reconstruction */
  W.sinc = (root) => {
    UI.head(root, "SINC INTERPOLATION · SUM OF SHIFTED SINCS");
    const k = UI.kpis(root, [["e", "max error (inner region)"], ["ny", "f_max / (f_s/2)"]]);
    const P = UI.canvas(root, 290); const c = UI.ctrls(root);
    let fs = 10, show = true;
    UI.slider(c, { label: "sampling rate f_s [Hz] (signal up to 3.7 Hz)", min: 4, max: 20, step: 0.1, value: fs, fmt: (v) => v.toFixed(1), on: (v) => { fs = v; draw(); } });
    UI.select(c, { label: "show individual sincs", options: [["1", "yes"], ["0", "no"]], on: (v) => { show = v === "1"; draw(); } });
    const sig = (t) => Math.sin(2 * Math.PI * 1.3 * t) + 0.5 * Math.cos(2 * Math.PI * 3.7 * t);
    const sinc = (u) => (Math.abs(u) < 1e-9 ? 1 : Math.sin(Math.PI * u) / (Math.PI * u));
    function draw() {
      const T = 1 / fs, ns = []; for (let n = -150; n <= 150; n++) ns.push(n);
      const tt = M.linspace(-1, 2, 600), rec = tt.map((t) => ns.reduce((s, n) => s + sig(n * T) * sinc((t - n * T) / T), 0));
      P.begin({ xlim: [-1, 2], ylim: [-2, 2], xlabel: "time [s]", ylabel: "x(t)" });
      if (show) ns.filter((n) => n * T >= -1 && n * T <= 2).forEach((n) => P.line(tt, tt.map((t) => sig(n * T) * sinc((t - n * T) / T)), { c: "purple", w: 0.8, a: 0.35 }));
      P.line(tt, tt.map(sig), { c: "ink", w: 3 }); P.line(tt, rec, { c: "orange", w: 2, dash: [6, 4] });
      const ts = ns.map((n) => n * T).filter((t) => t >= -1 && t <= 2); P.pts(ts, ts.map(sig), { c: "red", r: 4 });
      let em = 0; tt.forEach((t, i) => { if (t > 0 && t < 1) em = Math.max(em, Math.abs(rec[i] - sig(t))); });
      k.set("e", em.toExponential(1)); k.set("ny", (3.7 / (fs / 2)).toFixed(2));
    }
    root._redraw = draw; draw();
    UI.note(root, "Below f_s = 7.4 Hz the 3.7 Hz component is aliased and the reconstruction (orange) leaves the true signal (black), however many sincs you add.");
  };

  /* 5. DTFT of canonical sequences */
  W.dtft = (root) => {
    UI.head(root, "DTFT OF THREE CANONICAL SEQUENCES");
    const k = UI.kpis(root, [["p", "|X| at ω = 0"], ["w", "main lobe (null-to-null)"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 230), Bc = UI.canvas(box, 230); const c = UI.ctrls(root);
    let kind = "rect", L = 5, a = 0.6;
    UI.select(c, { label: "sequence", options: [["rect", "rectangular pulse, length L"], ["decay", "causal decay aⁿu[n]"], ["imp", "shifted impulse δ[n − 3]"]], on: (v) => { kind = v; draw(); } });
    UI.slider(c, { label: "L (pulse)", min: 1, max: 30, step: 1, value: L, on: (v) => { L = v; draw(); } });
    UI.slider(c, { label: "a (decay)", min: 0, max: 0.95, step: 0.01, value: a, fmt: (v) => v.toFixed(2), on: (v) => { a = v; draw(); } });
    function draw() {
      const n = M.linspace(0, 39, 40), x = n.map((i) => (kind === "rect" ? (i < L ? 1 : 0) : kind === "decay" ? Math.pow(a, i) : i === 3 ? 1 : 0));
      A.begin({ xlim: [-1, 40], ylim: [0, 1.2], xlabel: "n", ylabel: "x[n]" });
      n.forEach((i) => A.seg(i, 0, i, x[i], { c: "blue", w: 2 })); A.pts(n, x, { c: "blue", r: 3 });
      const w = M.linspace(-Math.PI, Math.PI, 600), X = w.map((om) => { let re = 0, im = 0; x.forEach((v, i) => { re += v * Math.cos(om * i); im -= v * Math.sin(om * i); }); return Math.hypot(re, im); });
      Bc.begin({ xlim: [-1, 1], ylim: [0, Math.max(...X) * 1.15], xlabel: "ω / π", ylabel: "|X(e^jω)|" });
      Bc.line(w.map((v) => v / Math.PI), X, { c: "purple", w: 2.5 });
      k.set("p", X[300].toFixed(3)); k.set("w", kind === "rect" ? `4π/L = ${(4 / L).toFixed(2)}π` : "—");
    }
    root._redraw = draw; draw();
  };

  /* 6. DFT grid: on-bin vs off-bin, zero padding */
  W.dftgrid = (root) => {
    UI.head(root, "THE DFT GRID · ON-BIN AND OFF-BIN TONES, ZERO PADDING");
    const k = UI.kpis(root, [["df", "bin spacing f_s/N"], ["n1", "bins above 1% of the peak"], ["pk", "peak frequency (padded)"]]);
    const P = UI.canvas(root, 280); const c = UI.ctrls(root);
    let f0 = 10.5, N = 64, pad = 8;
    UI.slider(c, { label: "tone frequency [Hz] (f_s = 64 Hz)", min: 5, max: 20, step: 0.05, value: f0, fmt: (v) => v.toFixed(2), on: (v) => { f0 = v; draw(); } });
    UI.slider(c, { label: "N samples", min: 16, max: 256, step: 16, value: N, on: (v) => { N = v; draw(); } });
    UI.select(c, { label: "zero padding factor", options: [["8", "×8"], ["1", "none"], ["32", "×32"]], on: (v) => { pad = +v; draw(); } });
    const dft = (x, Nf) => { const out = []; for (let kk = 0; kk <= Nf / 2; kk++) { let re = 0, im = 0; for (let i = 0; i < x.length; i++) { const ph = -2 * Math.PI * kk * i / Nf; re += x[i] * Math.cos(ph); im += x[i] * Math.sin(ph); } out.push(Math.hypot(re, im)); } return out; };
    function draw() {
      const fs = 64, x = []; for (let i = 0; i < N; i++) x.push(Math.cos(2 * Math.PI * f0 * i / fs));
      const X = dft(x, N), Xp = dft(x, N * pad), fk = X.map((_, i) => i * fs / N), fp = Xp.map((_, i) => i * fs / (N * pad));
      P.begin({ xlim: [0, 32], ylim: [0, N / 2 * 1.1], xlabel: "frequency [Hz]", ylabel: "|X[k]|" });
      P.line(fp, Xp, { c: "purple", w: 1.5 });
      fk.forEach((f, i) => P.seg(f, 0, f, X[i], { c: "blue", w: 2 })); P.pts(fk, X, { c: "blue", r: 3.5 });
      P.vline(f0, { c: "red", label: "true f" });
      const mx = Math.max(...X); k.set("df", (fs / N).toFixed(3) + " Hz"); k.set("n1", X.filter((v) => v > 0.01 * mx).length + " of " + X.length);
      k.set("pk", fp[Xp.indexOf(Math.max(...Xp))].toFixed(3) + " Hz");
    }
    root._redraw = draw; draw();
    UI.note(root, "Put the tone on a bin (multiples of f_s/N): one stem. Halfway between bins: leakage into every bin. The purple curve (zero padding) is the DTFT of the same N samples, sampled more finely: the main-lobe width f_s/N does not change.");
  };

  /* 7. windows and a weak companion */
  W.windows = (root) => {
    UI.head(root, "WINDOWS · A WEAK TONE NEXT TO A STRONG ONE");
    const k = UI.kpis(root, [["l", "level at the weak tone"], ["fl", "floor at 190 Hz"]]);
    const P = UI.canvas(root, 300); const c = UI.ctrls(root);
    let win = "rect", dB = -60, f2 = 175;
    UI.select(c, { label: "window", options: [["rect", "rectangular"], ["hann", "Hann"], ["hamming", "Hamming"], ["black", "Blackman"]], on: (v) => { win = v; draw(); } });
    UI.slider(c, { label: "weak tone level [dB]", min: -100, max: 0, step: 1, value: dB, on: (v) => { dB = v; draw(); } });
    UI.slider(c, { label: "weak tone frequency [Hz]", min: 152, max: 230, step: 0.5, value: f2, on: (v) => { f2 = v; draw(); } });
    function draw() {
      const N = 512, fs = 1000, A2 = Math.pow(10, dB / 20), w = [];
      for (let i = 0; i < N; i++) { const c1 = Math.cos(2 * Math.PI * i / N), c2 = Math.cos(4 * Math.PI * i / N); w.push(win === "rect" ? 1 : win === "hann" ? 0.5 - 0.5 * c1 : win === "hamming" ? 0.54 - 0.46 * c1 : 0.42 - 0.5 * c1 + 0.08 * c2); }
      const x = w.map((wi, i) => wi * (Math.sin(2 * Math.PI * 150.3 * i / fs) + A2 * Math.sin(2 * Math.PI * f2 * i / fs)));
      const fr = M.linspace(100, 250, 900), S = fr.map((f) => { let re = 0, im = 0; for (let i = 0; i < N; i++) { const ph = -2 * Math.PI * f * i / fs; re += x[i] * Math.cos(ph); im += x[i] * Math.sin(ph); } return re * re + im * im; });
      const mx = Math.max(...S), Sd = S.map((v) => 10 * Math.log10(v / mx + 1e-30));
      P.begin({ xlim: [100, 250], ylim: [-140, 5], xlabel: "frequency [Hz]", ylabel: "power [dB]" });
      P.line(fr, Sd, { c: "blue", w: 1.6 }); P.vline(f2, { c: "red", label: "weak tone" }); P.hline(dB, { c: "red", label: dB + " dB" });
      k.set("l", Sd[fr.findIndex((f) => f >= f2)].toFixed(1) + " dB"); k.set("fl", Sd[fr.findIndex((f) => f >= 190)].toFixed(1) + " dB");
    }
    root._redraw = draw; draw();
  };

  /* 8. cost */
  W.cost7 = (root) => {
    UI.head(root, "O(N²) vs O(N log N) · OPERATION COUNTS");
    const k = UI.kpis(root, [["d", "direct DFT N²"], ["f", "FFT (N/2) log₂N"], ["r", "ratio"], ["t", "direct at 1 Gop/s"]]);
    const P = UI.canvas(root, 270); const c = UI.ctrls(root); let q = 20;
    UI.slider(c, { label: "N = 2^q", min: 4, max: 30, step: 1, value: q, fmt: (v) => "2^" + v + " = " + M.fmt(Math.pow(2, v), 3), on: (v) => { q = v; draw(); } });
    function draw() {
      const qs = M.linspace(2, 30, 29), N = Math.pow(2, q);
      P.begin({ xlim: [4, 2e9], ylim: [1, 1e19], xlog: true, ylog: true, xlabel: "N", ylabel: "complex multiplications" });
      P.line(qs.map((v) => Math.pow(2, v)), qs.map((v) => Math.pow(4, v)), { c: "red", w: 2.5 });
      P.line(qs.map((v) => Math.pow(2, v)), qs.map((v) => Math.pow(2, v) / 2 * v), { c: "green", w: 2.5 });
      P.vline(N, { c: "orange", dash: [] });
      k.set("d", M.fmt(N * N, 3)); k.set("f", M.fmt(N / 2 * q, 3)); k.set("r", M.fmt(N * N / (N / 2 * q), 3)); k.set("t", M.human(N * N / 1e9));
    }
    root._redraw = draw; draw();
  };
})();
