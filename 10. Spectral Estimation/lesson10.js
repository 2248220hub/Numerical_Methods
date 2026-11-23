/* Lesson 10 — spectral estimation: symbols and widgets */
window.SYMBOLS = {
  Pxx: { n: "P<sub>xx</sub>(f) — true power spectral density", t: "power per unit frequency", s: "Property of the process, not of one record. What every estimator tries to approach.", py: "S = lambda f: 1e-3*(1 + 1/f)", f: "S = 1.0e-3_dp*(1 + 1/f)" },
  Rxx: { n: "R<sub>xx</sub>(k) — autocorrelation", t: "function of the lag", s: "E[x[n] x*[n−k]]. Its DTFT is the PSD (Wiener–Khinchin).", py: "np.correlate(x, x, 'full') / N", f: "R(k) = sum(x(k+1:N)*x(1:N-k))/N" },
  Pper: { n: "P̂<sub>PER</sub>(f) — periodogram", t: "estimator", s: "|X(f)|²/N. Asymptotically unbiased but inconsistent: each bin ~ P<sub>xx</sub> χ²₂/2.", py: "f, P = scipy.signal.periodogram(x, fs)", f: "P = abs(Xk)**2/N" },
  WB: { n: "W<sub>B</sub>(f) — Fejér kernel", t: "≥ 0, unit area", s: "(1/N)[sin πfN / sin πf]²: transform of the triangular weight. Smears the PSD over ~1/N; sidelobes −13 dB.", py: "W = (np.sin(np.pi*f*N)/np.sin(np.pi*f))**2/N", f: "W = (sin(pi*f*N)/sin(pi*f))**2/N" },
  Pw: { n: "P̂<sub>W</sub>(f) — Welch estimate", t: "estimator", s: "Average of K tapered, overlapping segment periodograms. ~ P<sub>xx</sub> χ²ν/ν with ν ≈ 2K<sub>eff</sub>.", py: "f, P = scipy.signal.welch(x, fs, 'hann', nperseg=L)", f: "P = P + abs(Xk)**2   ! over segments, then normalise" },
  K: { n: "K — number of segments", t: "integer", s: "⌊(N − L)/D⌋ + 1. Variance ∝ 1/K.", py: "K = (N - L)//D + 1", f: "K = (N - L)/D + 1" },
  L: { n: "L — segment length", t: "samples", s: "Sets the resolution: bins of f<sub>s</sub>/L, main lobe ~ENBW/L.", py: "nperseg = 512", f: "integer, parameter :: L = 512" },
  U: { n: "U — mean power of the taper", t: "dimensionless", s: "(1/L) Σ w². Rectangular 1, Hann 3/8. Restores the power removed by tapering.", py: "U = np.mean(w**2)", f: "U = sum(w**2)/L" },
  wn: { n: "w[n] — taper (window)", t: "0 … 1", s: "Multiplies each segment to reduce leakage. Periodic form for spectral estimation.", py: "w = scipy.signal.get_window('hann', L)", f: "w = 0.5_dp - 0.5_dp*cos(2*pi*[(n, n=0,L-1)]/L)" },
  Phi: { n: "Φ(f) — Wiener weight", t: "0 … 1", s: "(P̂ − Ŝ<sub>n</sub>)/P̂, clipped: the fraction of observed power that is signal.", py: "Phi = np.clip((P - Sn)/P, 0, 1)", f: "phi = max(0.0_dp, min(1.0_dp, (p - sn)/p))" },
  Sf: { n: "S(f) — noise model", t: "PSD", s: "White level plus a power law below the knee.", py: "S = sigma2*(1 + (fk/f)**alpha)", f: "S = sigma2*(1 + (fk/f)**alpha)" },
  sigma2: { n: "σ² — white noise level", t: "PSD units", s: "The high-frequency floor: the detector sensitivity.", py: "sigma2 = np.median(P[f > 10])", f: "sigma2 = 1.0e-3_dp" },
  fk: { n: "f<sub>k</sub> — knee frequency", t: "Hz", s: "Where the 1/f term equals the white level. 1/f<sub>k</sub> is the longest integration that still averages down.", py: "fk = 1.0", f: "fk = 1.0_dp" },
  alpha: { n: "α — spectral index (colour)", t: "≥ 0", s: "0 white, 1 pink, 2 red. Slope of log S vs log f below the knee.", py: "alpha = -np.polyfit(np.log10(f), np.log10(P), 1)[0]", f: "! least-squares slope of log P vs log f" }
};

(function () {
  const W = (window.WIDGETS = {});
  /* Timmer–König: real series of length n (power of 2) with one-sided PSD S(f) [units²/Hz] */
  const tk = (n, fs, S, rng) => { const re = new Array(n).fill(0), im = new Array(n).fill(0);
    for (let k = 1; k <= n / 2; k++) { const f = k * fs / n, a = Math.sqrt(n * fs * S(f) / 4); let g1 = rng.normal(), g2 = rng.normal(); if (k === n / 2) g2 = 0; re[k] = a * g1; im[k] = a * g2; if (k < n / 2) { re[n - k] = a * g1; im[n - k] = -a * g2; } }
    M.fft(re, im, true); return re.map((v) => v * n / n); };
  /* normalisation of the above: irfft in numpy divides by n; M.fft(inv) also divides by n — same convention */
  const win = (name, L) => Array.from({ length: L }, (_, i) => { const c1 = Math.cos(2 * Math.PI * i / L);
    if (name === "hann") return 0.5 - 0.5 * c1; if (name === "hamming") return 0.54 - 0.46 * c1;
    if (name === "tri") return 1 - Math.abs(2 * i - L) / L;
    if (name === "tukey") { const a = 0.5, m = a * L / 2; return i < m ? 0.5 * (1 - Math.cos(Math.PI * i / m)) : i > L - m ? 0.5 * (1 - Math.cos(Math.PI * (L - i) / m)) : 1; }
    if (name === "planck") { const m = 0.1 * L, t = (n) => (n <= 0 ? 0 : n >= m ? 1 : 1 / (1 + Math.exp(m / n - m / (m - n)))); return Math.min(t(i), t(L - i)); }
    return 1; });
  /* Welch: one-sided density, mean removed per segment */
  const welch = (x, fs, L, wname = "hann", D = L / 2) => {
    const w = win(wname, L), U = w.reduce((s, v) => s + v * v, 0) / L, K = Math.floor((x.length - L) / D) + 1, P = new Array(L / 2 + 1).fill(0);
    for (let i = 0; i < K; i++) { const s = i * D; let mu = 0; for (let n = 0; n < L; n++) mu += x[s + n]; mu /= L;
      const re = w.map((wi, n) => wi * (x[s + n] - mu)), im = new Array(L).fill(0); M.fft(re, im); for (let k = 0; k <= L / 2; k++) P[k] += re[k] * re[k] + im[k] * im[k]; }
    for (let k = 0; k <= L / 2; k++) P[k] *= (k === 0 || k === L / 2 ? 1 : 2) / (K * L * U * fs);
    return { f: P.map((_, k) => k * fs / L), P, K };
  };
  const chi2q = (nu, p) => { const z = M.nppf(p), a = 2 / (9 * nu); return nu * Math.pow(1 - a + z * Math.sqrt(a), 3); }; // Wilson–Hilferty
  const model = (f) => 1e-3 * (1 + 1 / f);
  const lines = (t) => 0.06 * Math.sin(2 * Math.PI * 7 * t) + 0.04 * Math.sin(2 * Math.PI * 14 * t + 1) + 0.03 * Math.sin(2 * Math.PI * 21 * t + 2) + 0.02 * Math.sin(2 * Math.PI * 28 * t + 3);

  /* 1. periodogram of the model signal for growing T */
  W.psdproblem = (root) => {
    UI.head(root, "|FFT|² OF ONE RECORD · MORE DATA, SAME SCATTER");
    const k = UI.kpis(root, [["T", "record length"], ["m", "mean P/S_n (32–45 Hz)"], ["s", "std P/S_n (32–45 Hz)"], ["n", "bins in that band"]]);
    const P = UI.canvas(root, 310); const c = UI.ctrls(root); let q = 13, seed = 1;
    UI.select(c, { label: "record length", options: [["13", "8192 samples (82 s)"], ["15", "32768 (328 s)"], ["17", "131072 (1311 s)"], ["11", "2048 (20 s)"]], on: (v) => { q = +v; draw(); } });
    UI.button(UI.btns(root), "new record", () => { seed++; draw(); });
    function draw() {
      const n = 1 << q, fs = 100, rng = M.rng(seed * 31 + q), x = tk(n, fs, model, rng).map((v, i) => v + lines(i / fs));
      const re = x.slice(), im = new Array(n).fill(0); M.fft(re, im);
      const f = [], Pp = []; for (let kk = 1; kk < n / 2; kk++) { f.push(kk * fs / n); Pp.push(2 * (re[kk] ** 2 + im[kk] ** 2) / (n * fs)); }
      const step = Math.max(1, Math.floor(f.length / 6000)), fd = f.filter((_, i) => i % step === 0), Pd = Pp.filter((_, i) => i % step === 0);
      P.begin({ xlim: [0.01, 50], ylim: [1e-7, 1], xlog: true, ylog: true, xlabel: "frequency [Hz]", ylabel: "PSD [units²/Hz]" });
      P.pts(fd, Pd, { c: "blue", r: 1.2, a: 0.5 }); P.line(f.filter((_, i) => i % 50 === 0), f.filter((_, i) => i % 50 === 0).map(model), { c: "red", w: 2.5 });
      P.rect(32, 1e-7, 45, 1, { c: "green", a: 0.1 });
      const r = []; f.forEach((ff, i) => { if (ff > 32 && ff < 45) r.push(Pp[i] / model(ff)); });
      k.set("T", (n / fs).toFixed(0) + " s"); k.set("m", M.mean(r).toFixed(2)); k.set("s", M.std(r).toFixed(2)); k.set("n", r.length);
    }
    root._redraw = draw; draw();
    UI.note(root, "Red: the true noise PSD 10⁻³(1 + 1 Hz/f). Blue: one periodogram (thinned for display). The lines at 7–28 Hz stand out, but the noise cloud spans two decades at every frequency — and stays that wide when T grows. More data give more points, not better ones.");
  };

  /* 2. distribution of P/S */
  W.chi2 = (root) => {
    UI.head(root, "DISTRIBUTION OF P̂ / P · χ²ν/ν");
    const k = UI.kpis(root, [["nu", "degrees of freedom ν = 2K"], ["s", "measured std"], ["t", "predicted √(2/ν)"], ["q", "95% range of P̂/P"]]);
    const P = UI.canvas(root, 290); const c = UI.ctrls(root); let K = 1;
    UI.slider(c, { label: "segments averaged K (1 = periodogram)", min: 1, max: 64, step: 1, value: K, on: (v) => { K = v; draw(); } });
    function draw() {
      const rng = M.rng(9 + K), nb = 4000, r = [];
      for (let b = 0; b < nb; b++) { let s = 0; for (let i = 0; i < K; i++) { const g1 = rng.normal(), g2 = rng.normal(); s += (g1 * g1 + g2 * g2) / 2; } r.push(s / K); }
      const nu = 2 * K, hi = K === 1 ? 6 : 1 + 6 / Math.sqrt(K), { edges, dens } = M.hist(r, 0, hi, 60);
      const xs = M.linspace(0.001, hi, 300);
      const pdf = xs.map((x) => Math.exp((nu / 2) * Math.log(nu / 2) + (nu / 2 - 1) * Math.log(x) - (nu / 2) * x - M.lgamma(nu / 2)));
      P.begin({ xlim: [0, hi], ylim: [0, Math.max(...dens, ...pdf) * 1.1], xlabel: "P̂ / P_true", ylabel: "density" });
      P.bars(edges, dens, { c: "blue", a: 0.5 }); P.line(xs, pdf, { c: "red", w: 2.5 }); P.vline(1, { c: "ink", label: "truth" });
      const lo = chi2q(nu, 0.025) / nu, up = chi2q(nu, 0.975) / nu;
      if (K > 2) { P.vline(lo, { c: "green" }); P.vline(up, { c: "green", label: "95%" }); }
      k.set("nu", nu); k.set("s", M.std(r).toFixed(3)); k.set("t", Math.sqrt(2 / nu).toFixed(3)); k.set("q", K === 1 ? "0.025 – 3.69" : lo.toFixed(2) + " – " + up.toFixed(2));
    }
    root._redraw = draw; draw();
    UI.note(root, "4000 simulated bins. K = 1: the exponential distribution e^{−x} — the most likely value is 0, and values 3–5× the truth are common. Averaging K independent periodograms gives χ²_{2K}/2K: narrower and more symmetric. The green lines are the 95% band used for error bars (Wilson–Hilferty approximation for the quantiles).");
  };

  /* 3. Welch: L vs K */
  W.welchK = (root) => {
    UI.head(root, "WELCH · SEGMENT LENGTH L AGAINST NUMBER OF SEGMENTS K");
    const k = UI.kpis(root, [["K", "segments K"], ["df", "bin width f_s/L"], ["s", "scatter of P̂/S (32–45 Hz)"], ["cv", "truth inside 95% band"]]);
    const P = UI.canvas(root, 320); const c = UI.ctrls(root); let q = 9, wn = "hann", seed = 2;
    UI.slider(c, { label: "L = 2^q samples", min: 6, max: 14, step: 1, value: q, fmt: (v) => (1 << v) + " (" + ((1 << v) / 100).toFixed(2) + " s)", on: (v) => { q = v; draw(); } });
    UI.select(c, { label: "taper", options: [["hann", "Hann"], ["rect", "rectangular (Bartlett-like)"]], on: (v) => { wn = v; draw(); } });
    UI.button(UI.btns(root), "new record", () => { seed++; draw(); });
    let x = null, lastSeed = -1;
    function draw() {
      const fs = 100, n = 1 << 16; if (seed !== lastSeed) { x = tk(n, fs, model, M.rng(seed * 7)).map((v, i) => v + lines(i / fs)); lastSeed = seed; }
      const L = 1 << q, { f, P: Pw, K } = welch(x, fs, L, wn), nu = 2 * K * (wn === "hann" ? 0.95 : 1) * (wn === "rect" ? 0.75 : 1);
      const lo = nu / chi2q(nu, 0.975), up = nu / chi2q(nu, 0.025);
      const ff = f.slice(1), pp = Pw.slice(1);
      P.begin({ xlim: [0.01, 50], ylim: [1e-6, 0.3], xlog: true, ylog: true, xlabel: "frequency [Hz]", ylabel: "PSD [units²/Hz]" });
      P.fill(ff, pp.map((v) => v * lo), pp.map((v) => v * up), { c: "blue", a: 0.18 });
      P.line(ff, pp, { c: "blue", w: 1.5 }); P.line(M.logspace(-2, 1.7, 200), M.logspace(-2, 1.7, 200).map(model), { c: "red", w: 2, dash: [5, 4] });
      const r = [], inside = []; ff.forEach((fv, i) => { if (fv > 32 && fv < 45) { r.push(pp[i] / model(fv)); inside.push(model(fv) >= pp[i] * lo && model(fv) <= pp[i] * up); } });
      k.set("K", K + " (ν ≈ " + nu.toFixed(0) + ")"); k.set("df", (fs / L).toFixed(3) + " Hz"); k.set("s", r.length > 2 ? M.std(r).toFixed(3) + "  (1/√K " + (1 / Math.sqrt(K)).toFixed(3) + ")" : "too few bins");
      k.set("cv", r.length > 2 ? Math.round(100 * inside.filter(Boolean).length / inside.length) + "% of " + inside.length : "—");
    }
    root._redraw = draw; draw();
    UI.note(root, "655 s of the model signal. Short segments: many averages, a smooth estimate with a tight band — but wide bins: the lines are broadened and diluted, and the knee at 1 Hz is unresolved. Long segments: sharp lines, noisy continuum. With 50% overlap and Hann, ν ≈ 1.9 K; the rectangular taper with 50% overlap gains less (strongly correlated segments).");
  };

  /* 4. windows */
  W.windows10 = (root) => {
    UI.head(root, "WINDOWS · A WEAK LINE 20 dB BELOW A STRONG ONE");
    const k = UI.kpis(root, [["c", "weak line / valley"], ["en", "ENBW [bins]"], ["U", "U = mean w²"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 260), Bc = UI.canvas(box, 260); const c = UI.ctrls(root);
    let wn = "rect", f2 = 0.22, dB = -20;
    UI.select(c, { label: "window", options: [["rect", "rectangular"], ["hann", "Hann"], ["hamming", "Hamming"], ["tri", "triangular"], ["tukey", "Tukey α = 0.5"], ["planck", "Planck ε = 0.1"]], on: (v) => { wn = v; draw(); } });
    UI.slider(c, { label: "weak line frequency [cycles/sample]", min: 0.205, max: 0.3, step: 0.001, value: f2, fmt: (v) => v.toFixed(3), on: (v) => { f2 = v; draw(); } });
    UI.slider(c, { label: "weak line level [dB]", min: -60, max: 0, step: 1, value: dB, on: (v) => { dB = v; draw(); } });
    function draw() {
      const L = 256, Kseg = 16, rng = M.rng(4), n = L * Kseg + L, A2 = 10 * Math.pow(10, dB / 20);
      const x = Array.from({ length: n }, (_, i) => 10 * Math.sin(2 * Math.PI * 0.2 * i + 0.3) + A2 * Math.sin(2 * Math.PI * f2 * i + 1.1) + 0.3 * rng.normal());
      const { f, P } = welch(x, 1, L, wn, L), w = win(wn, L);
      A.begin({ xlim: [0, L], ylim: [-0.05, 1.1], xlabel: "n", ylabel: "w[n]" }); A.line(M.linspace(0, L - 1, L), w, { c: "purple", w: 2.5 });
      const Pd = P.map((v) => 10 * Math.log10(v / Math.max(...P) + 1e-16));
      Bc.begin({ xlim: [0.1, 0.35], ylim: [-110, 5], xlabel: "f [cycles/sample]", ylabel: "Welch power [dB]" });
      Bc.line(f, Pd, { c: "blue", w: 1.6 }); Bc.vline(f2, { c: "red", label: "weak" });
      const iw = Math.round(f2 * L), iv = Math.round((0.2 + f2) / 2 * L);
      const pk = Math.max(P[iw - 1] || 0, P[iw], P[iw + 1] || 0), val = Math.min(...P.slice(Math.round(0.2 * L) + 2, iw - 1).concat([Infinity]));
      const enbw = L * w.reduce((s, v) => s + v * v, 0) / Math.pow(M.sum(w), 2);
      k.set("c", isFinite(val) ? (pk / val).toFixed(1) : "lines merged"); k.set("en", enbw.toFixed(2)); k.set("U", (w.reduce((s, v) => s + v * v, 0) / L).toFixed(3));
    }
    root._redraw = draw; draw();
    UI.note(root, "Welch with 16 segments of L = 256. The rectangular window buries the weak line in the skirt of the strong one (contrast ≈ 1.7). Hann and Hamming lift it out (≈ 30). Move the weak line closer than ~2 bins (0.008) and every window merges them: resolution comes from L, not from the window. At −60 dB only fast-decaying windows survive.");
  };

  /* 5. Wiener on lines in noise */
  W.wiener10 = (root) => {
    UI.head(root, "WIENER FROM THE DATA · LINES IN WHITE NOISE");
    const k = UI.kpis(root, [["b", "bins with Φ > ½"], ["e0", "rms error before"], ["e1", "rms error after"], ["sn", "Ŝn / true"]]);
    const A = UI.canvas(root, 230), Bc = UI.canvas(root, 210); const c = UI.ctrls(root); let sig = 0.3, q = 8;
    UI.slider(c, { label: "noise rms", min: 0.05, max: 1.5, step: 0.05, value: sig, on: (v) => { sig = v; draw(); } });
    UI.slider(c, { label: "Welch L = 2^q", min: 6, max: 11, step: 1, value: q, fmt: (v) => 1 << v, on: (v) => { q = v; draw(); } });
    function draw() {
      const fs = 100, n = 1 << 14, rng = M.rng(17), clean = Array.from({ length: n }, (_, i) => 3 * lines(i / fs)), y = clean.map((v) => v + sig * rng.normal());
      const L = 1 << q, { f, P } = welch(y, fs, L, "hann"), band = P.filter((_, i) => f[i] > 35 && f[i] < 48).sort((a, b) => a - b), Sn = band[Math.floor(band.length / 2)];
      const Phi = P.map((p) => Math.max(0, Math.min(1, (p - Sn) / p)));
      // apply Φ (interpolated on the fine grid) to the full FFT
      const re = y.slice(), im = new Array(n).fill(0); M.fft(re, im);
      for (let kk = 0; kk < n; kk++) { const fk = Math.min(kk, n - kk) * fs / n, j = Math.min(L / 2, Math.round(fk * L / fs)); re[kk] *= Phi[j]; im[kk] *= Phi[j]; }
      M.fft(re, im, true);
      const t = M.linspace(0, 1.5, 151).map((v) => v), idx = t.map((v) => Math.round(v * fs));
      A.begin({ xlim: [0, 1.5], ylim: [-1.2 - 2.5 * sig, 1.2 + 2.5 * sig], xlabel: "time [s]", ylabel: "signal" });
      A.line(t, idx.map((i) => y[i]), { c: "faint", w: 1 }); A.line(t, idx.map((i) => clean[i]), { c: "ink", w: 2 }); A.line(t, idx.map((i) => re[i]), { c: "purple", w: 1.8, dash: [5, 3] });
      Bc.begin({ xlim: [0, 50], ylim: [1e-6, 10], ylog: true, xlabel: "frequency [Hz]", ylabel: "Welch PSD / Φ" });
      Bc.line(f, P, { c: "orange", w: 1.5 }); Bc.hline(Sn, { c: "red", label: "Ŝn" }); Bc.line(f, Phi.map((v) => Math.max(1e-6, v)), { c: "purple", w: 2 });
      const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
      k.set("b", Phi.filter((v) => v > 0.5).length + " of " + Phi.length); k.set("e0", rms(y.map((v, i) => v - clean[i])).toFixed(3)); k.set("e1", rms(re.map((v, i) => v - clean[i])).toFixed(3));
      k.set("sn", (Sn / (2 * sig * sig / fs)).toFixed(2));
    }
    root._redraw = draw; draw();
    UI.legend(root, [["faint", "data"], ["ink", "clean signal"], ["purple", "Wiener-filtered  /  Φ(f)"], ["orange", "Welch PSD"]]);
    UI.note(root, "Φ is non-zero only around 7, 14, 21, 28 Hz: the optimal filter for a periodic signal is a comb. The floor Ŝn is the median in a line-free band (the median of χ²ν/ν is slightly below 1, hence Ŝn/true a bit under 1 for small K). Longer L: narrower teeth, less noise let through, but each Welch bin is noisier.");
  };

  /* 6. knee noise: generate, fit */
  W.knee = (root) => {
    UI.head(root, "WHITE NOISE WITH A 1/f KNEE · LEAST SQUARES vs WHITTLE");
    const k = UI.kpis(root, [["tr", "truth σ², f_k, α"], ["ls", "least squares on log Welch"], ["wh", "Whittle on periodogram"]]);
    const P = UI.canvas(root, 320); const c = UI.ctrls(root); let fk = 1, al = 1, seed = 3;
    UI.slider(c, { label: "knee f_k [Hz]", min: 0.1, max: 10, step: 0.1, value: fk, on: (v) => { fk = v; draw(); } });
    UI.slider(c, { label: "slope α", min: 0, max: 3, step: 0.1, value: al, fmt: (v) => v.toFixed(1), on: (v) => { al = v; draw(); } });
    UI.button(UI.btns(root), "new record (64 s)", () => { seed++; draw(); });
    function draw() {
      const fs = 100, n = 1 << 13, s2 = 1e-3, S = (f) => s2 * (1 + Math.pow(fk / f, al)), x = tk(n, fs, S, M.rng(seed * 13));
      const re = x.slice(), im = new Array(n).fill(0); M.fft(re, im);
      const fp = [], pp = []; for (let kk = 1; kk < n / 2; kk++) { fp.push(kk * fs / n); pp.push(2 * (re[kk] ** 2 + im[kk] ** 2) / (n * fs)); }
      const L = 1024, wl = welch(x, fs, L, "hann"), fw = wl.f.slice(1, -1), pw = wl.P.slice(1, -1);
      // grid search over (fk, alpha); sigma² analytic in each case
      const grid = (fit) => { let best = [Infinity]; for (let a = 0; a <= 3.0001; a += 0.05) for (let lf = -1.5; lf <= 1.3; lf += 0.035) { const kf = Math.pow(10, lf); const r = fit(kf, a); if (r[0] < best[0]) best = [r[0], r[1], kf, a]; } return best; };
      const ls = grid((kf, a) => { const g = fw.map((f) => Math.log(1 + Math.pow(kf / f, a))), lp = pw.map(Math.log); const ls2 = M.mean(lp.map((v, i) => v - g[i])); return [lp.reduce((s, v, i) => s + (v - ls2 - g[i]) ** 2, 0), Math.exp(ls2)]; });
      const wh = grid((kf, a) => { const g = fp.map((f) => 1 + Math.pow(kf / f, a)), s = M.mean(pp.map((v, i) => v / g[i])); return [fp.reduce((t, _, i) => t + Math.log(s * g[i]) + pp[i] / (s * g[i]), 0), s]; });
      P.begin({ xlim: [fs / n, 50], ylim: [1e-6, 10], xlog: true, ylog: true, xlabel: "frequency [Hz]", ylabel: "PSD" });
      const st = Math.max(1, Math.floor(fp.length / 3000)); P.pts(fp.filter((_, i) => i % st === 0), pp.filter((_, i) => i % st === 0), { c: "faint", r: 1.2 });
      P.line(fw, pw, { c: "blue", w: 2 });
      const ff = M.logspace(Math.log10(fs / n), Math.log10(50), 200);
      P.line(ff, ff.map(S), { c: "ink", w: 2.5 }); P.line(ff, ff.map((f) => ls[1] * (1 + Math.pow(ls[2] / f, ls[3]))), { c: "orange", w: 2, dash: [6, 4] });
      P.line(ff, ff.map((f) => wh[1] * (1 + Math.pow(wh[2] / f, wh[3]))), { c: "green", w: 2, dash: [2, 3] }); P.vline(fk, { c: "red", label: "f_k" });
      const fm = (b) => `${(b[1] * 1e3).toFixed(2)}e-3, ${b[2].toFixed(2)}, ${b[3].toFixed(2)}`;
      k.set("tr", `1.00e-3, ${fk.toFixed(2)}, ${al.toFixed(2)}`); k.set("ls", fm(ls)); k.set("wh", fm(wh));
    }
    root._redraw = draw; draw();
    UI.legend(root, [["faint", "periodogram"], ["blue", "Welch, L = 1024"], ["ink", "truth"], ["orange", "least squares on log Welch"], ["green", "Whittle"]]);
    UI.note(root, "Generated with Timmer–König. Least squares on log Welch tends to put σ² about 6% low (the log of a χ²ν/ν variable is biased by ≈ −1/ν) and has only a few Welch bins below a low knee. Whittle uses every periodogram bin with its exact distribution. Press 'new record' several times: the Whittle values scatter less around the truth.");
  };

  /* 7. steep spectrum */
  W.steep = (root) => {
    UI.head(root, "A STEEP SPECTRUM · LEAKAGE AND PREWHITENING");
    const k = UI.kpis(root, [["hi", "estimate / truth at 40 Hz"], ["lo", "estimate / truth at the lowest bins"]]);
    const P = UI.canvas(root, 320); const c = UI.ctrls(root); let wn = "rect", pw = false, slope = 4;
    UI.select(c, { label: "window", options: [["rect", "rectangular"], ["hann", "Hann"], ["hamming", "Hamming"]], on: (v) => { wn = v; draw(); } });
    UI.select(c, { label: "prewhitening", options: [["0", "none"], ["1", "first difference, then divide by |A|²"]], on: (v) => { pw = v === "1"; draw(); } });
    UI.slider(c, { label: "spectral slope (S ∝ f^−s)", min: 1, max: 6, step: 0.5, value: slope, on: (v) => { slope = v; draw(); } });
    function draw() {
      const fs = 100, n = 1 << 16, S = (f) => 1e-2 * Math.pow(Math.max(f, 0.05), -slope) + 1e-9, x = tk(n, fs, S, M.rng(21));
      const xx = pw ? x.slice(1).map((v, i) => v - x[i]) : x, L = 512, { f, P: Pw } = welch(xx.length >= n ? xx : xx.concat([0]), fs, L, wn);
      const est = Pw.map((v, i) => (pw ? (i ? v / (2 - 2 * Math.cos(2 * Math.PI * f[i] / fs)) : NaN) : v));
      P.begin({ xlim: [0.15, 50], ylim: [1e-12, 1e5], xlog: true, ylog: true, xlabel: "frequency [Hz]", ylabel: "PSD" });
      const ff = f.slice(1); P.line(ff, ff.map(S), { c: "red", w: 3 }); P.line(ff, est.slice(1), { c: "blue", w: 1.6 });
      const i40 = Math.round(40 * L / fs); k.set("hi", M.fmt(est[i40] / S(f[i40]), 3)); k.set("lo", M.fmt(M.mean([2, 3].map((i) => est[i] / S(f[i]))), 3));
    }
    root._redraw = draw; draw();
    UI.note(root, "Red: truth, f^−s over a floor at 10⁻⁹. Rectangular sidelobes fall as f⁻², so for s > 2 the leakage from the strong low frequencies sets a false floor. Hann (f⁻⁶) is fine up to s ≈ 6 except in the lowest bins; a first difference flattens the spectrum by (2πf/f_s)² and fixes them.");
  };

  /* 8. spectrogram of a linear chirp: ridge width vs window */
  W.stft10 = (root) => {
    UI.head(root, "SPECTROGRAM OF A LINEAR CHIRP · WHICH WINDOW LENGTH?");
    const k = UI.kpis(root, [["T", "window L / f_s"], ["w", "ridge width (median −3 dB)"], ["o", "1/√rate"]]);
    const P = UI.canvas(root, 320), Bc = UI.canvas(root, 170); const c = UI.ctrls(root); let q = 9;
    UI.slider(c, { label: "window L = 2^q (f_s = 2048 Hz)", min: 5, max: 11, step: 1, value: q, fmt: (v) => (1 << v) + " = " + (1000 * (1 << v) / 2048).toFixed(0) + " ms", on: (v) => { q = v; draw(); } });
    const fs = 2048, n = 4 * fs, rate = 87.5, rng = M.rng(8), x = Array.from({ length: n }, (_, i) => { const t = i / fs; return Math.sin(2 * Math.PI * (50 * t + 0.5 * rate * t * t)) + 0.5 * rng.normal(); });
    const widths = {};
    function spec(L) { const H = L / 4, nf = Math.floor((n - L) / H) + 1, nb = L / 2, w = win("hann", L), Z = []; for (let m = 0; m < nf; m++) { const re = w.map((wi, i) => wi * x[m * H + i]), im = new Array(L).fill(0); M.fft(re, im); Z.push(Array.from({ length: nb }, (_, b) => re[b] ** 2 + im[b] ** 2)); } return { Z, H, nf, nb }; }
    function width(L) { if (widths[L]) return widths[L]; const { Z } = spec(L), df = fs / L, ws = []; Z.forEach((col) => { const mx = Math.max(...col), cnt = col.filter((v) => v > mx / 2).length; ws.push(cnt * df); }); ws.sort((a, b) => a - b); return (widths[L] = ws[Math.floor(ws.length / 2)]); }
    function draw() {
      const L = 1 << q, { Z, H, nf, nb } = spec(L), img = new Array(nf * nb);
      let mx = 0; Z.forEach((col) => col.forEach((v) => (mx = Math.max(mx, v))));
      for (let m = 0; m < nf; m++) for (let b = 0; b < nb; b++) img[(nb - 1 - b) * nf + m] = 10 * Math.log10(Z[m][b] / mx + 1e-12);
      P.begin({ xlim: [0, 4], ylim: [0, 600], xlabel: "time [s]", ylabel: "frequency [Hz]" });
      P.image(img, nf, nb, L / 2 / fs, L / 2 / fs + nf * H / fs, 0, fs / 2, { cmap: "magma", lo: -40, hi: 0 });
      const qs = [5, 6, 7, 8, 9, 10, 11], ws = qs.map((qq) => width(1 << qq));
      Bc.begin({ xlim: [0.01, 1.2], ylim: [3, 200], xlog: true, ylog: true, xlabel: "window length [s]", ylabel: "ridge width [Hz]" });
      Bc.line(qs.map((qq) => (1 << qq) / fs), ws, { c: "purple", w: 2 }); Bc.pts(qs.map((qq) => (1 << qq) / fs), ws, { c: "purple", r: 3.5 }); Bc.vline(L / fs, { c: "orange", dash: [] }); Bc.vline(1 / Math.sqrt(rate), { c: "green", label: "1/√rate" });
      k.set("T", (1000 * L / fs).toFixed(0) + " ms"); k.set("w", width(L).toFixed(1) + " Hz"); k.set("o", (1000 / Math.sqrt(rate)).toFixed(0) + " ms");
    }
    root._redraw = draw; draw();
    UI.note(root, "Chirp 50 → 400 Hz in 4 s (87.5 Hz/s) in white noise; Hann, hop L/4. Short windows: wide bins (≈ 1.44 f_s/L). Long windows: the line sweeps across many bins within one window. The minimum of the bottom curve sits a little above 1/√rate, because the Hann taper shortens the effective window.");
  };
})();
