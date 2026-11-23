/* Lesson 9 — convolution, correlation, block algorithms, deconvolution: symbols and widgets */
window.SYMBOLS = {
  yn: { n: "y[n] — output of the system", t: "sequence", s: "The filtered (or blurred) signal: length N + M − 1 for a full linear convolution.", py: "y = np.convolve(x, h)", f: "y(n) = y(n) + x(k)*h(n-k+1)" },
  xn: { n: "x[k] — input sequence", t: "data", s: "The record being filtered: N samples, possibly an endless stream.", py: "x = data", f: "real(dp) :: x(N)" },
  hn: { n: "h[n] — impulse response (kernel)", t: "M samples", s: "Output of the system for a single unit impulse. A PSF, a smoothing window, an FIR filter, a template.", py: "h = np.hanning(M); h /= h.sum()", f: "real(dp) :: h(M)" },
  cn: { n: "c[n] — cross-correlation", t: "sequence", s: "Overlap of x with h shifted by n. Peaks where the data contain the template.", py: "c = scipy.signal.correlate(x, h)", f: "Xk = Xk*conjg(Hk)" },
  Xk: { n: "X[k] — spectrum of the input", t: "complex", s: "DFT of the (zero-padded) input.", py: "X = np.fft.rfft(x, nfft)", f: "call fftw_execute_dft_r2c(pf, xp, Xk)" },
  Hk: { n: "H[k] — transfer function", t: "complex", s: "DFT of the kernel stored in wrap-around order. Its eigenvalues for the circulant convolution matrix; |H| small ⇒ hard to deconvolve.", py: "H = np.fft.rfft(hb)", f: "call fftw_execute_dft_r2c(pf, buf, Hk)" },
  Nfft: { n: "N<sub>fft</sub> — FFT length", t: "integer", s: "Length of each transform. ≥ N + M − 1 for a single shot; ≈ 4–8 M for block algorithms; a fast length.", py: "nfft = scipy.fft.next_fast_len(N + M - 1)", f: "integer :: nfft = 1024" },
  M: { n: "M — kernel length", t: "integer", s: "m₊ + m₋ + 1: positive lags, negative lags and lag zero.", py: "M = len(h)", f: "M = size(h)" },
  Wk: { n: "W[k] — Wiener filter", t: "complex", s: "H*/(|H|² + S<sub>n</sub>/S<sub>x</sub>): the linear least-squares deconvolution.", py: "W = np.conj(H) / (abs(H)**2 + Sn/Sx)", f: "W = conjg(Hk)/(abs(Hk)**2 + sn/sx)" },
  Phi: { n: "Φ — Wiener weight", t: "0 … 1", s: "Signal / (signal + noise) at the detector, per frequency. 1: trust the data; 0: suppress.", py: "Phi = np.clip((P - Sn) / P, 0, 1)", f: "phi = max(0.0_dp, min(1.0_dp, (p - sn)/p))" },
  Sn: { n: "S<sub>n</sub>(f) — noise PSD", t: "power / Hz", s: "From the flat floor of |Y|², a signal-free region, or a detector model.", py: "Sn = np.median(P[f > f_floor])", f: "sn = median(p(kfloor:))" },
  Sx: { n: "S<sub>x</sub>(f) — signal PSD", t: "power / Hz", s: "Power spectrum of the true (unblurred) signal; in practice estimated as (|Y|² − S<sub>n</sub>)/|H|².", py: "Sx = np.clip(P - Sn, 0, None) / abs(H)**2", f: "sx = max(0.0_dp, p - sn)/abs(Hk)**2" },
  lam: { n: "λ — regularisation parameter", t: "≥ 0", s: "Constant replacing S<sub>n</sub>/S<sub>x</sub> (Tikhonov). Too small: noise; too large: blur left in.", py: "Xhat = np.conj(H)*Y/(abs(H)**2 + lam)", f: "Xk = conjg(Hk)*Yk/(abs(Hk)**2 + lam)" },
  taug: { n: "τ<sub>g</sub>(f) — group delay", t: "samples", s: "−(1/2π) dφ/df: delay of the envelope at frequency f. Constant for linear-phase FIR, frequency-dependent for IIR.", py: "w, gd = scipy.signal.group_delay((b, a))", f: "! numerical derivative of the unwrapped phase" }
};

(function () {
  const W = (window.WIDGETS = {});
  const conv = (x, h) => { const y = new Array(x.length + h.length - 1).fill(0); for (let i = 0; i < x.length; i++) for (let j = 0; j < h.length; j++) y[i + j] += x[i] * h[j]; return y; };
  const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);

  /* 1. flip and slide */
  W.convslide = (root) => {
    UI.head(root, "FLIP, SLIDE, MULTIPLY, ADD");
    const k = UI.kpis(root, [["n", "output index n"], ["y", "y[n] = Σ x[k] h[n−k]"], ["L", "output length N + M − 1"]]);
    const A = UI.canvas(root, 220), Bc = UI.canvas(root, 190); const c = UI.ctrls(root);
    const x = Array.from({ length: 40 }, (_, i) => (i >= 8 && i < 20 ? 1 : 0) + (i === 28 ? 2 : 0));
    let kind = "box", n = 14;
    const kernels = { box: [1, 1, 1, 1, 1, 1, 1].map((v) => v / 7), gauss: Array.from({ length: 11 }, (_, i) => Math.exp(-0.5 * ((i - 5) / 2) ** 2)).map((v, _, a) => v / 4.97), rc: Array.from({ length: 12 }, (_, i) => Math.exp(-i / 3) / 3.1) };
    UI.select(c, { label: "kernel h", options: [["box", "boxcar, M = 7"], ["gauss", "Gaussian, M = 11"], ["rc", "causal exponential (RC), M = 12"]], on: (v) => { kind = v; draw(); } });
    const sl = UI.slider(c, { label: "slide: output index n", min: 0, max: 50, step: 1, value: n, on: (v) => { n = v; draw(); } });
    UI.button(UI.btns(root), "▶ animate", () => { let t = 0; const id = setInterval(() => { n = t; sl.set(t); draw(); if (++t > x.length + kernels[kind].length - 2) clearInterval(id); }, 70); });
    function draw() {
      const h = kernels[kind], M = h.length, y = conv(x, h), ks = M.length;
      A.begin({ xlim: [-12, 52], ylim: [-0.1, 2.2], xlabel: "k", ylabel: "x[k] and h[n − k]" });
      x.forEach((v, i) => A.seg(i, 0, i, v, { c: "blue", w: 3 }));
      const hm = Math.max(...h), sc = 1 / hm; // flipped kernel drawn scaled to height 1
      h.forEach((v, j) => { const kk = n - j; A.seg(kk + 0.25, 0, kk + 0.25, v * sc, { c: "orange", w: 3 }); if (kk >= 0 && kk < x.length && x[kk]) A.rect(kk - 0.4, 0, kk + 0.5, Math.min(x[kk], v * sc), { c: "red", a: 0.25 }); });
      A.text(n - M + 1, 1.15, "h flipped (scaled)", { c: "orange", size: 10 });
      Bc.begin({ xlim: [-12, 52], ylim: [-0.1, Math.max(...y) * 1.15], xlabel: "n", ylabel: "y[n]" });
      y.forEach((v, i) => Bc.seg(i, 0, i, v, { c: i <= n ? "purple" : "line", w: 3 }));
      if (n < y.length) Bc.pts([n], [y[n]], { c: "red", r: 5 });
      k.set("n", n); k.set("y", n < y.length ? y[n].toFixed(3) : "0 (past the end)"); k.set("L", `${x.length} + ${M} − 1 = ${y.length}`);
    }
    root._redraw = draw; draw();
    UI.note(root, "Blue: input (a pulse and a spike). Orange: the kernel flipped and shifted to n. Red: the overlap whose sum is y[n]. The spike reproduces the kernel itself — that is why h is called the impulse response. With the causal kernel the output only starts after the input: a delay.");
  };

  /* 2. matched filter */
  W.matched = (root) => {
    UI.head(root, "MATCHED FILTER · FIND A KNOWN SHAPE IN NOISE");
    const k = UI.kpis(root, [["t", "true position"], ["f", "found (peak of correlation)"], ["s", "peak / rms"], ["x", "expected peak A‖h‖"]]);
    const A = UI.canvas(root, 190), Bc = UI.canvas(root, 190); const c = UI.ctrls(root);
    let amp = 0.8, kind = "chirp", seed = 1;
    UI.slider(c, { label: "signal amplitude (noise σ = 1)", min: 0, max: 2, step: 0.05, value: amp, on: (v) => { amp = v; draw(); } });
    UI.select(c, { label: "template", options: [["chirp", "chirp (rising frequency)"], ["pulse", "Gaussian pulse"]], on: (v) => { kind = v; draw(); } });
    UI.button(UI.btns(root), "new noise + position", () => { seed++; draw(); });
    function draw() {
      const N = 2048, L = 200, rng = M.rng(seed * 97 + 3), t0 = 300 + Math.floor(rng() * 1400);
      const h = Array.from({ length: L }, (_, i) => { const w = Math.sin(Math.PI * i / L) ** 2; return kind === "chirp" ? w * Math.sin(2 * Math.PI * (0.02 * i + 0.0006 * i * i)) : Math.exp(-0.5 * ((i - L / 2) / 18) ** 2); });
      const d = Array.from({ length: N }, (_, i) => rng.normal() + (i >= t0 && i < t0 + L ? amp * h[i - t0] : 0));
      const nf = 4096, re = d.concat(new Array(nf - N).fill(0)), im = new Array(nf).fill(0), hr = h.concat(new Array(nf - L).fill(0)), hi = new Array(nf).fill(0);
      M.fft(re, im); M.fft(hr, hi);
      for (let i = 0; i < nf; i++) { const a = re[i], b = im[i]; re[i] = a * hr[i] + b * hi[i]; im[i] = b * hr[i] - a * hi[i]; } // X H*
      M.fft(re, im, true); const cc = re.slice(0, N), nh = Math.sqrt(h.reduce((s, v) => s + v * v, 0)), cn = cc.map((v) => v / nh);
      let best = 0; cn.forEach((v, i) => { if (Math.abs(v) > Math.abs(cn[best])) best = i; });
      const ii = M.linspace(0, N - 1, N);
      A.begin({ xlim: [0, N], ylim: [-4.5, 4.5], xlabel: "sample", ylabel: "data" }); A.line(ii, d, { c: "slate", w: 0.8 });
      A.rect(t0, -4.5, t0 + L, 4.5, { c: "green", a: 0.12 }); A.line(ii.slice(t0, t0 + L), h.map((v) => amp * v), { c: "green", w: 1.5 });
      Bc.begin({ xlim: [0, N], ylim: [-6, 6 + Math.abs(cn[best])], xlabel: "lag", ylabel: "c / ‖h‖ (σ units)" }); Bc.line(ii, cn, { c: "purple", w: 1 });
      Bc.vline(t0, { c: "green", label: "true" }); Bc.pts([best], [cn[best]], { c: "red", r: 5 });
      k.set("t", t0); k.set("f", best + (Math.abs(best - t0) <= 3 ? " ✓" : " ✗")); k.set("s", (Math.abs(cn[best]) / M.std(cn, 0)).toFixed(1)); k.set("x", (amp * nh).toFixed(1) + " σ");
    }
    root._redraw = draw; draw();
    UI.note(root, "Correlation output normalised by ‖h‖ is in units of the noise σ. Expected peak = amplitude × ‖h‖ — a long template collects a lot of signal energy, so a chirp of amplitude 0.8, hard to see in the data, still stands out at ≈ 5σ. Below ≈ 0.5 (expected peak under ≈ 3σ) the highest noise peak among 2000 lags often wins.");
  };

  /* 3. wrap-around */
  W.wrap = (root) => {
    UI.head(root, "CIRCULAR vs LINEAR · WRAP-AROUND POLLUTION");
    const k = UI.kpis(root, [["e", "max error, first m₊ samples"], ["ec", "max error, central part"], ["nf", "N_fft"]]);
    const P = UI.canvas(root, 290); const c = UI.ctrls(root);
    let Mk = 15, pad = "none", kind = "causal";
    UI.slider(c, { label: "kernel length M", min: 3, max: 31, step: 2, value: Mk, on: (v) => { Mk = v; draw(); } });
    UI.select(c, { label: "kernel", options: [["causal", "causal (m₋ = 0)"], ["centred", "centred (m₊ = m₋)"]], on: (v) => { kind = v; draw(); } });
    UI.select(c, { label: "zero padding", options: [["none", "none: N_fft = N"], ["half", "K = max(m₊, m₋)"], ["full", "N_fft = N + M − 1"]], on: (v) => { pad = v; draw(); } });
    function draw() {
      const N = 64, x = Array.from({ length: N }, (_, i) => (i < 40 ? 0.2 : 1) + 0.15 * Math.sin(i / 5));
      const mm = kind === "causal" ? 0 : (Mk - 1) / 2, mp = Mk - 1 - mm, h = Array.from({ length: Mk }, (_, i) => Math.exp(-((i - (kind === "causal" ? 0 : mm)) ** 2) / (2 * (Mk / 4) ** 2)));
      const s = M.sum(h); h.forEach((v, i) => (h[i] = v / s));
      const nf = pad === "none" ? N : pad === "half" ? N + Math.max(mp, mm) : N + Mk - 1;
      // circular convolution of period nf, lag index j-mm
      const yc = Array.from({ length: N }, (_, n) => { let t = 0; for (let j = 0; j < Mk; j++) { const lag = j - mm, idx = ((n - lag) % nf + nf) % nf; t += h[j] * (idx < N ? x[idx] : 0); } return t; });
      const yl = Array.from({ length: N }, (_, n) => { let t = 0; for (let j = 0; j < Mk; j++) { const idx = n - (j - mm); t += h[j] * (idx >= 0 && idx < N ? x[idx] : 0); } return t; });
      P.begin({ xlim: [-1, N], ylim: [-0.1, 1.4], xlabel: "n", ylabel: "value" });
      P.rect(-1, -0.1, mp - 0.5, 1.4, { c: "red", a: 0.1 }); if (mm) P.rect(N - mm - 0.5, -0.1, N, 1.4, { c: "red", a: 0.1 });
      P.line(M.linspace(0, N - 1, N), x, { c: "faint", w: 1.5 });
      P.line(M.linspace(0, N - 1, N), yl, { c: "green", w: 2.5 }); P.line(M.linspace(0, N - 1, N), yc, { c: "orange", w: 2, dash: [5, 3] });
      P.text(1, 1.3, "red bands: samples that can be spoiled (m₊ at the start, m₋ at the end)", { size: 10, c: "red" });
      const err = yc.map((v, i) => Math.abs(v - yl[i]));
      k.set("e", Math.max(...err.slice(0, Math.max(1, mp))).toExponential(1)); k.set("ec", Math.max(...err.slice(mp, N - mm)).toExponential(1)); k.set("nf", nf);
    }
    root._redraw = draw; draw();
    UI.legend(root, [["faint", "data x (step near the end)"], ["green", "linear convolution (truth)"], ["orange", "IDFT{X·H}, circular"]]);
    UI.note(root, "With N_fft = N the high values at the end of the record leak into the first m₊ outputs (and, for a centred kernel, the low start leaks into the last m₋). A buffer of max(m₊, m₋) zeros is already enough.");
  };

  /* 4. overlap-save / overlap-add scheme + cost curve */
  W.olsola = (root) => {
    UI.head(root, "BLOCK CONVOLUTION · OVERLAP-SAVE AND OVERLAP-ADD");
    const k = UI.kpis(root, [["B", "new samples per block B"], ["nb", "blocks"], ["c", "cost / sample ∝ N_fft log₂N_fft / B"], ["e", "max error vs direct"]]);
    const A = UI.canvas(root, 300), Bc = UI.canvas(root, 200); const c = UI.ctrls(root);
    let meth = "save", q = 5, Mk = 9;
    UI.select(c, { label: "method", options: [["save", "overlap-save"], ["add", "overlap-add"]], on: (v) => { meth = v; draw(); } });
    UI.slider(c, { label: "N_fft = 2^q", min: 4, max: 7, step: 1, value: q, fmt: (v) => 1 << v, on: (v) => { q = v; draw(); } });
    UI.slider(c, { label: "kernel length M", min: 3, max: 15, step: 2, value: Mk, on: (v) => { Mk = v; draw(); } });
    function draw() {
      const nf = 1 << q, Nx = 100, Bk = nf - Mk + 1, rng = M.rng(5), x = Array.from({ length: Nx }, () => rng.normal()), h = Array.from({ length: Mk }, (_, i) => 1 / (1 + i));
      const yd = conv(x, h), Ny = Nx + Mk - 1, H = h.concat(new Array(nf - Mk).fill(0)), Hi = new Array(nf).fill(0); M.fft(H, Hi);
      const y = new Array(Ny + nf).fill(0), blocks = [];
      if (meth === "save") {
        const xp = new Array(Mk - 1).fill(0).concat(x, new Array(nf + Bk).fill(0));
        for (let s = 0; s < Ny; s += Bk) { const re = xp.slice(s, s + nf), im = new Array(nf).fill(0); M.fft(re, im); for (let i = 0; i < nf; i++) { const a = re[i], b = im[i]; re[i] = a * H[i] - b * Hi[i]; im[i] = a * Hi[i] + b * H[i]; } M.fft(re, im, true); const n = Math.min(Bk, Ny - s); for (let i = 0; i < n; i++) y[s + i] = re[Mk - 1 + i]; blocks.push([s - (Mk - 1), s - (Mk - 1) + nf, s, s + n]); }
      } else {
        for (let s = 0; s < Nx; s += Bk) { const re = x.slice(s, s + Bk); while (re.length < nf) re.push(0); const im = new Array(nf).fill(0); M.fft(re, im); for (let i = 0; i < nf; i++) { const a = re[i], b = im[i]; re[i] = a * H[i] - b * Hi[i]; im[i] = a * Hi[i] + b * H[i]; } M.fft(re, im, true); for (let i = 0; i < nf; i++) y[s + i] += re[i]; blocks.push([s, Math.min(s + Bk, Nx), s, s + nf]); }
      }
      const err = Math.max(...yd.map((v, i) => Math.abs(v - y[i])));
      const nb = blocks.length; A.begin({ xlim: [-(Mk + 2), Ny + nf / 2], ylim: [-0.5, nb * 2 + 0.6], xlabel: "sample index", ylabel: "" });
      blocks.forEach(([a, b, oa, ob], j) => {
        const yy = (nb - 1 - j) * 2 + 1;
        A.rect(a, yy + 0.1, b, yy + 0.7, { c: "blue", a: 0.35, stroke: "blue", w: 1 });
        if (meth === "save") { A.rect(a, yy + 0.1, a + Mk - 1, yy + 0.7, { c: "slate", a: 0.35 }); A.rect(oa, yy - 0.7, ob, yy - 0.1, { c: "green", a: 0.5 }); }
        else { A.rect(oa, yy - 0.7, ob, yy - 0.1, { c: "green", a: 0.35 }); A.rect(oa + Bk, yy - 0.7, ob, yy - 0.1, { c: "orange", a: 0.55 }); }
        A.text(-Mk - 1.5, yy + 0.25, "block " + (j + 1), { size: 9.5, c: "muted" });
      });
      A.vline(0, { c: "faint" }); A.vline(Nx, { c: "faint", label: "end of x" });
      // cost curve
      const qs = M.linspace(2, 16, 15).map(Math.round), Mc = 101;
      Bc.begin({ xlim: [64, 70000], ylim: [5, 60], xlog: true, ylog: true, xlabel: "N_fft (cost curve for M = 101)", ylabel: "cost / output sample" });
      const pts = qs.map((qq) => [1 << qq, (1 << qq) > Mc ? ((1 << qq) * qq) / ((1 << qq) - Mc + 1) : NaN]).filter((p) => isFinite(p[1]));
      Bc.line(pts.map((p) => p[0]), pts.map((p) => p[1]), { c: "purple", w: 2 }); Bc.pts(pts.map((p) => p[0]), pts.map((p) => p[1]), { c: "purple", r: 3 });
      Bc.rect(4 * Mc, 5, 8 * Mc, 60, { c: "green", a: 0.12 });
      k.set("B", Bk); k.set("nb", nb); k.set("c", (nf * q / Bk).toFixed(1)); k.set("e", err.toExponential(1));
    }
    root._redraw = draw; draw();
    UI.legend(root, [["blue", "input block (length N_fft or B)"], ["slate", "O&S: wrapped part, discarded"], ["green", "output kept / full block output"], ["orange", "O&A: tail added to the next block"]]);
  };

  /* 5. deconvolution */
  W.deconv = (root) => {
    UI.head(root, "DECONVOLUTION · INVERSE, TIKHONOV, WIENER");
    const k = UI.kpis(root, [["b", "rms error: blurred"], ["e", "rms error: estimate"], ["l", "best λ (scan)"]]);
    const A = UI.canvas(root, 240), Bc = UI.canvas(root, 210); const c = UI.ctrls(root);
    let meth = "wiener", lg = -3, sig = 6, noise = 0.02;
    UI.select(c, { label: "method", options: [["wiener", "Wiener, S_n read from the data"], ["tik", "Tikhonov, constant λ"], ["inv", "plain inverse Y/H"]], on: (v) => { meth = v; draw(); } });
    UI.slider(c, { label: "log₁₀ λ (Tikhonov)", min: -8, max: 0, step: 0.1, value: lg, fmt: (v) => v.toFixed(1), on: (v) => { lg = v; draw(); } });
    UI.slider(c, { label: "blur σ [samples]", min: 1, max: 12, step: 0.5, value: sig, on: (v) => { sig = v; draw(); } });
    UI.slider(c, { label: "noise σ", min: 0.001, max: 0.1, step: 0.001, value: noise, fmt: (v) => v.toFixed(3), on: (v) => { noise = v; draw(); } });
    const N = 512, rng = M.rng(11); const truth = (() => { let v = 0; const out = []; for (let i = 0; i < N; i++) { v = 0.97 * v + 0.25 * rng.normal(); out.push(v); } for (const [p, a] of [[120, 3], [300, -2.5], [305, 2.5], [430, 2]]) out[p] += a; return out; })();
    const nz = Array.from({ length: N }, () => rng.normal());
    function draw() {
      const g = Array.from({ length: N }, (_, i) => { const d = Math.min(i, N - i); return Math.exp(-0.5 * (d / sig) ** 2); }), gs = M.sum(g);
      const Hr = g.map((v) => v / gs), Hi = new Array(N).fill(0); M.fft(Hr, Hi);
      const Xr = truth.slice(), Xi = new Array(N).fill(0); M.fft(Xr, Xi);
      const Yr = Xr.map((a, i) => a * Hr[i] - Xi[i] * Hi[i]), Yi = Xr.map((a, i) => a * Hi[i] + Xi[i] * Hr[i]);
      const yt = Yr.slice(), yi = Yi.slice(); M.fft(yt, yi, true); const y = yt.map((v, i) => v + noise * nz[i]);
      const Rr = y.slice(), Ri = new Array(N).fill(0); M.fft(Rr, Ri);
      const P = Rr.map((a, i) => a * a + Ri[i] * Ri[i]), H2 = Hr.map((a, i) => a * a + Hi[i] * Hi[i]);
      const fl = P.slice(N / 4, N / 2).sort((a, b) => a - b), SnHat = fl[Math.floor(fl.length / 2)] / Math.log(2);
      const solve = (wfun) => { const er = new Array(N), ei = new Array(N); for (let i = 0; i < N; i++) { const w = wfun(i), dr = Hr[i] * w, di = -Hi[i] * w; er[i] = Rr[i] * dr - Ri[i] * di; ei[i] = Rr[i] * di + Ri[i] * dr; } M.fft(er, ei, true); return er; };
      // smooth |Y|^2 (running mean over 9 bins); the signal "sinks into the floor" at the first frequency where Ps < 2 Ŝn:
      // above it Φ = 0 (a single cutoff, otherwise noise bins with Φ > 0 would be divided by a tiny |H|²)
      const Ps = P.map((_, i) => { let sm = 0; for (let d = -4; d <= 4; d++) sm += P[(i + d + N) % N]; return sm / 9; });
      let kc = N / 2; for (let i = 1; i <= N / 2; i++) if (Ps[i] < 2 * SnHat) { kc = i; break; }
      const phi = Ps.map((p, i) => (Math.min(i, N - i) >= kc ? 0 : Math.max(0, Math.min(1, (p - SnHat) / p))));
      let est;
      if (meth === "inv") est = solve((i) => 1 / Math.max(1e-300, H2[i]));
      else if (meth === "tik") est = solve((i) => 1 / (H2[i] + Math.pow(10, lg)));
      else est = solve((i) => phi[i] / Math.max(1e-300, H2[i]));
      let best = [Infinity, 0]; for (let l = -8; l <= 0; l += 0.25) { const e = rms(solve((i) => 1 / (H2[i] + Math.pow(10, l))).map((v, i) => v - truth[i])); if (e < best[0]) best = [e, l]; }
      const ii = M.linspace(0, N - 1, N), lim = Math.max(...truth.map(Math.abs)) * 1.4;
      A.begin({ xlim: [0, N], ylim: [-lim, lim], xlabel: "sample", ylabel: "signal" });
      A.line(ii, truth, { c: "ink", w: 1.4 }); A.line(ii, y, { c: "orange", w: 1.6 }); A.line(ii, est.map((v) => Math.max(-3 * lim, Math.min(3 * lim, v))), { c: "purple", w: 1.4 });
      const fq = ii.slice(1, N / 2).map((i) => i / N);
      Bc.begin({ xlim: [1 / N, 0.5], ylim: [1e-8, 1e5], xlog: true, ylog: true, xlabel: "frequency [cycles/sample]", ylabel: "power" });
      Bc.line(fq, P.slice(1, N / 2), { c: "orange", w: 1 }); Bc.line(fq, H2.slice(1, N / 2).map((v) => v * 1e3), { c: "teal", w: 1.5, dash: [4, 3] }); Bc.hline(SnHat, { c: "red", label: "Ŝn" });
      if (meth === "wiener") Bc.line(fq, phi.slice(1, N / 2).map((v) => Math.max(1e-8, v * 1e4)), { c: "purple", w: 2 });
      k.set("b", rms(y.map((v, i) => v - truth[i])).toFixed(3)); const e = rms(est.map((v, i) => v - truth[i])); k.set("e", e > 1e3 ? e.toExponential(1) : e.toFixed(3)); k.set("l", "10^" + best[1].toFixed(2) + " → " + best[0].toFixed(3));
    }
    root._redraw = draw; draw();
    UI.legend(root, [["ink", "truth"], ["orange", "observed y = h ∗ x + n  /  |Y|²"], ["purple", "estimate  /  Φ × 10⁴"], ["teal", "|H|² × 10³"]]);
    UI.note(root, "The inverse divides by |H|² as small as 10⁻³⁰ at high f: noise is amplified beyond recognition. Wiener reads the noise floor Ŝn from the flat high-frequency part of |Y|², smooths |Y|² and sets Φ = 0 above the frequency where it sinks into the floor — no tuning, and close to the best Tikhonov λ found by scanning against the truth (impossible with real data).");
  };

  /* 6. causal delay */
  W.delay = (root) => {
    UI.head(root, "CENTRED vs CAUSAL FILTERS · DELAY AND DISTORTION");
    const k = UI.kpis(root, [["c", "centroid: input"], ["z", "centred Hann"], ["h", "causal Hann"], ["r", "causal RC"]]);
    const P = UI.canvas(root, 280); const c = UI.ctrls(root);
    let Mh = 21, tau = 6, wp = 4;
    UI.slider(c, { label: "Hann length M", min: 5, max: 41, step: 2, value: Mh, on: (v) => { Mh = v; draw(); } });
    UI.slider(c, { label: "RC time constant τ [samples]", min: 1, max: 20, step: 0.5, value: tau, on: (v) => { tau = v; draw(); } });
    UI.slider(c, { label: "input pulse width [samples]", min: 1, max: 15, step: 0.5, value: wp, on: (v) => { wp = v; draw(); } });
    function draw() {
      const N = 220, t0 = 80, x = Array.from({ length: N }, (_, i) => Math.exp(-0.5 * ((i - t0) / wp) ** 2));
      const hh = Array.from({ length: Mh }, (_, i) => Math.sin(Math.PI * (i + 1) / (Mh + 1)) ** 2), hs = M.sum(hh); hh.forEach((v, i) => (hh[i] = v / hs));
      const yc = conv(x, hh), m = (Mh - 1) / 2, yz = yc.slice(m, m + N), ycs = yc.slice(0, N);
      const a = Math.exp(-1 / tau), yr = []; let s = 0; for (let i = 0; i < N; i++) { s = a * s + (1 - a) * x[i]; yr.push(s); }
      const cen = (y) => M.sum(y.map((v, i) => v * i)) / M.sum(y), ii = M.linspace(0, N - 1, N);
      P.begin({ xlim: [40, 180], ylim: [0, 1.05], xlabel: "sample", ylabel: "amplitude" });
      [[x, "ink"], [yz, "green"], [ycs, "orange"], [yr, "red"]].forEach(([y, cc]) => { P.line(ii, y, { c: cc, w: 2 }); P.vline(cen(y), { c: cc, w: 1 }); });
      k.set("c", cen(x).toFixed(1)); k.set("z", cen(yz).toFixed(1)); k.set("h", cen(ycs).toFixed(1) + "  (+" + m + ")"); k.set("r", cen(yr).toFixed(1) + "  (+" + (cen(yr) - cen(x)).toFixed(1) + ")");
    }
    root._redraw = draw; draw();
    UI.legend(root, [["ink", "input pulse"], ["green", "centred Hann (zero phase)"], ["orange", "causal Hann (pure delay (M−1)/2)"], ["red", "RC / one-pole IIR (skewed tail)"]]);
    UI.note(root, "Vertical lines: centroids. The causal Hann output is the centred one shifted by exactly (M−1)/2 — same shape. The RC output has a long tail: its centroid moves by ≈ τ but its peak moves by a different amount depending on the pulse width — that is phase distortion.");
  };

  /* 7. Toeplitz / circulant and its eigenvalues */
  W.toeplitz = (root) => {
    UI.head(root, "THE CONVOLUTION MATRIX AND ITS EIGENVALUES");
    const k = UI.kpis(root, [["cn", "condition number max|H_k| / min|H_k|"], ["mn", "min |H_k|"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 280, { pad: [10, 12, 34, 40] }), Bc = UI.canvas(box, 280); const c = UI.ctrls(root);
    let circ = true, sig = 1.5, kind = "gauss";
    UI.select(c, { label: "matrix", options: [["1", "circulant (circular convolution)"], ["0", "Toeplitz (linear, truncated)"]], on: (v) => { circ = v === "1"; draw(); } });
    UI.select(c, { label: "kernel", options: [["gauss", "Gaussian blur"], ["box", "boxcar"], ["diff", "derivative [1, −1]"]], on: (v) => { kind = v; draw(); } });
    UI.slider(c, { label: "width", min: 0.5, max: 4, step: 0.1, value: sig, fmt: (v) => v.toFixed(1), on: (v) => { sig = v; draw(); } });
    function draw() {
      const N = 32, h = new Array(N).fill(0);
      if (kind === "gauss") for (let j = -8; j <= 8; j++) h[(j + N) % N] = Math.exp(-0.5 * (j / sig) ** 2);
      else if (kind === "box") { const w = Math.max(1, Math.round(sig * 2)); for (let j = -Math.floor(w / 2); j <= Math.floor(w / 2); j++) h[(j + N) % N] = 1; }
      else { h[0] = 1; h[1] = -1; }
      const s = kind === "diff" ? 1 : M.sum(h); h.forEach((v, i) => (h[i] = v / s));
      const Mx = new Array(N * N);
      for (let r = 0; r < N; r++) for (let cc = 0; cc < N; cc++) { const d = r - cc; Mx[r * N + cc] = circ ? h[((d % N) + N) % N] : Math.abs(d) <= 8 && (d >= 0 ? true : true) ? (d >= 0 ? h[d] : h[N + d]) : 0; }
      if (!circ) for (let r = 0; r < N; r++) for (let cc = 0; cc < N; cc++) { const d = r - cc; if (Math.abs(d) > 8) Mx[r * N + cc] = 0; }
      const mx = Math.max(...Mx.map(Math.abs));
      A.begin({ xlim: [0, N], ylim: [0, N], xlabel: "column", ylabel: "row (0 at top)" }); A.image(Mx, N, N, 0, N, 0, N, { cmap: "div", lo: -mx, hi: mx });
      const re = h.slice(), im = new Array(N).fill(0); M.fft(re, im); const ab = re.map((v, i) => Math.hypot(v, im[i]));
      Bc.begin({ xlim: [0, N - 1], ylim: [1e-12, 3], ylog: true, xlabel: "k (Fourier mode)", ylabel: "|H_k| = |eigenvalue|" });
      Bc.pts(M.linspace(0, N - 1, N), ab.map((v) => Math.max(1e-12, v)), { c: "purple", r: 4 });
      const mn = Math.min(...ab); k.set("cn", mn < 1e-14 ? "∞ (singular)" : M.fmt(Math.max(...ab) / mn, 2)); k.set("mn", mn.toExponential(1));
    }
    root._redraw = draw; draw();
    UI.note(root, "Circulant: every row is the previous one shifted cyclically — the corners show the wrap-around. Its eigenvalues are exactly the DFT of h. A wider Gaussian blur pushes the high-k eigenvalues towards 10⁻¹⁰: deconvolution must divide by them. The derivative kernel has H₀ = 0: the mean is lost and cannot be recovered.");
  };
})();
