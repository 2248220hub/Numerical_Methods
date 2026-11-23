/* Lesson 8 — the FFT, 2D transforms, Fourier optics, notch, STFT: symbols and widgets */
window.SYMBOLS = {
  Xk: { n: "X[k] — DFT output", t: "complex", s: "The k-th coefficient of the N-point DFT, k = 0 … N−1.", py: "X = np.fft.fft(x)", f: "call fftw_execute_dft(plan, x, X)" },
  Ek: { n: "E[k] — DFT of the even samples", t: "complex, period N/2", s: "N/2-point DFT of e[m] = x[2m]. Computed recursively by the same algorithm.", py: "E = fft(x[0::2])", f: "call fft(x(1::2), E)   ! Fortran is 1-based: x(1), x(3) … are the even-index samples" },
  Ok: { n: "O[k] — DFT of the odd samples", t: "complex, period N/2", s: "N/2-point DFT of o[m] = x[2m+1]; it enters multiplied by the twiddle W<sub>N</sub><sup>k</sup>.", py: "O = fft(x[1::2])", f: "call fft(x(2::2), O)" },
  WN: { n: "W<sub>N</sub><sup>k</sup> — twiddle factor", t: "complex, |W| = 1", s: "e<sup>−j2πk/N</sup>. W<sub>N</sub>² = W<sub>N/2</sub> (the halves are DFTs) and W<sub>N</sub><sup>k+N/2</sup> = −W<sub>N</sub><sup>k</sup> (one product serves two outputs).", py: "w = np.exp(-2j*np.pi*np.arange(N//2)/N)", f: "w = [(exp(cmplx(0.0_dp, -2*pi*k/N, dp)), k = 0, N/2-1)]" }
};

(function () {
  const W = (window.WIDGETS = {});
  const rev = (i, q) => { let r = 0; for (let b = 0; b < q; b++) if (i & (1 << b)) r |= 1 << (q - 1 - b); return r; };

  /* 1. the decimation tree */
  W.opcount = (root) => {
    UI.head(root, "DECIMATION IN TIME · SPLIT UNTIL ONE SAMPLE IS LEFT");
    const k = UI.kpis(root, [["n", "N"], ["d", "direct N²"], ["f", "FFT (N/2) log₂N"], ["s", "speed-up"]]);
    const P = UI.canvas(root, 300); const c = UI.ctrls(root); let q = 3;
    UI.select(c, { label: "N", options: [["3", "8"], ["4", "16"], ["2", "4"]], on: (v) => { q = +v; draw(); } });
    function draw() {
      const N = 1 << q;
      P.begin({ xlim: [0, N], ylim: [-0.6, q + 0.6], xlabel: "each row: the index sets after one more even/odd split", ylabel: "split level" });
      for (let L = 0; L <= q; L++) {
        const nb = 1 << L, sz = N / nb;
        for (let b = 0; b < nb; b++) {
          // block b at level L holds the indices with the same L lowest bits, in bit-reversed block order
          const low = rev(b, L), idx = []; for (let i = 0; i < N; i++) if ((i & (nb - 1)) === low) idx.push(i);
          const y = q - L; P.rect(b * sz + 0.08, y - 0.32, (b + 1) * sz - 0.08, y + 0.32, { c: L === q ? "orange" : L === 0 ? "blue" : "purple", a: 0.18, stroke: "line", w: 1 });
          P.text(b * sz + sz / 2, y + 0.12, idx.join(" "), { align: "center", size: N > 8 ? 9.5 : 12, c: "ink" });
        }
        P.text(0.1, q - L + 0.5, L === 0 ? "N-point DFT" : L === q ? "1-point DFTs: x[n] itself" : `${1 << L} DFTs of ${N >> L} points`, { size: 10, c: "muted" });
      }
      k.set("n", N); k.set("d", N * N); k.set("f", N / 2 * q); k.set("s", (N * N / (N / 2 * q)).toFixed(1) + "×");
    }
    root._redraw = draw; draw();
    UI.note(root, "Read the bottom row left to right: it is the bit-reversed order (for N = 8: 0 4 2 6 1 5 3 7). The FFT reads the data in that order and climbs back up, combining pairs of blocks with butterflies: log₂N levels of N/2 butterflies each.");
  };

  /* 2. butterfly graph, stage by stage */
  W.butterfly = (root) => {
    UI.head(root, "THE 8-POINT FFT SIGNAL-FLOW GRAPH · STEP THROUGH THE STAGES");
    const k = UI.kpis(root, [["s", "stage"], ["m", "block size m"], ["d", "pair distance m/2"], ["e", "max |X − DFT| so far"]]);
    const P = UI.canvas(root, 380); const b = UI.btns(root);
    const N = 8, q = 3; let stage = 0;
    const x = [1, 2, 0, -1, 3, 0.5, -2, 1];
    const val = []; // values after each stage
    (() => { let re = Array.from({ length: N }, (_, i) => x[rev(i, q)]), im = new Array(N).fill(0); val.push([re.slice(), im.slice()]);
      for (let s = 1, m = 2; s <= q; s++, m *= 2) { for (let kk = 0; kk < N; kk += m) for (let j = 0; j < m / 2; j++) { const th = -2 * Math.PI * j / m, wr = Math.cos(th), wi = Math.sin(th), a = kk + j, c2 = a + m / 2, tr = wr * re[c2] - wi * im[c2], ti = wr * im[c2] + wi * re[c2]; re[c2] = re[a] - tr; im[c2] = im[a] - ti; re[a] += tr; im[a] += ti; } val.push([re.slice(), im.slice()]); } })();
    const ref = Array.from({ length: N }, (_, kk) => { let r = 0, i = 0; x.forEach((v, n) => { r += v * Math.cos(-2 * Math.PI * kk * n / N); i += v * Math.sin(-2 * Math.PI * kk * n / N); }); return [r, i]; });
    UI.button(b, "◀ back", () => { stage = Math.max(0, stage - 1); draw(); });
    UI.button(b, "next stage ▶", () => { stage = Math.min(q, stage + 1); draw(); }, "primary");
    UI.button(b, "reset", () => { stage = 0; draw(); });
    function draw() {
      P.begin({ xlim: [-1.1, q + 1.2], ylim: [-0.7, N - 0.3], xlabel: "stage →", ylabel: "" });
      const Y = (r) => N - 1 - r;
      for (let s = 1, m = 2; s <= q; s++, m *= 2) {
        const on = s === stage, done = s < stage;
        for (let kk = 0; kk < N; kk += m) for (let j = 0; j < m / 2; j++) {
          const a = kk + j, c2 = a + m / 2, cc = on ? "orange" : done ? "blue" : "line", w = on ? 2.4 : 1.2;
          P.seg(s - 1, Y(a), s, Y(a), { c: cc, w }); P.seg(s - 1, Y(c2), s, Y(c2), { c: cc, w });
          P.seg(s - 1, Y(a), s, Y(c2), { c: cc, w }); P.seg(s - 1, Y(c2), s, Y(a), { c: cc, w });
          if (on && j > 0) P.text(s - 0.92, Y(c2) + 0.28, `W${m}^${j}`, { size: 10, c: "red" });
        }
      }
      const [re, im] = val[stage];
      for (let s = 0; s <= q; s++) for (let r = 0; r < N; r++) P.pts([s], [Y(r)], { c: s <= stage ? "ink" : "faint", r: 4 });
      for (let r = 0; r < N; r++) {
        P.text(-0.15, Y(r) - 0.12, `x[${rev(r, q)}]`, { align: "right", size: 11.5 });
        P.text(stage + 0.1, Y(r) + 0.36, `${re[r].toFixed(2)}${im[r] >= 0 ? "+" : "−"}${Math.abs(im[r]).toFixed(2)}j`, { size: 9.5, c: "purple", bg: "paper" });
        P.text(q + 0.15, Y(r) - 0.12, `X[${r}]`, { size: 11.5, c: stage === q ? "green" : "faint" });
      }
      const m = 1 << Math.max(1, stage); k.set("s", stage + " / " + q); k.set("m", stage ? m : "—"); k.set("d", stage ? m / 2 : "—");
      k.set("e", stage === q ? Math.max(...ref.map(([r, i], kk) => Math.hypot(r - re[kk], i - im[kk]))).toExponential(1) : "not finished");
    }
    root._redraw = draw; draw();
    UI.note(root, "Purple numbers: the content of the 8 memory slots after each stage — the same array is overwritten (in place). Each crossing is one butterfly: (u, t) → (u + W t, u − W t). After stage 3 the slots hold X[0] … X[7] in natural order.");
  };

  /* 3. measured timings */
  W.timing8 = (root) => {
    UI.head(root, "MEASURED TIME PER TRANSFORM · ONE RUN OF fft_algorithms.py AND fft_radix2.f90");
    const P = UI.canvas(root, 280);
    const Ns = [256, 1024, 4096], data = [["direct (matrix)", "red", [0.153, 1.316, 26.77]], ["recursive", "orange", [1.31, 7.85, 19.77]], ["iterative", "olive", [0.685, 3.81, 8.77]], ["vectorised", "purple", [0.137, 0.284, 0.395]], ["Fortran radix-2", "teal", [0.011, 0.043, 0.184]], ["numpy.fft", "green", [0.023, 0.033, 0.056]]];
    UI.legend(root, data.map(([n, c]) => [c, n]));
    function draw() {
      P.begin({ xlim: [-0.5, 2.5], ylim: [0.005, 60], ylog: true, xlabel: "N = 256, 1024, 4096", ylabel: "time per FFT [ms]" });
      const w = 0.13;
      Ns.forEach((_, i) => data.forEach(([, c, t], j) => { const x0 = i - 0.4 + j * w; P.rect(x0, 0.005, x0 + w * 0.9, t[i], { c, a: 0.85 }); }));
      Ns.forEach((n, i) => P.text(i, 40, "N = " + n, { align: "center", size: 11 }));
    }
    root._redraw = draw; draw();
    UI.note(root, "Fortran radix-2 at N = 256 is estimated from the N log N trend. Absolute numbers change with the machine; what does not change: the direct DFT grows as N², interpreted loops are slow, and library FFTs (FFTW inside numpy and in Fortran) are fastest.");
  };

  /* 4. plane wave and its 2D spectrum */
  W.plane2d = (root) => {
    UI.head(root, "A PLANE WAVE AND ITS 2D SPECTRUM (fftshift, log scale)");
    const k = UI.kpis(root, [["l", "wavelength [px]"], ["a", "direction of (u, v)"], ["p", "brightest spectral pixels"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 290, { pad: [10, 12, 34, 40] }), Bc = UI.canvas(box, 290, { pad: [10, 12, 34, 40] }); const c = UI.ctrls(root);
    const N = 64; let u = 4, v = 2, frac = 0, two = false, apod = false;
    UI.slider(c, { label: "u [cycles per image, x]", min: -16, max: 16, step: 1, value: u, on: (x) => { u = x; draw(); } });
    UI.slider(c, { label: "v [cycles per image, y]", min: -16, max: 16, step: 1, value: v, on: (x) => { v = x; draw(); } });
    UI.slider(c, { label: "off-grid shift of u", min: 0, max: 0.5, step: 0.05, value: frac, fmt: (x) => x.toFixed(2), on: (x) => { frac = x; draw(); } });
    UI.select(c, { label: "extra", options: [["0", "single wave"], ["1", "+ wave (u, v) = (−10, 7), half amplitude"]], on: (x) => { two = x === "1"; draw(); } });
    UI.select(c, { label: "edge taper", options: [["0", "none"], ["1", "Hann apodisation"]], on: (x) => { apod = x === "1"; draw(); } });
    function draw() {
      const re = new Array(N * N), im = new Array(N * N).fill(0);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        let f = Math.cos(2 * Math.PI * ((u + frac) * x + v * y) / N) + (two ? 0.5 * Math.cos(2 * Math.PI * (-10 * x + 7 * y) / N) : 0);
        if (apod) f *= (0.5 - 0.5 * Math.cos(2 * Math.PI * x / N)) * (0.5 - 0.5 * Math.cos(2 * Math.PI * y / N));
        re[y * N + x] = f;
      }
      A.begin({ xlim: [0, N], ylim: [0, N], xlabel: "x [px]", ylabel: "y [px]" }); A.image(re, N, N, 0, N, 0, N, { cmap: "gray" });
      M.fft2(re, im, N, N);
      const S = new Array(N * N); let mx = 0;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const ys = (y + N / 2) % N, xs = (x + N / 2) % N, p = Math.hypot(re[y * N + x], im[y * N + x]); S[(N - 1 - ys) * N + xs] = Math.log10(1e-3 + p); mx = Math.max(mx, p); }
      Bc.begin({ xlim: [-N / 2, N / 2], ylim: [-N / 2, N / 2], xlabel: "u [cycles/image]", ylabel: "v" });
      Bc.image(S, N, N, -N / 2, N / 2, -N / 2, N / 2, { cmap: "magma", lo: Math.log10(mx) - 4, hi: Math.log10(mx) });
      Bc.seg(0, 0, u + frac, v, { c: "teal", w: 1.5 });
      const L = N / Math.hypot(u + frac, v);
      k.set("l", isFinite(L) ? L.toFixed(1) : "∞ (constant)"); k.set("a", (Math.atan2(v, u + frac) * 180 / Math.PI).toFixed(0) + "°");
      k.set("p", S.filter((s) => s > Math.log10(mx) - 1).length);
    }
    root._redraw = draw; draw();
    UI.note(root, "Integer (u, v): exactly two bright pixels at ±(u, v) — the wave fits the periodic box. Shift u off the grid: leakage spreads along the u axis (the 1D story of Lecture 7, now in 2D) and the edge discontinuity makes a cross. The Hann taper suppresses it. Image rows are drawn with y increasing upwards.");
  };

  /* 5. pupil → PSF */
  W.pupil = (root) => {
    UI.head(root, "PUPIL → PSF · |FFT|² OF A CIRCULAR APERTURE");
    const k = UI.kpis(root, [["z", "first dark ring (measured)"], ["c", "Strehl-like peak (vs clear)"], ["e", "energy inside 1.22 λ/D"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 280, { pad: [10, 12, 34, 40] }), Bc = UI.canvas(box, 280); const c = UI.ctrls(root);
    const N = 128; let R = 16, eps = 0, spider = false;
    UI.slider(c, { label: "pupil radius [px] in a 128 grid (padding N/2R)", min: 6, max: 40, step: 1, value: R, on: (x) => { R = x; draw(); } });
    UI.slider(c, { label: "central obstruction ε", min: 0, max: 0.6, step: 0.05, value: eps, fmt: (x) => x.toFixed(2), on: (x) => { eps = x; draw(); } });
    UI.select(c, { label: "spider vanes", options: [["0", "none"], ["1", "4 vanes"]], on: (x) => { spider = x === "1"; draw(); } });
    function psf(R0, e0, sp) {
      const re = new Array(N * N).fill(0), im = new Array(N * N).fill(0);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const dx = x - N / 2 + 0.5, dy = y - N / 2 + 0.5, r = Math.hypot(dx, dy); if (r <= R0 && r >= e0 * R0 && !(sp && (Math.abs(dx) < 0.8 || Math.abs(dy) < 0.8))) re[y * N + x] = 1; }
      M.fft2(re, im, N, N); const P2 = new Array(N * N);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const ys = (y + N / 2) % N, xs = (x + N / 2) % N; P2[(N - 1 - ys) * N + xs] = re[y * N + x] ** 2 + im[y * N + x] ** 2; }
      return P2;
    }
    function draw() {
      const P2 = psf(R, eps, spider), Pc = psf(R, 0, false), pk = Math.max(...P2), pkc = Math.max(...Pc), tot = M.sum(P2), unit = 2 * R / N; // PSF pixel in λ/D
      A.begin({ xlim: [-N / 2 * unit, N / 2 * unit], ylim: [-N / 2 * unit, N / 2 * unit], xlabel: "θ [λ/D]", ylabel: "θ [λ/D]" });
      A.image(P2.map((p) => Math.log10(p / pk + 1e-7)), N, N, -N / 2 * unit, N / 2 * unit, -N / 2 * unit, N / 2 * unit, { cmap: "magma", lo: -5, hi: 0 });
      // radial profile along +x
      const row = N / 2 - 1, xs = [], ys = [];
      for (let x = N / 2; x < N; x++) { xs.push((x - N / 2) * unit); ys.push(P2[row * N + x] / pk); }
      const th = M.linspace(0.001, Math.min(6, N / 2 * unit), 400), J1 = (z) => { let s = 0; for (let m = 0; m < 30; m++) { let t = Math.pow(z / 2, 2 * m + 1); for (let i = 1; i <= m; i++) t /= i * i; t /= m + 1; s += (m % 2 ? -1 : 1) * t; if (Math.abs(t) < 1e-14) break; } return s; };
      const airy = th.map((t) => { const z = Math.PI * t; return z > 18 ? NaN : (2 * J1(z) / z) ** 2; });
      Bc.begin({ xlim: [0, Math.min(6, N / 2 * unit)], ylim: [1e-6, 1.5], ylog: true, xlabel: "θ [λ/D]", ylabel: "PSF / peak" });
      Bc.line(th, airy, { c: "faint", w: 2, dash: [5, 4] }); Bc.line(xs, ys, { c: "purple", w: 2 }); Bc.pts(xs, ys, { c: "purple", r: 2.5 }); Bc.vline(1.22, { c: "red", label: "1.22" });
      let z = NaN; for (let i = 1; i < ys.length - 1; i++) if (ys[i] < ys[i - 1] && ys[i] <= ys[i + 1]) { z = xs[i]; break; }
      let ein = 0; for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (Math.hypot((x - N / 2) * unit, (y - N / 2 + 1) * unit) <= 1.22) ein += P2[y * N + x];
      k.set("z", isFinite(z) ? z.toFixed(2) + " λ/D (sampled every " + unit.toFixed(2) + ")" : "—");
      k.set("c", (pk / pkc).toFixed(2)); k.set("e", (100 * ein / tot).toFixed(0) + "%");
    }
    root._redraw = draw; draw();
    UI.note(root, "Grey dashes: the analytic Airy profile [2J₁(πθ)/(πθ)]². A small pupil in a big grid = heavy zero padding = fine sampling of the PSF. An obstruction moves energy from the core into the rings (≈ 84% inside the first dark ring for a clear pupil); vanes add diffraction spikes.");
  };

  /* 6. notch filter on a striped image */
  W.notch = (root) => {
    UI.head(root, "DESTRIPING WITH A NOTCH FILTER");
    const k = UI.kpis(root, [["b", "rms error before"], ["a", "rms error after"], ["s", "spikes notched"]]);
    const r1 = UI.el("div", "split"), r2 = UI.el("div", "split"); root.append(r1);
    const pd = { pad: [8, 10, 30, 34] };
    const A = UI.canvas(r1, 230, pd), Sx = UI.canvas(r1, 230, pd); root.append(r2); const Cc = UI.canvas(r2, 230, pd), D = UI.canvas(r2, 230, pd);
    const c = UI.ctrls(root); const N = 64, rng = M.rng(7); let on = "1", sig = 1.5, amp = 0.6;
    UI.select(c, { label: "notch", options: [["1", "on (spike + conjugate)"], ["h", "only one of the pair"], ["0", "off"]], on: (x) => { on = x; draw(); } });
    UI.slider(c, { label: "Gaussian notch width σ [freq px]", min: 0.5, max: 6, step: 0.25, value: sig, on: (x) => { sig = x; draw(); } });
    UI.slider(c, { label: "stripe amplitude", min: 0, max: 1.5, step: 0.05, value: amp, on: (x) => { amp = x; draw(); } });
    const truth = new Array(N * N).fill(0);
    const craters = Array.from({ length: 14 }, () => [rng() * N, rng() * N, 2 + 5 * rng(), 0.4 + rng()]);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) truth[y * N + x] = 0.3 * Math.sin(x / 9) * Math.cos(y / 13) + craters.reduce((s, [cx, cy, r, a]) => { const d = Math.hypot(x - cx, y - cy) / r; return s + a * (d < 1 ? -0.6 * (1 - d * d) : 0.25 * Math.exp(-((d - 1) ** 2) * 20)); }, 0);
    const spikes = [[0, 12], [9, -9]]; // (u, v) of the stripe patterns
    function draw() {
      const re = truth.map((t, i) => { const x = i % N, y = Math.floor(i / N); return t + amp * (Math.cos(2 * Math.PI * 12 * y / N) + 0.6 * Math.cos(2 * Math.PI * (9 * x - 9 * y) / N + 1)); }), im = new Array(N * N).fill(0);
      const noisy = re.slice();
      const lo = Math.min(...truth) - 0.8, hi = Math.max(...truth) + 0.8;
      A.begin({ xlim: [0, N], ylim: [0, N], title: "striped image" }); A.image(noisy, N, N, 0, N, 0, N, { cmap: "gray", lo, hi });
      M.fft2(re, im, N, N);
      const H = new Array(N * N).fill(1), wrap = (a) => ((a % N) + N) % N;
      if (on !== "0") spikes.forEach(([u, v]) => { const list = on === "1" ? [[u, v], [-u, -v]] : [[u, v]]; list.forEach(([uu, vv]) => { for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const dx = wrap(x - uu + N / 2) - N / 2, dy = wrap(y - vv + N / 2) - N / 2; H[y * N + x] *= 1 - Math.exp(-(dx * dx + dy * dy) / (2 * sig * sig)); } }); });
      const S = new Array(N * N); let mx = 0;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const p = Math.hypot(re[y * N + x], im[y * N + x]) * H[y * N + x]; mx = Math.max(mx, p); S[(N - 1 - ((y + N / 2) % N)) * N + ((x + N / 2) % N)] = Math.log10(p + 1e-3); }
      Sx.begin({ xlim: [-N / 2, N / 2], ylim: [-N / 2, N / 2], title: "|F| · H (log, centred)" }); Sx.image(S, N, N, -N / 2, N / 2, -N / 2, N / 2, { cmap: "magma", lo: Math.log10(mx) - 4, hi: Math.log10(mx) });
      for (let i = 0; i < N * N; i++) { re[i] *= H[i]; im[i] *= H[i]; }
      M.fft2(re, im, N, N, true);
      Cc.begin({ xlim: [0, N], ylim: [0, N], title: "cleaned: Re F⁻¹{F H}" }); Cc.image(re, N, N, 0, N, 0, N, { cmap: "gray", lo, hi });
      const res = re.map((v, i) => v - truth[i]);
      D.begin({ xlim: [0, N], ylim: [0, N], title: "cleaned − truth" }); D.image(res, N, N, 0, N, 0, N, { cmap: "div", lo: -0.5, hi: 0.5 });
      const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
      k.set("b", rms(noisy.map((v, i) => v - truth[i])).toFixed(3)); k.set("a", rms(res).toFixed(3)); k.set("s", on === "1" ? "2 pairs" : on === "h" ? "2 halves" : "none");
          }
    root._redraw = draw; draw();
    UI.note(root, "Two periodic patterns = two pairs of spikes. Notching only one of each pair removes half the stripe energy and leaves a complex image (we display its real part: stripes at half amplitude). Wider notches remove more of the true image near those frequencies (structure in the residual).");
  };

  /* 7. STFT */
  W.stft8 = (root) => {
    UI.head(root, "SPECTROGRAM · CHIRP + TWO CLOSE TONES + A CLICK");
    const k = UI.kpis(root, [["dt", "time resolution L/f_s"], ["df", "bin width f_s/L"], ["tf", "Δt · Δf"]]);
    const P = UI.canvas(root, 320); const c = UI.ctrls(root);
    const fs = 1024, T = 2, n = fs * T, rng = M.rng(3); let L = 128, ov = 0.5;
    const x = Array.from({ length: n }, (_, i) => { const t = i / fs; return Math.sin(2 * Math.PI * (40 * t + 60 * t * t)) + 0.6 * Math.sin(2 * Math.PI * 300 * t) + 0.6 * Math.sin(2 * Math.PI * 310 * t) + (Math.abs(t - 1.3) < 0.002 ? 6 : 0) + 0.3 * rng.normal(); });
    UI.select(c, { label: "window length L", options: [["128", "128 (125 ms)"], ["32", "32 (31 ms)"], ["64", "64"], ["256", "256"], ["512", "512 (0.5 s)"]], on: (v) => { L = +v; draw(); } });
    UI.select(c, { label: "overlap", options: [["0.5", "50%"], ["0", "0%"], ["0.75", "75%"]], on: (v) => { ov = +v; draw(); } });
    function draw() {
      const H = Math.max(1, Math.round(L * (1 - ov))), nf = Math.floor((n - L) / H) + 1, nb = L / 2, w = Array.from({ length: L }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / L));
      const Z = new Array(nf * nb);
      for (let m = 0; m < nf; m++) { const re = w.map((wi, i) => wi * x[m * H + i]), im = new Array(L).fill(0); M.fft(re, im); for (let b = 0; b < nb; b++) Z[(nb - 1 - b) * nf + m] = 10 * Math.log10(re[b] ** 2 + im[b] ** 2 + 1e-9); }
      const mx = Math.max(...Z);
      P.begin({ xlim: [0, T], ylim: [0, fs / 2], xlabel: "time [s]", ylabel: "frequency [Hz]" });
      P.image(Z, nf, nb, (L / 2) / fs - H / 2 / fs, (L / 2) / fs + (nf - 0.5) * H / fs, 0, fs / 2, { cmap: "magma", lo: mx - 45, hi: mx });
      const tt = M.linspace(0, T, 100); P.line(tt, tt.map((t) => 40 + 120 * t), { c: "teal", w: 1, dash: [4, 4] });
      k.set("dt", (1000 * L / fs).toFixed(0) + " ms"); k.set("df", (fs / L).toFixed(1) + " Hz"); k.set("tf", (L / fs * fs / L).toFixed(0) + " (fixed!)");
    }
    root._redraw = draw; draw();
    UI.note(root, "Teal dashes: the true chirp frequency 40 + 120t. The 300 and 310 Hz tones are separated only if f_s/L < 10 Hz (L ≥ 128); the click at 1.3 s is sharp only for short windows. No single L is best for everything: that is the uncertainty principle.");
  };
})();
