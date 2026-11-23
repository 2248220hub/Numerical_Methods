/* Hyperspectral project — symbols and widgets (cube viewer, sampling, destriping, response, noise, Bayes, unmixing, Monte Carlo) */
window.SYMBOLS = {
  ck: { n: "c — a class (material)", t: "label", s: "Vegetation, soil, water, roof … in the hyperspectral project.", py: "classes = np.unique(labels)", f: "! integer class labels 1..K" },
  ypix: { n: "𝐲 — the spectrum of a pixel", t: "data vector", s: "B reflectance values, one per band (or its PCA components).", py: "y = cube[row, col, :]", f: "y = cube(:, col, row)   ! band index fastest" },
  Ngauss: { n: "𝒩(m, C) — Gaussian distribution", t: "distribution", s: "Multivariate normal with mean m and covariance C.", py: "scipy.stats.multivariate_normal(m, C)", f: "! sample: m + matmul(L, z), L = chol(C), z ~ N(0,1)" },
  Pc: { n: "P(c) — class prior", t: "prior", s: "Flat (1/K) or the class frequency in the training set. Affects rare classes most.", py: "prior = np.bincount(train_labels) / train_labels.size", f: "prior = real(counts,dp)/sum(counts)" },
  Amat: { n: "A — endmember matrix", t: "B × K matrix", s: "Each column is the pure spectrum of one material. The model of a pixel is A θ.", py: "A = endmembers.T", f: "real(dp) :: A(B, K)" },
  abund: { n: "θ — abundances", t: "parameters", s: "Fraction of the pixel covered by each material (physically ≥ 0 and summing to 1).", py: "theta = Sigma @ (A.T @ (y / sig**2))", f: "theta = matmul(Sigma, matmul(transpose(A), y/sig**2))" },
  Cn: { n: "C — noise covariance", t: "B × B matrix", s: "Here diagonal: σ²<sub>b</sub> per band, larger in the water-vapour bands.", py: "Cinv = 1 / sig**2", f: "Cinv = 1.0_dp/sig**2" },
  Sig: { n: "Σ — posterior covariance of the abundances", t: "K × K matrix", s: "(AᵀC⁻¹A + P⁻¹)⁻¹. With a flat prior it is exactly the inverse Fisher matrix.", py: "Sigma = np.linalg.inv(A.T @ (A * Cinv[:, None]) + P0inv)", f: "Sigma = inverse(matmul(transpose(A), A*spread(Cinv,2,K)) + P0inv)" },
  NIR: { n: "N — near-infrared reflectance", t: "band ~850 nm", s: "Vegetation reflects strongly here (leaf cell structure).", py: "NIR = cube[..., band(wl, 850)]", f: "nir = cube(band850, :, :)" },
  Rb: { n: "R — red reflectance", t: "band ~660 nm", s: "Chlorophyll absorbs red light: vegetation is dark here.", py: "R = cube[..., band(wl, 660)]", f: "red = cube(band660, :, :)" },
  srf: { n: "SRF — spectral response function", t: "kernel along λ", s: "How much light of each wavelength a band collects: ≈ Gaussian of width FWHM. A band value is the spectrum convolved with the SRF, then sampled at the band centre.", py: "band = (W * spec).sum(1) / W.sum(1)", f: "band(j) = sum(w(:,j)*spec)/sum(w(:,j))" },
  dlam: { n: "Δλ — band spacing", t: "nm", s: "Distance between band centres: the sampling interval along wavelength.", py: "centres = np.arange(400, 2500, dlam)", f: "centres = [(400 + j*dlam, j = 0, nb-1)]" },
  fwhm: { n: "FWHM — band width", t: "nm", s: "Full width at half maximum of the SRF (σ = FWHM/2.355). FWHM ≳ Δλ acts as an anti-alias filter.", py: "s = fwhm / 2.3548", f: "s = fwhm/2.3548_dp" },
  stripe: { n: "s(x) — column offset", t: "per detector", s: "Gain/offset error of the detector that records column x in a push-broom imager: the same for every row.", py: "striped = img + s[None, :]", f: "img = img + spread(s, 1, ny)" },
  Fuv: { n: "F(u, v) — 2D spectrum of a band", t: "complex", s: "A function of x only has F ≠ 0 only on the v = 0 line.", py: "F = np.fft.fft2(img)", f: "call fftw_execute_dft(plan2d, img, F)" },
  sighat: { n: "σ̂<sub>b</sub> — estimated noise of band b", t: "reflectance units", s: "From neighbour differences: the scene is smooth on a pixel scale, the noise is not. Second differences remove gradients.", py: "sig = 1.4826*np.median(abs(d2 - np.median(d2)))/np.sqrt(6)", f: "! median absolute deviation of x(j+1) - 2x(j) + x(j-1)" }
};

(function () {
  const W = (window.WIDGETS = {});
  /* ---------------------------------------------------------------- shared synthetic scene (same physics as make_synthetic_cube.py) */
  const gs = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2), sgm = (x, x0, kk) => 1 / (1 + Math.exp(-(x - x0) / kk));
  const endmembers = (wl) => {
    const wd = wl.map((x) => 1 - 0.55 * gs(x, 1450, 45) - 0.75 * gs(x, 1940, 60));
    return [
      wl.map((x, i) => (0.04 + 0.06 * gs(x, 550, 35) + 0.46 * sgm(x, 715, 12) - 0.2 * sgm(x, 1300, 80) - 0.08 * sgm(x, 2000, 90)) * wd[i]),
      wl.map((x) => (0.08 + 0.3 * sgm(x, 900, 260) - 0.05 * gs(x, 2200, 40)) * (1 - 0.25 * gs(x, 1450, 45) - 0.35 * gs(x, 1940, 60))),
      wl.map((x) => 0.07 * Math.exp(-(x - 400) / 180) + 0.005),
      wl.map((x) => (0.28 + 0.04 * (x - 400) / 2100 - 0.04 * gs(x, 1730, 30)) * (1 - 0.15 * gs(x, 1450, 45) - 0.2 * gs(x, 1940, 60)))];
  };
  const noiseSigma = (wl) => wl.map((x) => 0.004 + 0.03 * gs(x, 1400, 40) + 0.04 * gs(x, 1900, 50) + 0.01 * (x > 2350));
  const SCENES = {};
  function scene(N = 64, nb = 80, seed = 42) {
    const key = N + "_" + nb + "_" + seed; if (SCENES[key]) return SCENES[key];
    const rng = M.rng(seed), wl = M.linspace(400, 2500, nb), E = endmembers(wl), sigB = noiseSigma(wl);
    const blobs = [0, 1, 2, 3].map(() => Array.from({ length: 9 }, () => [rng() * N, rng() * N, N / 10 + (N / 6) * rng(), rng.normal()]));
    const ab = [], lab = [], cube = [];
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const z = blobs.map((bl) => bl.reduce((s, [bx, by, r, a]) => s + a * Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * r * r)), 0));
      z[0] += 0.4; z[2] += 4 * gs(x - 0.25 * y, 0.55 * N, N / 30) - 1.3;
      z[3] += (Math.abs(x - 0.22 * N) < 0.1 * N && Math.abs(y - 0.23 * N) < 0.1 * N ? 4 : 0) + (Math.abs(x - 0.78 * N) < 0.07 * N && Math.abs(y - 0.8 * N) < 0.11 * N ? 3.5 : 0) - 1;
      const e = z.map((v) => Math.exp(2.4 * v)), se = M.sum(e), a = e.map((v) => v / se);
      ab.push(a); lab.push(a.indexOf(Math.max(...a)));
      cube.push(wl.map((_, b) => a[0] * E[0][b] + a[1] * E[1][b] + a[2] * E[2][b] + a[3] * E[3][b] + sigB[b] * rng.normal()));
    }
    return (SCENES[key] = { N, nb, wl, E, sigB, ab, lab, cube, names: ["vegetation", "soil", "water", "roof"], cols: ["green", "orange", "blue", "slate"] });
  }
  const bandOf = (wl, nm) => wl.reduce((bi, w, i) => (Math.abs(w - nm) < Math.abs(wl[bi] - nm) ? i : bi), 0);
  const rmsOf = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
  const flipRows = (img, N) => { const o = new Array(N * N); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) o[(N - 1 - y) * N + x] = img[y * N + x]; return o; };
  const imgAxes = { pad: [8, 10, 30, 34] };

  /* 1. the cube: click a pixel, slide a band */
  W.cubeview = (root) => {
    UI.head(root, "A DATA CUBE · ONE IMAGE PER BAND, ONE SPECTRUM PER PIXEL");
    const k = UI.kpis(root, [["b", "band"], ["p", "pixel (x, y)"], ["c", "dominant material"], ["a", "true abundances veg / soil / water / roof"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 300, imgAxes), S = UI.canvas(box, 300); const c = UI.ctrls(root);
    const sc = scene(); let bi = bandOf(sc.wl, 860), sel = [20, 40], view = "band";
    UI.slider(c, { label: "band", min: 0, max: sc.nb - 1, step: 1, value: bi, fmt: (v) => sc.wl[v].toFixed(0) + " nm", on: (v) => { bi = v; draw(); } });
    UI.select(c, { label: "show", options: [["band", "one band (grey)"], ["labels", "dominant material (truth)"]], on: (v) => { view = v; draw(); } });
    function draw() {
      const N = sc.N, img = view === "band" ? sc.cube.map((s) => s[bi]) : sc.lab.map((l) => l);
      A.begin({ xlim: [0, N], ylim: [0, N], xlabel: "x [px]", ylabel: "y [px] (row 0 at top)" });
      A.image(img, N, N, 0, N, 0, N, view === "band" ? { cmap: "gray" } : { cmap: "div", lo: 0, hi: 3 });
      A.rect(sel[0], N - sel[1] - 1, sel[0] + 1, N - sel[1], { c: "red", a: 0.9, stroke: "red", w: 2 });
      const i = sel[1] * N + sel[0], y = sc.cube[i];
      S.begin({ xlim: [400, 2500], ylim: [-0.02, 0.6], xlabel: "wavelength [nm]", ylabel: "reflectance" });
      [1400, 1900].forEach((w) => S.rect(w - 60, -0.02, w + 60, 0.6, { c: "blue", a: 0.08 }));
      sc.E.forEach((e, m) => S.line(sc.wl, e, { c: sc.cols[m], w: 1, dash: [4, 3] }));
      S.pts(sc.wl, y, { c: "ink", r: 2.5 }); S.vline(sc.wl[bi], { c: "red" });
      k.set("b", `${bi} · ${sc.wl[bi].toFixed(0)} nm`); k.set("p", `(${sel[0]}, ${sel[1]})`); k.set("c", sc.names[sc.lab[i]]); k.set("a", sc.ab[i].map((v) => v.toFixed(2)).join(" / "));
    }
    A.c.addEventListener("click", (e) => { const r = A.c.getBoundingClientRect(), [x, yv] = A.inv(e.clientX - r.left, e.clientY - r.top); if (x >= 0 && x < sc.N && yv >= 0 && yv < sc.N) { sel = [Math.floor(x), sc.N - 1 - Math.floor(yv)]; draw(); } });
    A.c.style.cursor = "crosshair";
    root._redraw = draw; draw();
    UI.legend(root, [["green", "vegetation"], ["orange", "soil"], ["blue", "water"], ["slate", "roof"], ["ink", "selected pixel"]]);
    UI.note(root, "Click a pixel, move the band. Vegetation is dark in the red (≈ 670 nm) and bright beyond the red edge (≈ 720 nm); water is dark everywhere beyond the visible; the shaded water-vapour bands at 1400 and 1900 nm are noisy for every material.");
  };

  /* 2. the cube as a matrix: singular values of the whitened, centred cube */
  W.linalgcube = (root) => {
    UI.head(root, "THE CUBE AS A MATRIX · SINGULAR VALUES ABOVE THE NOISE");
    const k = UI.kpis(root, [["ke", "κ(E) / κ(EᵀE)"], ["r", "σ above the noise level"], ["m", "materials (+1 for the mean)"], ["d", "rank-r denoising gain"]]);
    const P = UI.canvas(root, 300); const c = UI.ctrls(root); let boost = 1, white = true, nmat = 4;
    UI.slider(c, { label: "noise × ", min: 0.25, max: 20, step: 0.25, value: boost, on: (v) => { boost = v; draw(); } });
    UI.select(c, { label: "band weighting", options: [["1", "whitened (divide by σ_b)"], ["0", "raw reflectance"]], on: (v) => { white = v === "1"; draw(); } });
    UI.select(c, { label: "materials in the scene", options: [["4", "4 (vegetation, soil, water, roof)"], ["3", "3 (no roofs)"], ["2", "2 (vegetation, soil)"]], on: (v) => { nmat = +v; draw(); } });
    const N = 48, nb = 60, wl = M.linspace(400, 2500, nb), E = endmembers(wl), sigB = noiseSigma(wl), rng = M.rng(11);
    const fields = [0, 1, 2, 3].map((m) => Array.from({ length: 9 }, () => [rng() * N, rng() * N, N / 8 + (N / 5) * rng(), rng.normal()]));
    const nz = Array.from({ length: N * N * nb }, () => rng.normal());
    const symEig = (A0) => { const n = A0.length, A = A0.map((r) => r.slice());           // cyclic Jacobi eigenvalues of a symmetric matrix
      for (let sw = 0; sw < 50; sw++) { let off = 0; for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] ** 2; if (off < 1e-22) break;
        for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) { if (Math.abs(A[p][q]) < 1e-300) continue; const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
          for (let r = 0; r < n; r++) { const a = A[r][p], b = A[r][q]; A[r][p] = cs * a - sn * b; A[r][q] = sn * a + cs * b; }
          for (let r = 0; r < n; r++) { const a = A[p][r], b = A[q][r]; A[p][r] = cs * a - sn * b; A[q][r] = sn * a + cs * b; } } }
      return A.map((r, i) => r[i]).sort((a, b) => b - a); };
    function draw() {
      const npx = N * N, X = new Array(npx), clean = new Array(npx);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const z = fields.map((bl) => bl.reduce((s, [bx, by, r, a]) => s + a * Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * r * r)), 0)).slice(0, nmat);
        const e = z.map((v) => Math.exp(2.4 * v)), se = M.sum(e), a = e.map((v) => v / se), i = y * N + x;
        clean[i] = wl.map((_, b) => a.reduce((s, ai, m) => s + ai * E[m][b], 0));
        X[i] = clean[i].map((v, b) => (v + boost * sigB[b] * nz[i * nb + b]) / (white ? boost * sigB[b] : 0.01));
        clean[i] = clean[i].map((v, b) => v / (white ? boost * sigB[b] : 0.01));
      }
      const mu = wl.map((_, b) => X.reduce((s, r) => s + r[b], 0) / npx), muc = wl.map((_, b) => clean.reduce((s, r) => s + r[b], 0) / npx);
      const C = wl.map(() => new Array(nb).fill(0));
      for (const r of X) { const d = r.map((v, b) => v - mu[b]); for (let i = 0; i < nb; i++) for (let j = i; j < nb; j++) C[i][j] += d[i] * d[j]; }
      for (let i = 0; i < nb; i++) for (let j = 0; j < i; j++) C[i][j] = C[j][i];
      const s = symEig(C).map((v) => Math.sqrt(Math.max(v, 0))), noiseLvl = white ? Math.sqrt(npx) : NaN;
      const xs = M.linspace(1, 20, 20);
      P.begin({ xlim: [0.5, 20.5], ylim: [Math.max(1e-3, Math.min(...s.slice(0, 20)) / 3), s[0] * 3], ylog: true, xlabel: "index i", ylabel: "singular value σᵢ" });
      if (white) P.hline(noiseLvl, { c: "red", label: "noise ≈ √N_pix" });
      P.line(xs, s.slice(0, 20), { c: "blue", w: 2 }); P.pts(xs, s.slice(0, 20), { c: s.slice(0, 20).map((v) => (white && v > 1.5 * noiseLvl ? "orange" : "blue")), r: 5 });
      const r = white ? s.filter((v) => v > 1.5 * noiseLvl).length : NaN;
      const Es = E.slice(0, nmat), G = Es.map((a) => Es.map((b) => a.reduce((t, v, i) => t + v * b[i], 0)));
      const ge = symEig(G), ke = Math.sqrt(ge[0] / ge[ge.length - 1]);
      k.set("ke", `${ke.toFixed(1)} / ${(ke * ke).toFixed(0)}`); k.set("r", white ? r : "whiten first"); k.set("m", white ? r + 1 + " (true " + nmat + ")" : "—");
      k.set("d", white ? (1 / Math.sqrt(Math.max(r, 1) / nb)).toFixed(1) + "× (≈ √(bands / rank))" : "—");
    }
    root._redraw = draw; draw();
    UI.note(root, "48 × 48 pixels × 60 bands, generated like the project cube. Whitened, every pure-noise direction has σ ≈ √N_pix (red line); each material beyond the first adds one singular value above it (the mean takes one). Raise the noise: the weakest material sinks into the noise floor first. Without whitening the noisy water-vapour bands dominate and the noise level is not flat.");
  };

  /* 2. sampling a spectrum: band spacing, SRF width, grid phase, bits */
  W.hsisample = (root) => {
    UI.head(root, "SAMPLING A SPECTRUM · BAND SPACING, BAND WIDTH, GRID PHASE");
    const k = UI.kpis(root, [["d", "measured O2-A depth (true 0.60)"], ["r", "FWHM / spacing"], ["e", "reconstruction rms (650–900 nm)"], ["q", "quantisation rms / sensor σ"]]);
    const P = UI.canvas(root, 300); const c = UI.ctrls(root);
    let sp = 10, fw = 10, ph = 0, bits = 10;
    UI.slider(c, { label: "band spacing [nm]", min: 2, max: 100, step: 1, value: sp, on: (v) => { sp = v; draw(); } });
    UI.slider(c, { label: "band FWHM [nm]", min: 1, max: 80, step: 1, value: fw, on: (v) => { fw = v; draw(); } });
    UI.slider(c, { label: "grid phase (shift / spacing)", min: 0, max: 0.95, step: 0.05, value: ph, fmt: (v) => v.toFixed(2), on: (v) => { ph = v; draw(); } });
    UI.slider(c, { label: "ADC bits per band", min: 4, max: 16, step: 1, value: bits, on: (v) => { bits = v; draw(); } });
    const lam = M.linspace(600, 950, 351), veg = endmembers(lam)[0], truth = veg.map((v, i) => v * (1 - 0.6 * gs(lam[i], 760, 1.5)));
    const obs = (cs, f, spec) => cs.map((c0) => { const s = f / 2.3548; let a = 0, w = 0; lam.forEach((l, i) => { const g = gs(l, c0, s); a += g * spec[i]; w += g; }); return a / w; });
    function draw() {
      const cs = []; for (let x = 600 + ph * sp; x <= 950; x += sp) cs.push(x);
      const v = obs(cs, fw, truth), v0 = obs(cs, fw, veg), depth = Math.max(...v.map((x, i) => 1 - x / v0[i]));
      const D = 1 / Math.pow(2, bits), vq = v.map((x) => D * Math.round(x / D));
      P.begin({ xlim: [650, 900], ylim: [0, 0.6], xlabel: "wavelength [nm]", ylabel: "reflectance" });
      P.line(lam, truth, { c: "ink", w: 1.2 });
      cs.forEach((c0) => { if (c0 > 650 && c0 < 900) { const s = fw / 2.3548, xs = M.linspace(c0 - 3 * s, c0 + 3 * s, 40); P.line(xs, xs.map((x) => 0.05 * gs(x, c0, s)), { c: "purple", w: 1, a: 0.5 }); } });
      const rec = lam.map((l) => { let j = cs.findIndex((c0) => c0 > l); if (j <= 0) return NaN; const t = (l - cs[j - 1]) / (cs[j] - cs[j - 1]); return vq[j - 1] * (1 - t) + vq[j] * t; });
      P.line(lam, rec, { c: "orange", w: 1.6 }); P.pts(cs, vq, { c: "red", r: 4 });
      const err = rmsOf(lam.map((l, i) => (l > 650 && l < 900 && isFinite(rec[i]) ? rec[i] - truth[i] : 0)).filter((_, i) => lam[i] > 650 && lam[i] < 900 && isFinite(rec[i])));
      k.set("d", depth.toFixed(3)); k.set("r", (fw / sp).toFixed(2) + (fw / sp < 1 ? "  (undersampled)" : "")); k.set("e", err.toFixed(4)); k.set("q", (D / Math.sqrt(12) / 0.004).toFixed(3));
    }
    root._redraw = draw; draw();
    UI.legend(root, [["ink", "true spectrum (1 nm)"], ["purple", "band responses (SRF)"], ["red", "measured bands"], ["orange", "linear interpolation"]]);
    UI.note(root, "Spacing 30 nm with FWHM 4 nm: move the grid phase and the 3.5-nm O2 line appears with full depth or vanishes — the bands sample the spectrum without an anti-alias filter. With FWHM ≥ spacing the depth is smaller (smoothed) but stable. Quantisation: with ≥ 8 bits the rounding error of reflectance is well below the sensor noise σ ≈ 0.004.");
  };

  /* 3. destriping */
  W.destripe = (root) => {
    UI.head(root, "PUSH-BROOM STRIPES · NOTCH ON THE v = 0 LINE");
    const k = UI.kpis(root, [["b", "rms error striped"], ["a", "rms error after"], ["f", "power on v = 0 (striped)"], ["n", "rows along track"]]);
    const r1 = UI.el("div", "split"); root.append(r1); const A = UI.canvas(r1, 250, imgAxes), F = UI.canvas(r1, 250, imgAxes);
    const r2 = UI.el("div", "split"); root.append(r2); const Cc = UI.canvas(r2, 250, imgAxes), D = UI.canvas(r2, 250, imgAxes);
    const c = UI.ctrls(root); let amp = 1, meth = "v0", keep = 4, T = 4;
    UI.slider(c, { label: "stripe amplitude ×", min: 0, max: 3, step: 0.1, value: amp, on: (v) => { amp = v; draw(); } });
    UI.select(c, { label: "destriper", options: [["v0", "notch v = 0, keep |u| ≤ keep"], ["per", "periodic notch only (module of 8)"], ["none", "none"]], on: (v) => { meth = v; draw(); } });
    UI.slider(c, { label: "keep |u| ≤", min: 0, max: 16, step: 1, value: keep, on: (v) => { keep = v; draw(); } });
    UI.select(c, { label: "acquisition length", options: [["4", "256 rows (4 scenes)"], ["1", "64 rows (1 scene)"], ["8", "512 rows (8 scenes)"]], on: (v) => { T = +v; draw(); } });
    const N = 64, strips = {}, rng = M.rng(3);
    const strip = (T) => { if (strips[T]) return strips[T]; const out = []; for (let t = 0; t < T; t++) { const sc = scene(64, 80, 42 + t), b = bandOf(sc.wl, 650); sc.cube.forEach((p) => out.push(p[b])); } return (strips[T] = out); };
    const stripe0 = Array.from({ length: N }, (_, x) => 0.015 * rng.normal() + 0.02 * Math.cos(2 * Math.PI * x / 8));
    function draw() {
      const Ny = 64 * T, clean = strip(T), striped = clean.map((v, i) => v + amp * stripe0[i % N]);
      const re = striped.slice(), im = new Array(N * Ny).fill(0); M.fft2(re, im, N, Ny);
      let on0 = 0, tot = 0; for (let i = 1; i < N * Ny; i++) { const p = re[i] ** 2 + im[i] ** 2; tot += p; if (i < N) on0 += p; }
      // display spectrum: the central 64 x 64 frequencies (v in cycles per 64 rows)
      const S = new Array(N * N); let mx = 0;
      for (let yy = 0; yy < N; yy++) for (let x = 0; x < N; x++) { const v = yy - N / 2, row = ((v * T) % Ny + Ny) % Ny, p = re[row * N + x] ** 2 + im[row * N + x] ** 2; if (row || x) mx = Math.max(mx, p); S[(N - 1 - yy) * N + ((x + N / 2) % N)] = Math.log10(p + 1e-12); }
      let out;
      if (meth === "none") out = striped;
      else { const r2 = re.slice(), i2 = im.slice(); for (let x = 0; x < N; x++) { const u = x <= N / 2 ? x : x - N; const kill = meth === "v0" ? Math.abs(u) > keep : u !== 0 && Math.abs(u) % (N / 8) === 0; if (kill) { r2[x] = 0; i2[x] = 0; } } M.fft2(r2, i2, N, Ny, true); out = r2; }
      const top = (a) => a.slice(0, N * N), lo = Math.min(...top(clean)), hi = Math.max(...top(clean));
      A.begin({ xlim: [0, N], ylim: [0, N], title: "650 nm, striped (first 64 rows)" }); A.image(top(striped), N, N, 0, N, 0, N, { cmap: "gray", lo, hi });
      F.begin({ xlim: [-N / 2, N / 2], ylim: [-N / 2, N / 2], title: "log |F|², centred" }); F.image(S, N, N, -N / 2, N / 2, -N / 2, N / 2, { cmap: "magma", lo: Math.log10(mx) - 5, hi: Math.log10(mx) });
      Cc.begin({ xlim: [0, N], ylim: [0, N], title: "destriped" }); Cc.image(top(out), N, N, 0, N, 0, N, { cmap: "gray", lo, hi });
      D.begin({ xlim: [0, N], ylim: [0, N], title: "destriped − clean" }); D.image(top(out).map((v, i) => v - clean[i]), N, N, 0, N, 0, N, { cmap: "div", lo: -0.04, hi: 0.04 });
      k.set("b", rmsOf(striped.map((v, i) => v - clean[i])).toFixed(4)); k.set("a", rmsOf(out.map((v, i) => v - clean[i])).toFixed(4)); k.set("f", (100 * on0 / tot).toFixed(0) + "%"); k.set("n", Ny);
    }
    root._redraw = draw; draw();
    UI.note(root, "A pattern that depends on x only has a 2D spectrum on the line v = 0. Zeroing that line except |u| ≤ keep is the column-mean destriper written in Fourier space. It also removes the part of the scene that varies only across the track — which shrinks as the acquisition gets longer, because the scene's column means average out while the stripes do not. Try 64 rows vs 512 rows. The periodic notch removes only the 8-column module pattern.");
  };

  /* 4. spectral response and spatial PSF */
  W.srfpsf = (root) => {
    UI.head(root, "TWO CONVOLUTIONS · SPECTRAL RESPONSE AND SPATIAL PSF");
    const k = UI.kpis(root, [["d", "O2-A depth after SRF (1 nm: 0.60)"], ["b", "rms error blurred"], ["w", "rms error Wiener (from the data)"], ["i", "rms error plain inverse"]]);
    const box = UI.el("div", "split"); root.append(box);
    const S = UI.canvas(box, 260); const right = UI.el("div", "split"); box.append(right);
    const A = UI.canvas(right, 260, imgAxes), Bc = UI.canvas(right, 260, imgAxes); const c = UI.ctrls(root);
    let fw = 10, ps = 1.5, nz = 0.003;
    UI.slider(c, { label: "SRF FWHM [nm]", min: 1, max: 40, step: 1, value: fw, on: (v) => { fw = v; draw(); } });
    UI.slider(c, { label: "PSF σ [px]", min: 0.5, max: 3, step: 0.1, value: ps, on: (v) => { ps = v; draw(); } });
    UI.slider(c, { label: "noise σ", min: 0.0005, max: 0.02, step: 0.0005, value: nz, fmt: (v) => v.toFixed(4), on: (v) => { nz = v; draw(); } });
    const lam = M.linspace(680, 840, 161), veg = endmembers(lam)[0], spec = veg.map((v, i) => v * (1 - 0.6 * gs(lam[i], 760, 1.5)));
    const sc = scene(), N = sc.N, b = bandOf(sc.wl, 860), x = sc.cube.map((s) => s[b]), rn = Array.from({ length: N * N }, ((r) => () => r.normal())(M.rng(1)));
    function draw() {
      const s = fw / 2.3548, conv = lam.map((l) => { let a = 0, w = 0; lam.forEach((l2, j) => { const g = gs(l2, l, s); a += g * spec[j]; w += g; }); return a / w; });
      const conv0 = lam.map((l) => { let a = 0, w = 0; lam.forEach((l2, j) => { const g = gs(l2, l, s); a += g * veg[j]; w += g; }); return a / w; });
      S.begin({ xlim: [700, 820], ylim: [0, 0.55], xlabel: "wavelength [nm]", ylabel: "reflectance" }); S.line(lam, spec, { c: "ink", w: 1.2 }); S.line(lam, conv, { c: "red", w: 2.2 });
      // spatial: blur in Fourier space, add noise, Wiener with a floor read from the data
      const Hk = new Array(N * N); for (let yy = 0; yy < N; yy++) for (let xx = 0; xx < N; xx++) { const fy = (yy <= N / 2 ? yy : yy - N) / N, fx = (xx <= N / 2 ? xx : xx - N) / N; Hk[yy * N + xx] = Math.exp(-2 * Math.PI ** 2 * ps * ps * (fx * fx + fy * fy)); }
      const re = x.slice(), im = new Array(N * N).fill(0); M.fft2(re, im, N, N); for (let i = 0; i < N * N; i++) { re[i] *= Hk[i]; im[i] *= Hk[i]; } M.fft2(re, im, N, N, true);
      const y = re.map((v, i) => v + nz * rn[i]), Yr = y.slice(), Yi = new Array(N * N).fill(0); M.fft2(Yr, Yi, N, N);
      const P = Yr.map((v, i) => (v * v + Yi[i] * Yi[i]) / (N * N)), rr = Hk.map((_, i) => { const yy = Math.floor(i / N), xx = i % N; return Math.hypot((yy <= N / 2 ? yy : yy - N) / N, (xx <= N / 2 ? xx : xx - N) / N); });
      const fl = P.filter((_, i) => rr[i] > 0.4).sort((a, b2) => a - b2), Sn = fl[Math.floor(fl.length / 2)] / Math.LN2;
      const nb2 = 24, rad = new Array(nb2).fill(0), cnt = new Array(nb2).fill(0); P.forEach((p, i) => { const j = Math.min(nb2 - 1, Math.floor(rr[i] / 0.7072 * nb2)); rad[j] += p; cnt[j]++; });
      const Pr = rad.map((v, j) => v / Math.max(1, cnt[j])); let rc = 0.71; for (let j = 0; j < nb2; j++) if (Pr[j] < 2 * Sn) { rc = j * 0.7072 / nb2; break; }
      const wr = Yr.slice(), wi = Yi.slice(), ir = Yr.slice(), ii = Yi.slice();
      for (let i = 0; i < N * N; i++) { const Ps = Pr[Math.min(nb2 - 1, Math.floor(rr[i] / 0.7072 * nb2))], phi = rr[i] >= rc ? 0 : Math.max(0, Math.min(1, (Ps - Sn) / Ps)); wr[i] *= phi / Hk[i]; wi[i] *= phi / Hk[i]; ir[i] /= Hk[i]; ii[i] /= Hk[i]; }
      M.fft2(wr, wi, N, N, true); M.fft2(ir, ii, N, N, true);
      const lo = Math.min(...x), hi = Math.max(...x);
      A.begin({ xlim: [0, N], ylim: [0, N], title: "860 nm: blurred + noise" }); A.image(y, N, N, 0, N, 0, N, { cmap: "gray", lo, hi });
      Bc.begin({ xlim: [0, N], ylim: [0, N], title: "Wiener" }); Bc.image(wr, N, N, 0, N, 0, N, { cmap: "gray", lo, hi });
      k.set("d", Math.max(...conv.map((v, i) => 1 - v / conv0[i])).toFixed(2)); k.set("b", rmsOf(y.map((v, i) => v - x[i])).toFixed(4));
      k.set("w", rmsOf(wr.map((v, i) => v - x[i])).toFixed(4)); const ei = rmsOf(ir.map((v, i) => v - x[i])); k.set("i", ei > 10 ? ei.toExponential(1) : ei.toFixed(4));
    }
    root._redraw = draw; draw();
    UI.note(root, "Left: the 1-nm vegetation spectrum with the O2-A line (black) and what a band of the chosen FWHM records (red). Right: the 860-nm band of the 64×64 scene blurred by a Gaussian PSF. Wiener reads the noise floor from the outer part of |Y|² and stops where the signal sinks into it; the plain inverse divides noise by |H| ≈ 10⁻⁸.");
  };

  /* 5. noise per band and along track */
  W.bandnoise = (root) => {
    UI.head(root, "NOISE OF EVERY BAND · FROM THE DATA ALONE");
    const k = UI.kpis(root, [["r1", "first difference: median σ̂/σ"], ["r2", "second difference: median σ̂/σ"], ["sn", "bands with SNR < 5"], ["kn", "knee seen at"]]);
    const A = UI.canvas(root, 230), Bc = UI.canvas(root, 250); const c = UI.ctrls(root); let fk = 2, al = 1.5;
    UI.slider(c, { label: "drift knee f_k [Hz] (100 lines/s)", min: 0.1, max: 20, step: 0.1, value: fk, on: (v) => { fk = v; draw2(); } });
    UI.slider(c, { label: "drift slope α", min: 0.5, max: 3, step: 0.1, value: al, fmt: (v) => v.toFixed(1), on: (v) => { al = v; draw2(); } });
    const sc = scene(), N = sc.N, med = (a) => { const s = a.slice().sort((p, q) => p - q); return s[Math.floor(s.length / 2)]; };
    const mad = (a) => { const m = med(a); return 1.4826 * med(a.map((v) => Math.abs(v - m))); };
    const s1 = [], s2 = [], snr = [];
    for (let b = 0; b < sc.nb; b++) { const d1 = [], d2 = []; for (let y = 0; y < N; y++) for (let x = 0; x < N - 2; x++) { const a0 = sc.cube[y * N + x][b], a1 = sc.cube[y * N + x + 1][b], a2 = sc.cube[y * N + x + 2][b]; d1.push(a1 - a0); d2.push(a2 - 2 * a1 + a0); }
      s1.push(mad(d1) / Math.SQRT2); s2.push(mad(d2) / Math.sqrt(6)); snr.push(M.mean(sc.cube.map((s) => s[b])) / s2[b]); }
    function draw1() {
      A.begin({ xlim: [400, 2500], ylim: [0.002, 0.08], ylog: true, xlabel: "wavelength [nm]", ylabel: "noise σ per band" });
      A.line(sc.wl, sc.sigB, { c: "ink", w: 2.5 }); A.line(sc.wl, s1, { c: "orange", w: 1.6, dash: [5, 3] }); A.line(sc.wl, s2, { c: "green", w: 1.6 });
      k.set("r1", med(s1.map((v, i) => v / sc.sigB[i])).toFixed(3)); k.set("r2", med(s2.map((v, i) => v / sc.sigB[i])).toFixed(3)); k.set("sn", snr.filter((v) => v < 5).length + " of " + sc.nb);
    }
    function draw2() {
      const fs = 100, n = 2048, L = 256, D = 128, ndet = 24, rng = M.rng(5), s2w = 0.004 ** 2, Pm = new Array(L / 2 + 1).fill(0);
      const w = Array.from({ length: L }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / L)), U = 0.375; let K = 0;
      for (let d = 0; d < ndet; d++) {
        const re = new Array(n).fill(0), im = new Array(n).fill(0);
        for (let kk = 1; kk <= n / 2; kk++) { const f = kk * fs / n, a = Math.sqrt(n * fs * (2 * s2w / fs) * Math.pow(fk / f, al) / 4); const g1 = rng.normal(), g2 = kk === n / 2 ? 0 : rng.normal(); re[kk] = a * g1; im[kk] = a * g2; if (kk < n / 2) { re[n - kk] = a * g1; im[n - kk] = -a * g2; } }
        M.fft(re, im, true); const xs = re.map((v) => v + 0.004 * rng.normal());
        for (let s = 0; s + L <= n; s += D) { let mu = 0; for (let i = 0; i < L; i++) mu += xs[s + i]; mu /= L; const sr = w.map((wi, i) => wi * (xs[s + i] - mu)), si = new Array(L).fill(0); M.fft(sr, si); for (let q = 0; q <= L / 2; q++) Pm[q] += sr[q] ** 2 + si[q] ** 2; K++; }
      }
      const f = Pm.map((_, q) => q * fs / L), P = Pm.map((v, q) => (q === 0 || q === L / 2 ? 1 : 2) * v / (K * L * U * fs)), truth = f.map((ff) => (2 * s2w / fs) * (1 + Math.pow(fk / Math.max(ff, 1e-9), al)));
      const nu = 2 * 0.95 * K, lo = nu / (nu * Math.pow(1 - 2 / (9 * nu) + 1.96 * Math.sqrt(2 / (9 * nu)), 3)), hi = nu / (nu * Math.pow(1 - 2 / (9 * nu) - 1.96 * Math.sqrt(2 / (9 * nu)), 3));
      Bc.begin({ xlim: [0.3, 50], ylim: [1e-7, 1e-3], xlog: true, ylog: true, xlabel: "frequency along track [Hz]", ylabel: "PSD of (data − model)" });
      Bc.fill(f.slice(1), P.slice(1).map((v) => v * lo), P.slice(1).map((v) => v * hi), { c: "blue", a: 0.2 }); Bc.line(f.slice(1), P.slice(1), { c: "blue", w: 1.8 }); Bc.line(f.slice(1), truth.slice(1), { c: "ink", w: 2, dash: [5, 4] }); Bc.vline(fk, { c: "red", label: "f_k" });
      let kn = NaN; for (let q = f.length - 2; q >= 1; q--) if (P[q] > 2 * (2 * s2w / fs) * 1.0) { kn = f[q]; break; }
      k.set("kn", isFinite(kn) ? kn.toFixed(2) + " Hz (P = 2σ²)" : "below the first bin");
    }
    root._redraw = () => { draw1(); draw2(); }; draw1(); draw2();
    UI.legend(root, [["ink", "truth"], ["orange", "first difference (scene gradients leak in)"], ["green", "second difference"], ["blue", "Welch of 24 detectors, 95% band"]]);
    UI.note(root, "Top: σ of every band estimated from neighbouring-pixel differences, robust to edges (median absolute deviation). Bottom: a push-broom records one row per time step, so each detector column is a time series; with a slow drift the residual spectrum has a knee. Above the knee averaging rows still helps; below it, it does not.");
  };

  /* 7. which bands carry the information? */
  W.fisherbands = (root) => {
    UI.head(root, "WHICH BANDS CONSTRAIN THE ABUNDANCES? · POSTERIOR STD (AᵀC⁻¹A)^−½");
    const k = UI.kpis(root, [["n", "bands used"], ["w", "σ(water)"], ["v", "σ(vegetation)"]]);
    const P = UI.canvas(root, 280), S = UI.canvas(root, 200); const c = UI.ctrls(root); let sub = "all";
    UI.select(c, { label: "band subset", options: [["all", "all 120 bands"], ["every4", "every 4th band (30)"], ["vnir", "visible + NIR only (400–1000 nm)"], ["swir", "SWIR only (1000–2500 nm)"], ["nowater", "drop the water-vapour bands"], ["rgbn", "4 bands: 470, 550, 650, 850 nm"]], on: (v) => { sub = v; draw(); } });
    const wl = M.linspace(400, 2500, 120), E = endmembers(wl), sigB = noiseSigma(wl), names = ["vegetation", "soil", "water", "roof"], cols = ["green", "orange", "blue", "slate"];
    const inv4 = (m) => { const n = 4, a = m.map((r, i) => r.concat([0, 1, 2, 3].map((j) => (i === j ? 1 : 0)))); for (let i = 0; i < n; i++) { let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(a[r][i]) > Math.abs(a[p][i])) p = r; [a[i], a[p]] = [a[p], a[i]]; const d = a[i][i]; for (let j = 0; j < 2 * n; j++) a[i][j] /= d; for (let r = 0; r < n; r++) if (r !== i) { const f = a[r][i]; for (let j = 0; j < 2 * n; j++) a[r][j] -= f * a[i][j]; } } return a.map((r) => r.slice(n)); };
    const stdFor = (use) => { const F = [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => M.sum(wl.map((_, b) => (use[b] ? E[i][b] * E[j][b] / sigB[b] ** 2 : 0))))); const Si = inv4(F); return [0, 1, 2, 3].map((i) => Math.sqrt(Math.max(0, Si[i][i]))); };
    const pick = (s) => wl.map((w, b) => s === "all" ? true : s === "every4" ? b % 4 === 0 : s === "vnir" ? w <= 1000 : s === "swir" ? w > 1000 : s === "nowater" ? !(Math.abs(w - 1400) < 120 || Math.abs(w - 1900) < 140) : [470, 550, 650, 850].some((t) => Math.abs(w - t) < 9));
    const base = stdFor(pick("all"));
    function draw() {
      const use = pick(sub), sd = stdFor(use);
      P.begin({ xlim: [400, 2500], ylim: [0, 0.55], xlabel: "wavelength [nm]", ylabel: "reflectance" });
      wl.forEach((w, b) => { if (use[b]) P.rect(w - 8.8, 0, w + 8.8, 0.55, { c: "yellow", a: 0.45 }); });
      E.forEach((e, m) => P.line(wl, e, { c: cols[m], w: 2 }));
      S.begin({ xlim: [-0.5, 3.5], ylim: [1e-4, 10], ylog: true, xlabel: "material", ylabel: "posterior std of θ" });
      sd.forEach((v, i) => { S.rect(i - 0.35, 1e-4, i, Math.max(1e-4, v), { c: cols[i], a: 0.8 }); S.rect(i, 1e-4, i + 0.35, base[i], { c: "faint", a: 0.5 }); S.text(i, 4, names[i], { align: "center", size: 10.5 }); });
      k.set("n", use.filter(Boolean).length); k.set("w", sd[2].toFixed(4) + "  (all: " + base[2].toFixed(4) + ")"); k.set("v", sd[0].toFixed(4) + "  (all: " + base[0].toFixed(4) + ")");
    }
    root._redraw = draw; draw();
    UI.note(root, "Highlighted: the bands used. Bottom: posterior std of each abundance with this subset (colour) and with all bands (grey), for one pixel with the noise of the synthetic cube — no data needed, this is a Fisher forecast for a sensor design. Water is the hardest material: dark and featureless, it is only told apart in the visible.");
  };

  /* 6. Bayes classifier + abundance posterior (moved from Lesson 12) */
  W.hsi = (root) => {
    UI.head(root, "CLICK A PIXEL · CLASS POSTERIOR AND ABUNDANCE POSTERIOR");
    const k = UI.kpis(root, [["px", "pixel"], ["map", "MAP class"], ["conf", "P(MAP class)"], ["truth", "true abundances"]]);
    const box = UI.el("div", "split"); root.append(box);
    const imgW = UI.el("div", "canvas-wrap"); const img = document.createElement("canvas"); imgW.append(img); box.append(imgW);
    const right = UI.el("div"); box.append(right);
    const S = UI.canvas(right, 200), Bc = UI.canvas(right, 150, { pad: [10, 12, 30, 92] });
    const c = UI.ctrls(root);
    const H = 48, Wd = 48, B = 60, rng = M.rng(42), wl = M.linspace(400, 2500, B);
    const gs = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2), sig = (x, x0, kk) => 1 / (1 + Math.exp(-(x - x0) / kk));
    const wd = wl.map((x) => 1 - 0.55 * gs(x, 1450, 45) - 0.75 * gs(x, 1940, 60));
    const E = [
      wl.map((x, i) => (0.04 + 0.06 * gs(x, 550, 35) + 0.46 * sig(x, 715, 12) - 0.2 * sig(x, 1300, 80) - 0.08 * sig(x, 2000, 90)) * wd[i]),
      wl.map((x) => (0.08 + 0.3 * sig(x, 900, 260) - 0.05 * gs(x, 2200, 40)) * (1 - 0.25 * gs(x, 1450, 45) - 0.35 * gs(x, 1940, 60))),
      wl.map((x) => 0.07 * Math.exp(-(x - 400) / 180) + 0.005),
      wl.map((x) => (0.28 + 0.04 * (x - 400) / 2100 - 0.04 * gs(x, 1730, 30)) * (1 - 0.15 * gs(x, 1450, 45) - 0.2 * gs(x, 1940, 60)))];
    const names = ["vegetation", "soil", "water", "roof"], ccol = ["green", "orange", "blue", "slate"];
    const sigB = wl.map((x) => 0.004 + 0.03 * gs(x, 1400, 40) + 0.04 * gs(x, 1900, 50) + 0.01 * (x > 2350));
    // abundance fields: random blobs + river + building, softmax
    const blobs = [0, 1, 2, 3].map(() => Array.from({ length: 7 }, () => [rng() * Wd, rng() * H, 5 + 8 * rng(), rng.normal()]));
    const ab = [], lab = [], cube = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < Wd; x++) {
      const z = blobs.map((bl) => bl.reduce((s, [bx, by, r, a]) => s + a * Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * r * r)), 0));
      z[0] += 0.4; z[2] += 4 * gs(x - 0.25 * y, 28, 1.6) - 1.3; z[3] += (Math.abs(x - 10) < 5 && Math.abs(y - 11) < 5 ? 4 : 0) + (Math.abs(x - 38) < 4 && Math.abs(y - 38) < 6 ? 3.5 : 0) - 1;
      const e = z.map((v) => Math.exp(2.4 * v)), se = M.sum(e), a = e.map((v) => v / se);
      ab.push(a); lab.push(a.indexOf(Math.max(...a)));
      cube.push(wl.map((_, b) => a[0] * E[0][b] + a[1] * E[1][b] + a[2] * E[2][b] + a[3] * E[3][b] + sigB[b] * rng.normal()));
    }
    // linear Gaussian posterior (flat prior): Sigma = (A^T C^-1 A)^-1
    const AtCA = [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => M.sum(wl.map((_, b) => E[i][b] * E[j][b] / sigB[b] ** 2))));
    const inv4 = (m) => { const n = 4, a = m.map((r, i) => r.concat([0, 1, 2, 3].map((j) => (i === j ? 1 : 0)))); for (let i = 0; i < n; i++) { let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(a[r][i]) > Math.abs(a[p][i])) p = r; [a[i], a[p]] = [a[p], a[i]]; const d = a[i][i]; for (let j = 0; j < 2 * n; j++) a[i][j] /= d; for (let r = 0; r < n; r++) if (r !== i) { const f = a[r][i]; for (let j = 0; j < 2 * n; j++) a[r][j] -= f * a[i][j]; } } return a.map((r) => r.slice(n)); };
    const Sig = inv4(AtCA);
    let prior = "flat", sel = 14 * Wd + 27, comp = "true";
    const freq = [0, 1, 2, 3].map((c2) => lab.filter((l) => l === c2).length / lab.length);
    UI.select(c, { label: "image composite", options: [["true", "true colour (650 / 550 / 470 nm)"], ["cir", "false colour (NIR 830 / 650 / 550 nm)"]], on: (v) => { comp = v; draw(); } });
    UI.select(c, { label: "class prior", options: [["flat", "flat (¼ each)"], ["freq", "class frequency in the scene"]], on: (v) => { prior = v; draw(); } });
    const tau = 0.03; // within-class variability added to the noise
    function draw() {
      // image
      const wpx = imgW.clientWidth, cs = Math.floor(Math.min(wpx, 420) / Wd), dpr = window.devicePixelRatio || 1;
      img.width = Wd * cs * dpr; img.height = H * cs * dpr; img.style.width = Wd * cs + "px"; img.style.height = H * cs + "px";
      const g = img.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
      const [bR, bG, bB, gain] = comp === "true" ? [7, 4, 2, 1100] : [12, 7, 4, 520]; // band indices (35.6 nm apart from 400 nm)
      for (let i = 0; i < H * Wd; i++) { const s = cube[i]; g.fillStyle = `rgb(${Math.max(0, Math.min(255, s[bR] * gain))},${Math.max(0, Math.min(255, s[bG] * gain))},${Math.max(0, Math.min(255, s[bB] * gain))})`; g.fillRect((i % Wd) * cs, Math.floor(i / Wd) * cs, cs, cs); }
      g.strokeStyle = "#ff3b30"; g.lineWidth = 2.5; g.strokeRect((sel % Wd) * cs - 2, Math.floor(sel / Wd) * cs - 2, cs + 4, cs + 4);
      // pixel analysis
      const y = cube[sel];
      const ll = [0, 1, 2, 3].map((c2) => M.sum(wl.map((_, b) => { const v = sigB[b] ** 2 + tau * tau; return -0.5 * (y[b] - E[c2][b]) ** 2 / v - 0.5 * Math.log(v); })));
      const lpost = ll.map((v, c2) => v + Math.log(prior === "flat" ? 0.25 : Math.max(freq[c2], 1e-4))), mx = Math.max(...lpost);
      const pe = lpost.map((v) => Math.exp(v - mx)), Zc = M.sum(pe), pc = pe.map((v) => v / Zc);
      const Aty = [0, 1, 2, 3].map((i) => M.sum(wl.map((_, b) => E[i][b] * y[b] / sigB[b] ** 2)));
      const th = Sig.map((r) => M.sum(r.map((v, j) => v * Aty[j]))), er = [0, 1, 2, 3].map((i) => Math.sqrt(Sig[i][i]));
      S.begin({ xlim: [400, 2500], ylim: [-0.02, 0.55], xlabel: "wavelength [nm]", ylabel: "reflectance", title: "spectrum (dots) and posterior-mean mixture (line)" });
      S.pts(wl, y, { c: "ink", r: 2.2 });
      S.line(wl, wl.map((_, b) => M.sum([0, 1, 2, 3].map((i) => th[i] * E[i][b]))), { c: "red", w: 2 });
      Bc.begin({ xlim: [-0.15, 1.15], ylim: [-0.5, 3.5], axes: true, xlabel: "probability / abundance", yticks: 1 });
      names.forEach((nm, i) => {
        Bc.rect(0, 3 - i - 0.42, pc[i], 3 - i - 0.05, { c: ccol[i], a: 0.7 });
        Bc.rect(Math.max(-0.15, th[i] - er[i]), 3 - i + 0.08, Math.min(1.15, th[i] + er[i]), 3 - i + 0.38, { c: "purple", a: 0.35 });
        Bc.pts([th[i]], [3 - i + 0.23], { c: "purple", r: 3.5 }); Bc.pts([ab[sel][i]], [3 - i + 0.23], { c: "red", r: 3.5 });
        Bc.text(Bc.x0 - 6, Bc.py(3 - i), nm, { px: true, align: "right", base: "middle", size: 11 });
      });
      k.set("px", `(${sel % Wd}, ${Math.floor(sel / Wd)})`); const im = pc.indexOf(Math.max(...pc)); k.set("map", names[im]); k.set("conf", (100 * pc[im]).toFixed(1) + "%");
      k.set("truth", ab[sel].map((v) => v.toFixed(2)).join(" / "));
    }
    img.addEventListener("click", (e) => { const r = img.getBoundingClientRect(), cs = r.width / Wd; sel = Math.floor((e.clientY - r.top) / cs) * Wd + Math.floor((e.clientX - r.left) / cs); draw(); });
    img.style.cursor = "crosshair";
    root._redraw = draw; draw();
    UI.note(root, "Bars: coloured = posterior class probability; purple = abundance posterior mean ± 1σ (flat prior, exact Gaussian); red dot = true abundance. Click the river edges and the field borders: the class probability splits, the abundances tell you why.");
  };

  /* 8. Monte Carlo on the scene (moved from Lesson 13) */
  W.hsimc = (root) => {
    UI.head(root, "RANDOM PIXELS & NOISY BANDS · MONTE CARLO ON A SYNTHETIC SCENE");
    const k = UI.kpis(root, [["rms", "rms error of the mean spectrum"], ["cov", "bands within ±1σ"], ["nd", "NDVI: MC mean ± std"], ["lin", "linear propagation ±"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 280), B = UI.canvas(box, 280);
    const c = UI.ctrls(root);
    const nb = 80, wl = M.linspace(400, 2500, nb), rng = M.rng(17);
    const gs = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2), sig = (x, x0, kk) => 1 / (1 + Math.exp(-(x - x0) / kk));
    const soil = wl.map((x) => (0.08 + 0.3 * sig(x, 900, 260) - 0.05 * gs(x, 2200, 40)) * (1 - 0.25 * gs(x, 1450, 45) - 0.35 * gs(x, 1940, 60)));
    const sigB = wl.map((x) => 0.004 + 0.03 * gs(x, 1400, 40) + 0.04 * gs(x, 1900, 50));
    const pop = []; for (let i = 0; i < 5000; i++) { const br = 1 + 0.12 * rng.normal(); pop.push(soil.map((v, j) => v * br + sigB[j] * rng.normal())); }
    const truth = wl.map((_, j) => M.mean(pop.map((p) => p[j])));
    let n = 25, noise = 1, target = "veg";
    UI.slider(c, { label: "random pixels N", min: 0.5, max: 3.3, step: 0.05, value: Math.log10(25), fmt: (v) => Math.round(Math.pow(10, v)).toLocaleString(), on: (v) => { n = Math.round(Math.pow(10, v)); draw(); } });
    UI.slider(c, { label: "band noise × (NDVI panel)", min: 1, max: 30, step: 0.5, value: 1, fmt: (v) => v.toFixed(1) + "×", on: (v) => { noise = v; draw(); } });
    UI.select(c, { label: "NDVI pixel", options: [["veg", "vegetation (R 0.05, NIR 0.48)"], ["water", "water (R 0.012, NIR 0.006)"]], on: (v) => { target = v; draw(); } });
    const b = UI.btns(root); UI.button(b, "new random sample", () => draw(), "go");
    function draw() {
      const smp = []; for (let i = 0; i < n; i++) smp.push(pop[Math.floor(rng() * pop.length)]);
      const mean = wl.map((_, j) => M.mean(smp.map((p) => p[j]))), err = wl.map((_, j) => (n > 1 ? M.std(smp.map((p) => p[j])) / Math.sqrt(n) : 0));
      A.begin({ xlim: [400, 2500], ylim: [0, 0.5], xlabel: "wavelength [nm]", ylabel: "reflectance", title: "class mean: all pixels vs N random" });
      A.fill(wl, mean.map((m, j) => m - err[j]), mean.map((m, j) => m + err[j]), { c: "orange", a: 0.35 }); A.line(wl, truth, { c: "ink", w: 2 }); A.line(wl, mean, { c: "orange", w: 1.5 });
      const rmsE = Math.sqrt(M.mean(mean.map((m, j) => (m - truth[j]) ** 2))), cov = mean.filter((m, j) => Math.abs(m - truth[j]) < err[j]).length / nb;
      k.set("rms", rmsE.toExponential(2)); k.set("cov", (100 * cov).toFixed(0) + "% (correlated bands!)");
      const [R0, N0] = target === "veg" ? [0.05, 0.48] : [0.012, 0.006], sR = 0.004 * noise, sN = 0.005 * noise, nds = [];
      for (let i = 0; i < 20000; i++) { const r = R0 + sR * rng.normal(), q = N0 + sN * rng.normal(); const v = (q - r) / (q + r); if (isFinite(v)) nds.push(v); }
      const nd0 = (N0 - R0) / (N0 + R0), lin = Math.hypot(2 * N0 / (N0 + R0) ** 2 * sR, 2 * R0 / (N0 + R0) ** 2 * sN);
      const srt = nds.slice().sort((a, b2) => a - b2), lo = Math.max(-1.5, M.quantile(srt, 0.005)), hi = Math.min(1.5, M.quantile(srt, 0.995)), h = M.hist(nds, lo, hi, 60), t = M.linspace(lo, hi, 200);
      B.begin({ xlim: [lo, hi], ylim: [0, Math.max(...h.dens, M.npdf(0, 0, lin)) * 1.15], xlabel: "NDVI", ylabel: "density", title: "noise → NDVI: simulation vs linear" });
      B.bars(h.edges, h.dens, { c: "green" }); B.line(t, t.map((x) => M.npdf(x, nd0, lin)), { c: "ink", w: 2 });
      k.set("nd", `${M.mean(nds).toFixed(3)} ± ${M.std(nds).toFixed(3)}`); k.set("lin", lin.toFixed(3));
    }
    root._redraw = draw; draw();
    UI.note(root, "Water pixel with noise ×5: the denominator NIR + R approaches zero, the simulated NDVI distribution becomes skewed and heavy-tailed, and the linear formula is wrong. Simulation needs no derivatives and no Gaussian assumption.");
  };
})();
