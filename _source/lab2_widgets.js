/* Lab 2 — Planck power spectrum: symbols and widgets (data: window.LAB2DATA, binned from the real maps) */
window.SYMBOLS = {
  Bl: { n: "B<sub>ℓ</sub> — beam transfer function", t: "0 … 1", s: "Gaussian beam: exp(−ℓ(ℓ+1)σ²/2), σ = FWHM/2.355. 70 GHz: 13.08′; 100 GHz: 9.66′.", py: "B = hp.gauss_beam(np.radians(FWHM_ARCMIN[FREQ]/60), lmax=lmax)", f: "bl = exp(-0.5_dp*l*(l+1)*sig**2)" },
  Wl: { n: "W<sub>ℓ</sub> — pixel window", t: "0 … 1", s: "Averaging over the HEALPix pixel. Read from pixel_window_n1024.fits.", py: "W = np.atleast_2d(hp.read_cl(path('pixel_window_n1024.fits')))[0][:lmax+1]", f: "! read with the HEALPix F90 fits routines" },
  Nn: { n: "N — white noise level", t: "µK² sr", s: "Mean of Ĉ<sub>ℓ</sub> where the beam has erased the sky: 3.27×10⁻³ (70 GHz), 5.14×10⁻⁴ (100 GHz).", py: "N = cl_mask[(ell >= 2000) & (ell <= 2500)].mean()", f: "noise = sum(cl(2000:2500))/501" },
  fsky: { n: "f<sub>sky</sub> — sky fraction", t: "0 … 1", s: "Mean of the UT78 mask: 0.778.", py: "fsky = mask.mean()", f: "fsky = sum(mask)/npix" }
};

(function () {
  const W = (window.WIDGETS = {});
  const D = window.LAB2DATA, el = UI.el, ARCMIN = Math.PI / 180 / 60;
  const FW = { 70: 13.08, 100: 9.66 }, NB = D.nb, LC = Array.from({ length: NB }, (_, i) => D.l0 + i * D.dl + (D.dl - 1) / 2);
  const toD = (c, l) => l * (l + 1) * c / (2 * Math.PI);
  const B2bin = (fw) => { const s = fw * ARCMIN / Math.sqrt(8 * Math.log(2)); return LC.map((_, i) => { let a = 0; for (let l = D.l0 + i * D.dl; l < D.l0 + (i + 1) * D.dl; l++) a += Math.exp(-l * (l + 1) * s * s); return a / D.dl; }); };
  const Nof = (f, a = 2000, b = 2500) => { let s = 0, n = 0; LC.forEach((l, i) => { if (l >= a && l <= b) { s += D[f].mask[i]; n++; } }); return s / n; };
  const rebin = (arr, k) => { const o = []; for (let i = 0; i + k <= arr.length; i += k) { let s = 0; for (let j = 0; j < k; j++) s += arr[i + j]; o.push(s / k); } return o; };
  const DTH = LC.map((l, i) => toD(D.Cth[i], l));
  const chanSel = (c, on, v = "70") => UI.select(c, { label: "channel", value: v, options: [["70", "70 GHz (core)"], ["100", "100 GHz (ext. A)"]], on });

  /* ------------------------------------------------------------------ 1. overview: the measurement chain */
  W.overview = (root) => {
    UI.head(root, "THE PIPELINE · CLICK A STEP");
    const steps = [
      ["0", "Read & look", "Map, mask, f_sky = 0.778", "s2"], ["1", "Full-sky Ĉ_ℓ", "Galaxy + noise: 10× the model", "s3"],
      ["2", "Mask the Galaxy", "remove ℓ = 0, 1; ÷ f_sky", "s4"], ["3", "Noise bias N", "3.27×10⁻³ µK² sr", "s5"],
      ["4", "Beam & pixels", "(Ĉ − N)/B²W², bins of 30", "s6"], ["A–C", "Extensions", "100 GHz · excess A · cross", "s7"], ["★", "Results", "3 peaks, reach ℓ ≈ 860", "s8"]];
    const row = el("div"); row.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:8px;margin:6px 0 4px";
    steps.forEach(([n, t, d, s], i) => { const a = el("a", "unit", `<h3 style="font-size:13.5px">${n} · ${t}</h3><p>${d}</p>`); a.href = "#" + s; a.style.cssText = "padding:10px;box-shadow:none;border-width:1.5px;background:" + (i % 2 ? "var(--paper)" : "var(--panel)"); row.append(a); });
    root.append(row);
    UI.note(root, "Each step removes one term of the measurement equation ⟨Ĉ_ℓ⟩ = B_ℓ²W_ℓ² C_ℓ^sky + N, from the outside in. The notebook <a href=\"Lab2_Planck_power_spectrum.ipynb\" download><b>Lab2_Planck_power_spectrum.ipynb</b></a> follows the professor's code block by block.");
    root._redraw = () => {};
  };

  /* ------------------------------------------------------------------ 2. the sky (Mollweide from the real map) */
  W.sky = (root) => {
    UI.head(root, "THE 70 GHz SKY · REAL PLANCK MAP (DEGRADED FOR THE WEB) · MOVE THE COLOUR RANGE");
    const k = UI.kpis(root, [["v", "value under the pointer"], ["r", "colour range"], ["sat", "pixels saturated"], ["fs", "kept sky"]]);
    const P = UI.canvas(root, (w) => Math.round(w * 0.5), { pad: [4, 4, 4, 4] });
    const c = UI.ctrls(root); let R = 300, showMask = false;
    UI.slider(c, { label: "colour range ±", min: Math.log10(30), max: Math.log10(150000), step: 0.01, value: Math.log10(300), fmt: (v) => Math.round(10 ** v).toLocaleString("en") + " µK", on: (v) => { R = 10 ** v; draw(); } });
    UI.select(c, { label: "mask UT78", options: [["0", "not applied"], ["1", "applied (grey)"]], on: (v) => { showMask = v === "1"; draw(); } });
    const b = UI.btns(root); [["±300 µK (CMB)", 300], ["±3 000 µK", 3000], ["±150 000 µK (full range)", 150000]].forEach(([t, v]) => UI.button(b, t, () => { R = v; c.querySelector("input").set(Math.log10(v)); draw(); }));
    const I = D.img, nx = I.nx, ny = I.ny, raw = atob(I.map), mk = atob(I.mask), val = new Float32Array(nx * ny), ms = new Uint8Array(nx * ny);
    for (let i = 0; i < nx * ny; i++) { let v = raw.charCodeAt(2 * i) | (raw.charCodeAt(2 * i + 1) << 8); if (v >= 32768) v -= 65536; val[i] = v === -32768 ? NaN : v * I.scale; ms[i] = mk.charCodeAt(i); }
    const cm = (t) => { const s = [[33, 102, 172], [103, 169, 207], [247, 247, 247], [239, 138, 98], [178, 24, 43]], x = t * 4, j = Math.min(3, Math.floor(x)), f = x - j; return s[j].map((a, q) => a + (s[j + 1][q] - a) * f); };
    const oc = document.createElement("canvas"); oc.width = nx; oc.height = ny;
    function draw() {
      const g2 = oc.getContext("2d"), im = g2.createImageData(nx, ny); let sat = 0, tot = 0;
      for (let i = 0; i < nx * ny; i++) { const v = val[i]; if (!isFinite(v)) { im.data[4 * i + 3] = 0; continue; } tot++;
        let rgb; if (showMask && ms[i] === 1) rgb = [190, 190, 190]; else { if (Math.abs(v) > R) sat++; rgb = cm(Math.max(0, Math.min(1, (v / R + 1) / 2))); }
        im.data[4 * i] = rgb[0]; im.data[4 * i + 1] = rgb[1]; im.data[4 * i + 2] = rgb[2]; im.data[4 * i + 3] = 255; }
      g2.putImageData(im, 0, 0);
      P.begin({ xlim: [0, 1], ylim: [0, 1], axes: false }); const g = P.ctx; g.imageSmoothingEnabled = true; g.drawImage(oc, 4, 4, P.W - 8, P.H - 8);
      k.set("r", "±" + Math.round(R).toLocaleString("en") + " µK"); k.set("sat", (100 * sat / tot).toFixed(1) + "%"); k.set("fs", (100 * D["70"].fsky).toFixed(1) + "%");
    }
    P.c.addEventListener("mousemove", (e) => { const r = P.c.getBoundingClientRect(), i = Math.floor((e.clientX - r.left - 4) / (r.width - 8) * nx), j = Math.floor((e.clientY - r.top - 4) / (r.height - 8) * ny);
      if (i < 0 || j < 0 || i >= nx || j >= ny) return; const v = val[j * nx + i]; k.set("v", isFinite(v) ? Math.round(v).toLocaleString("en") + " µK" + (ms[j * nx + i] === 1 ? " (cut)" : "") : "—"); });
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 3. full-sky spectrum */
  W.fullsky = (root) => {
    UI.head(root, "D_ℓ OF THE WHOLE SKY VS THE MODEL · BINS OF 10");
    const k = UI.kpis(root, [["r220", "full / model at ℓ ≈ 220"], ["r800", "at ℓ ≈ 800"], ["r2000", "at ℓ ≈ 2000"], ["n", "noise D_ℓ at ℓ = 2000"]]);
    const P = UI.canvas(root, 320);
    const c = UI.ctrls(root); let f = "70", showN = true, showM = false;
    chanSel(c, (v) => { f = v; draw(); });
    UI.select(c, { label: "overlay", options: [["n", "white noise ℓ(ℓ+1)N/2π"], ["m", "noise + masked spectrum (Step 2)"], ["0", "nothing"]], on: (v) => { showN = v !== "0"; showM = v === "m"; draw(); } });
    function draw() {
      const full = LC.map((l, i) => toD(D[f].full[i], l)), N = Nof(f, f === "70" ? 2000 : 2600, f === "70" ? 2500 : 3000);
      P.begin({ xlim: [2, 3071], ylim: [10, 2e5], ylog: true, xlabel: "multipole ℓ", ylabel: "D_ℓ [µK²]" });
      P.line(LC, full, { c: "muted", w: 1.6 });
      if (showM) P.line(LC, LC.map((l, i) => toD(D[f].mask[i], l)), { c: "blue", w: 1.6 });
      P.line(LC.filter((l) => l < 2508), DTH.filter((_, i) => LC[i] < 2508), { c: "ink", w: 2.4 });
      if (showN) P.line(LC, LC.map((l) => toD(N, l)), { c: "red", w: 2, dash: [6, 4] });
      P.legendBox([["muted", "full sky"], ...(showM ? [["blue", "masked / f_sky"]] : []), ["ink", "Planck best fit"], ...(showN ? [["red", "white noise N", [6, 4]]] : [])], { bottom: true });
      const at = (L) => { const i = Math.round((L - D.l0 - 4.5) / D.dl); return full[i] / DTH[i]; };
      k.set("r220", at(220).toFixed(1) + "×"); k.set("r800", at(800).toFixed(1) + "×"); k.set("r2000", at(2000).toFixed(1) + "×"); k.set("n", Math.round(toD(N, 2000)) + " µK²");
    }
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 4. masking: leakage and f_sky */
  W.masking = (root) => {
    UI.head(root, "WHAT THE MASK DOES · TOGGLE EACH CORRECTION");
    const k = UI.kpis(root, [["fs", "f_sky"], ["r220", "masked / model at ℓ ≈ 220"], ["leak", "raw / clean at ℓ = 10"], ["q", "D₂ masked vs model"]]);
    const sp = el("div", "split"); root.append(sp);
    const A = UI.canvas(sp, 300), B = UI.canvas(sp, 300);
    const c = UI.ctrls(root); let f = "70", div = true, dip = true;
    chanSel(c, (v) => { f = v; draw(); });
    UI.select(c, { label: "divide by f_sky", options: [["1", "yes"], ["0", "no (22% low)"]], on: (v) => { div = v === "1"; draw(); } });
    UI.select(c, { label: "monopole + dipole", options: [["1", "removed on the kept sky"], ["0", "not removed (raw m·mask)"]], on: (v) => { dip = v === "1"; draw(); } });
    function draw() {
      const fs = D[f].fsky, sc = div ? 1 : fs, arr = dip ? D[f].mask : D[f].raw;
      const Dm = LC.map((l, i) => toD(arr[i] * sc, l)), full = LC.map((l, i) => toD(D[f].full[i], l));
      A.begin({ xlim: [2, 1500], ylim: [0, 9000], xlabel: "multipole ℓ", ylabel: "D_ℓ [µK²]" });
      A.line(LC, DTH, { c: "ink", w: 2.2 }); A.line(LC, full, { c: "muted", w: 1.2 }); A.line(LC, Dm, { c: "blue", w: 1.8 });
      A.legendBox([["ink", "model"], ["muted", "full sky (off scale)"], ["blue", "masked" + (div ? " / f_sky" : "")]]);
      const ls = Array.from({ length: 29 }, (_, i) => i + 2), raw = D[f].lowraw, cln = D[f].lowmask;
      B.begin({ xlim: [1.5, 30.5], ylim: [0.8, 2.0], xlabel: "multipole ℓ", ylabel: "raw / cleaned Ĉ_ℓ" });
      B.hline(1, { c: "ink", dash: [] }); B.pts(ls, ls.map((_, i) => raw[i] / cln[i]), { c: "red", r: 4 });
      B.text(3, 1.95, "leakage of ℓ = 0, 1 through the mask", { c: "muted" });
      const i220 = Math.round((220 - 6.5) / 10);
      k.set("fs", fs.toFixed(3)); k.set("r220", (Dm[i220] / DTH[i220]).toFixed(2)); k.set("leak", (raw[8] / cln[8]).toFixed(2)); k.set("q", Math.round(toD(cln[0] * sc, 2)) + " vs 1017 µK²");
    }
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 5. noise: band and whiteness */
  W.noise = (root) => {
    UI.head(root, "Ĉ_ℓ / N · CHOOSE THE SKY-FREE BAND THAT DEFINES N");
    const k = UI.kpis(root, [["N", "N from the band"], ["w1", "Ĉ/N, ℓ 1500–2000"], ["w2", "Ĉ/N, ℓ 2500–3000"], ["d", "noise D_ℓ at ℓ = 220 / 1000"]]);
    const P = UI.canvas(root, 300);
    const c = UI.ctrls(root); let f = "70", a = 2000, b = 2500;
    chanSel(c, (v) => { f = v; if (v === "100") { a = 2600; b = 3000; sa.set(a); sb.set(b); } else { a = 2000; b = 2500; sa.set(a); sb.set(b); } draw(); });
    const sa = UI.slider(c, { label: "band start", min: 800, max: 2900, step: 10, value: a, on: (v) => { a = Math.min(v, b - 50); draw(); } });
    const sb = UI.slider(c, { label: "band end", min: 900, max: 3060, step: 10, value: b, on: (v) => { b = Math.max(v, a + 50); draw(); } });
    function draw() {
      const N = Nof(f, a, b), r = D[f].mask.map((v) => v / N), r30 = rebin(r, 3), l30 = rebin(LC, 3);
      P.begin({ xlim: [500, 3071], ylim: [0.6, 3], ylog: true, xlabel: "multipole ℓ", ylabel: "Ĉ_ℓ(masked) / N" });
      P.rect(a, 0.6, b, 3, { c: "green", a: 0.15 }); P.hline(1, { c: "ink", dash: [] });
      P.pts(l30, r30, { c: "blue", r: 2.5 });
      const mb = (x, y) => { let s = 0, n = 0; LC.forEach((l, i) => { if (l >= x && l < y) { s += r[i]; n++; } }); return s / n; };
      k.set("N", N.toExponential(3) + " µK² sr"); k.set("w1", mb(1500, 2000).toFixed(3)); k.set("w2", mb(2500, 3000).toFixed(3));
      k.set("d", Math.round(toD(N, 220)) + " / " + Math.round(toD(N, 1000)) + " µK²");
    }
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 6. recover: deconvolution of the real data */
  W.recover = (root) => {
    UI.head(root, "(Ĉ − N) / B²W² ON THE REAL DATA · CHANGE THE ASSUMPTIONS");
    const k = UI.kpis(root, [["r1", "recovered/model ℓ 30–300"], ["r2", "ℓ 300–600"], ["r3", "ℓ 600–800"], ["r4", "ℓ 800–1000"], ["reach", "noise/B²W² > CMB at"]]);
    const P = UI.canvas(root, 330);
    const c = UI.ctrls(root); let f = "70", fw = FW[70], dN = 0, bins = 3, fwd = false;
    chanSel(c, (v) => { f = v; fw = FW[v]; sf.set(fw); draw(); });
    const sf = UI.slider(c, { label: "assumed beam FWHM", min: 8, max: 15, step: 0.01, value: fw, fmt: (v) => v.toFixed(2) + "′", on: (v) => { fw = v; draw(); } });
    UI.slider(c, { label: "error on N", min: -5, max: 5, step: 0.1, value: 0, fmt: (v) => (v > 0 ? "+" : "") + v.toFixed(1) + "%", on: (v) => { dN = v / 100; draw(); } });
    UI.select(c, { label: "bin width", value: "3", options: [["1", "Δℓ = 10"], ["3", "Δℓ = 30 (lab)"], ["6", "Δℓ = 60"]], on: (v) => { bins = +v; draw(); } });
    UI.select(c, { label: "view", options: [["0", "corrected spectrum"], ["1", "forward model vs raw data"]], on: (v) => { fwd = v === "1"; draw(); } });
    function draw() {
      const fs = D[f].fsky, N0 = Nof(f, f === "70" ? 2000 : 2600, f === "70" ? 2500 : 3000), N = N0 * (1 + dN);
      const bw = B2bin(fw).map((b, i) => b * D.W2[i]), cm = D[f].mask;
      const l = rebin(LC, bins), th = rebin(DTH, bins), dl = D.dl * bins;
      if (fwd) {
        const raw = rebin(LC.map((L, i) => toD(cm[i], L)), bins), seen = rebin(LC.map((L, i) => toD(D.Cth[i] * bw[i] + N0, L)), bins);
        P.begin({ xlim: [2, 3071], ylim: [10, 2e4], ylog: true, xlabel: "multipole ℓ", ylabel: "D_ℓ [µK²]" });
        P.line(l, th, { c: "ink", w: 2 }); P.pts(l, raw, { c: "blue", r: 2.2 }); P.line(l, seen, { c: "red", w: 2, dash: [6, 4] });
        P.legendBox([["ink", "model"], ["blue", "masked data / f_sky"], ["red", "model × B²W² + noise", [6, 4]]], { bottom: true });
      } else {
        const rec = rebin(LC.map((L, i) => toD((cm[i] - N) / bw[i], L)), bins), dn = rebin(LC.map((L, i) => toD(N0 / bw[i], L)), bins);
        const er = l.map((L, i) => Math.sqrt(2 / ((2 * L + 1) * fs * dl)) * (th[i] + dn[i]));
        P.begin({ xlim: [0, 2000], ylim: [-1000, 8000], xlabel: "multipole ℓ", ylabel: "D_ℓ [µK²]" });
        P.fill(l, l.map((_, i) => th[i] - er[i]), l.map((_, i) => th[i] + er[i]), { c: "green", a: 0.22 });
        P.line(l, th, { c: "ink", w: 2.2 }); P.pts(l, rec, { c: "blue", r: 2.6 });
        P.legendBox([["ink", "Planck best fit"], ["green", "± Knox + noise error"], ["blue", "recovered " + f + " GHz"]]);
        const band = (x, y) => { let s = 0, n = 0; l.forEach((L, i) => { if (L >= x && L < y) { s += rec[i] / th[i]; n++; } }); return (s / n).toFixed(2); };
        k.set("r1", band(30, 300)); k.set("r2", band(300, 600)); k.set("r3", band(600, 800)); k.set("r4", band(800, 1000));
        let reach = null; l.forEach((L, i) => { if (reach === null && L > 200 && dn[i] > th[i]) reach = L; }); k.set("reach", reach ? "ℓ ≈ " + Math.round(reach) : "—");
        if (reach) P.vline(reach, { c: "red", label: "reach" });
      }
    }
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 7. extension B: the leftover excess */
  W.excess = (root) => {
    UI.head(root, "EXTENSION B · FIT A WHITE SKY COMPONENT A BY HAND (ℓ = 30–1200)");
    const k = UI.kpis(root, [["A", "your A"], ["best", "best A (bins of 10)"], ["chi", "χ² your A / no A"], ["fw", "best FWHM without A"]]);
    const sp = el("div", "split"); root.append(sp);
    const P = UI.canvas(sp, 300), Q = UI.canvas(sp, 300);
    const c = UI.ctrls(root); let f = "70", A = 0;
    chanSel(c, (v) => { f = v; draw(); });
    const sa = UI.slider(c, { label: "A [µK² sr]", min: 0, max: 0.01, step: 0.0001, value: 0, fmt: (v) => v.toExponential(2), on: (v) => { A = v; draw(); } });
    UI.button(UI.btns(root), "set the best A", () => { A = fit(f).A; sa.set(A); draw(); }, "primary");
    function chi(f, fw, A) { const fs = D[f].fsky, N = Nof(f, f === "70" ? 2000 : 2600, f === "70" ? 2500 : 3000), bw = B2bin(fw).map((b, i) => b * D.W2[i]); let s = 0;
      LC.forEach((l, i) => { if (l < 30 || l > 1200) return; const v = 2 * (D.Cth[i] * bw[i] + N) ** 2 / ((2 * l + 1) * fs * D.dl), y = D[f].mask[i] - N - (D.Cth[i] + A) * bw[i]; s += y * y / v; }); return s; }
    function fit(f) { const fs = D[f].fsky, N = Nof(f, f === "70" ? 2000 : 2600, f === "70" ? 2500 : 3000), bw = B2bin(FW[f]).map((b, i) => b * D.W2[i]); let sxy = 0, sxx = 0;
      LC.forEach((l, i) => { if (l < 30 || l > 1200) return; const v = 2 * (D.Cth[i] * bw[i] + N) ** 2 / ((2 * l + 1) * fs * D.dl), y = D[f].mask[i] - N - D.Cth[i] * bw[i], x = bw[i]; sxy += x * y / v; sxx += x * x / v; });
      return { A: sxy / sxx, N, bw }; }
    function draw() {
      const { A: Ab, N, bw } = fit(f), sel = LC.map((l) => l >= 30 && l <= 1200);
      const ex = LC.map((l, i) => ((D[f].mask[i] - N) / bw[i] - D.Cth[i]) * 1e3), l30 = rebin(LC, 3), e30 = rebin(ex, 3);
      P.begin({ xlim: [0, 1300], ylim: [-5, 15], xlabel: "multipole ℓ", ylabel: "Ĉ_rec − C_th [10⁻³ µK² sr]" });
      P.hline(0, { c: "ink", dash: [] }); P.pts(l30.filter((l) => l >= 30 && l <= 1250), e30.filter((_, i) => l30[i] >= 30 && l30[i] <= 1250), { c: f === "70" ? "blue" : "orange", r: 3 });
      P.hline(A * 1e3, { c: "red", dash: [], w: 2.5, label: "your A" }); P.hline(Ab * 1e3, { c: "green", w: 1.5, label: "best A" });
      const rec = rebin(LC.map((l, i) => toD((D[f].mask[i] - N) / bw[i], l)), 3), th = rebin(DTH, 3), thA = rebin(LC.map((l, i) => toD(D.Cth[i] + A, l)), 3);
      Q.begin({ xlim: [0, 1500], ylim: [-500, 8000], xlabel: "multipole ℓ", ylabel: "D_ℓ [µK²]" });
      Q.line(l30, th, { c: "ink", w: 2 }); Q.line(l30, thA, { c: "red", w: 2, dash: [6, 4] }); Q.pts(l30, rec, { c: f === "70" ? "blue" : "orange", r: 2.5 });
      Q.legendBox([["ink", "model"], ["red", "model + A", [6, 4]]], { bottom: true, left: true });
      let best = [Infinity, 0]; for (let fw = 0.85 * FW[f]; fw <= FW[f] * 1.0001; fw += 0.0025 * FW[f]) { const x2 = chi(f, fw, 0); if (x2 < best[0]) best = [x2, fw]; }
      k.set("A", A.toExponential(2)); k.set("best", Ab.toExponential(2)); k.set("chi", Math.round(chi(f, FW[f], A)) + " / " + Math.round(chi(f, FW[f], 0)));
      k.set("fw", best[1].toFixed(2) + "′ (χ² " + Math.round(best[0]) + ")");
      void sel;
    }
    root._redraw = draw; draw();
  };
})();
