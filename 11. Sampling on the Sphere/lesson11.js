/* Lesson 11 — sampling on the sphere: symbols and widgets */
window.SYMBOLS = {
  phi: { n: "φ — longitude (azimuth)", t: "rad, 0 … 2π", s: "Angle around the polar axis. Along a ring of constant θ a spherical harmonic is a pure wave e<sup>imφ</sup>.", py: "theta, phi = hp.pix2ang(nside, ipix)", f: "call pix2ang_ring(nside, ipix, theta, phi)" },
  dOm: { n: "dΩ — solid-angle element", t: "sr", s: "sin θ dθ dφ. The whole sphere is 4π sr; a latitude–longitude cell shrinks like sin θ near the poles.", py: "dOmega = np.sin(th) * dth * dphi", f: "domega = sin(th)*dth*dphi" },
  fk: { n: "f̂<sub>k</sub> — Fourier coefficient on the circle", t: "complex", s: "Weight of e<sup>ikφ</sup>. Exactly returned by the DFT of N ≥ 2B + 1 equispaced samples.", py: "fk = np.fft.fft(f) / N", f: "fk(k) = sum(f*exp(-(0,1)*k*phi))/N" },
  B: { n: "B — band limit", t: "integer", s: "Highest |k| present on the circle.", py: "B = 3", f: "integer :: B = 3" },
  Ns: { n: "N — samples on the circle", t: "integer", s: "Must be ≥ 2B + 1 for an exact DFT; otherwise k aliases to k ± N.", py: "N = 2*B + 1", f: "N = 2*B + 1" },
  Ylm: { n: "Y<sub>ℓm</sub>(θ, φ) — spherical harmonic", t: "complex function on the sphere", s: "Eigenfunction of the Laplacian on the sphere, eigenvalue −ℓ(ℓ+1). Orthonormal on the full sphere.", py: "scipy.special.sph_harm_y(l, m, theta, phi)", f: "! lambda_lm(theta) * exp((0,1)*m*phi)" },
  Plm: { n: "P<sub>ℓ</sub><sup>m</sup>(x) — associated Legendre function", t: "x = cos θ", s: "Polynomial in cos θ times sin<sup>m</sup>θ. Computed by a three-term recursion in ℓ, never from factorials.", py: "scipy.special.lpmv(m, l, x)", f: "! recursion in l at fixed m" },
  lam: { n: "λ<sub>ℓm</sub>(θ) — normalised Legendre function", t: "real", s: "The θ part of Y<sub>ℓm</sub>, normalisation included. Tabulated once per ring in the fast transform.", py: "lam = np.sqrt((2*l+1)/(4*np.pi)*factorial(l-m)/factorial(l+m)) * lpmv(m, l, np.cos(th))", f: "lam(l) = sqrt((4*l*l-1.0_dp)/(l*l-m*m))*(x*lam(l-1) - c*lam(l-2))" },
  ell: { n: "ℓ — multipole", t: "integer ≥ 0", s: "Sets the angular size of a pattern: ≈ 180°/ℓ. ℓ = 220 ↔ 0.8°, the first acoustic peak.", py: "ell = np.arange(lmax + 1)", f: "do l = 0, lmax" },
  m: { n: "m — azimuthal index", t: "−ℓ … ℓ", s: "Number of oscillations around a ring. 2ℓ + 1 values for each ℓ.", py: "hp.Alm.getidx(lmax, l, m)", f: "do m = 0, l" },
  alm: { n: "a<sub>ℓm</sub> — harmonic coefficients", t: "complex", s: "The map's content in harmonic space. Real map: a<sub>ℓ,−m</sub> = (−1)<sup>m</sup> a*<sub>ℓm</sub>; healpy stores only m ≥ 0.", py: "alm = hp.map2alm(m, lmax=lmax, iter=3)", f: "complex(dp) :: alm(1, 0:lmax, 0:lmax)" },
  Cl: { n: "C<sub>ℓ</sub> — angular power spectrum", t: "µK² sr", s: "Variance of a<sub>ℓm</sub>: all the information of an isotropic Gaussian sky.", py: "cl = hp.anafast(m, lmax=lmax)", f: "call alm2cl(lmax, lmax, alm, cl)" },
  Chat: { n: "Ĉ<sub>ℓ</sub> — estimate from one sky", t: "µK² sr", s: "Mean of |a<sub>ℓm</sub>|² over m. (2ℓ+1)Ĉ/C ~ χ²<sub>2ℓ+1</sub>: cosmic variance.", py: "chat = hp.alm2cl(alm)", f: "chat(l) = (abs(alm(l,0))**2 + 2*sum(abs(alm(l,1:l))**2))/(2*l+1)" },
  fsky: { n: "f<sub>sky</sub> — sky fraction", t: "0 … 1", s: "Mean of the 0/1 mask. A cut sky keeps on average f<sub>sky</sub> of the power and of the modes. Lab 2: 0.778.", py: "fsky = mask.mean()", f: "fsky = sum(mask)/npix" },
  Dell: { n: "Δℓ — bin width", t: "multipoles", s: "Averaging Δℓ multipoles divides the variance by Δℓ; needed anyway on a cut sky (resolution ~2π/Θ). Lab 2: 30.", py: "dl = 30", f: "integer, parameter :: dl = 30" },
  Dl: { n: "D<sub>ℓ</sub> — scaled spectrum", t: "µK²", s: "ℓ(ℓ+1)C<sub>ℓ</sub>/2π: flat for a scale-invariant sky; white noise becomes ∝ ℓ².", py: "Dl = ell*(ell+1)*cl/(2*np.pi)", f: "dl = l*(l+1)*cl/(2*pi)" },
  wp: { n: "w<sub>p</sub> — quadrature weight", t: "sr", s: "Weight of pixel p in the analysis sum. HEALPix: Ω<sub>pix</sub> (optionally ring weights).", py: "w = 4*np.pi/npix", f: "w = 4*pi/npix" },
  Npix: { n: "N<sub>pix</sub> — number of pixels", t: "12 N<sub>side</sub>²", s: "12 582 912 at N<sub>side</sub> = 1024.", py: "npix = hp.nside2npix(nside)", f: "npix = 12*nside**2" },
  Nside: { n: "N<sub>side</sub> — HEALPix resolution", t: "power of 2", s: "Each of the 12 base pixels is split into N<sub>side</sub> × N<sub>side</sub>. Choose it for 2–3 pixels per beam FWHM.", py: "nside = hp.npix2nside(m.size)", f: "nside = nint(sqrt(npix/12.0_dp))" },
  Opix: { n: "Ω<sub>pix</sub> — pixel solid angle", t: "sr", s: "4π/N<sub>pix</sub> = π/(3N<sub>side</sub>²): all pixels have the same area.", py: "hp.nside2pixarea(nside)", f: "opix = pi/(3.0_dp*nside**2)" },
  thpix: { n: "θ<sub>pix</sub> — pixel size", t: "≈ 58.6°/N<sub>side</sub>", s: "√Ω<sub>pix</sub>. 3.44′ at N<sub>side</sub> = 1024.", py: "hp.nside2resol(nside, arcmin=True)", f: "thpix = sqrt(opix)" },
  Fm: { n: "F<sub>m</sub>(r) — FFT of ring r", t: "complex", s: "Step 1 of the fast transform: one real FFT per ring, phase-shifted by the ring's first longitude.", py: "F = np.fft.rfft(ring) * np.exp(-1j*m*phi0)", f: "call fftw_execute_dft_r2c(plan, seg, F)" },
  nr: { n: "n<sub>r</sub> — pixels on ring r", t: "4i (caps) or 4N<sub>side</sub>", s: "Short polar rings alias high m: a 4-pixel ring cannot tell m from m ± 4.", py: "nr = 4*min(i, nside, 4*nside - i)", f: "nr = 4*min(i, nside, 4*nside - i)" },
  Bl: { n: "B<sub>ℓ</sub> — beam transfer function", t: "0 … 1", s: "A circular beam multiplies a<sub>ℓm</sub> by B<sub>ℓ</sub>; power by B<sub>ℓ</sub>². Gaussian: exp(−ℓ(ℓ+1)σ²/2).", py: "B = hp.gauss_beam(np.radians(fwhm_arcmin/60), lmax=lmax)", f: "bl = exp(-0.5_dp*l*(l+1)*sig**2)" },
  sigb: { n: "σ — beam width", t: "rad", s: "FWHM/√(8 ln 2) = FWHM/2.355. 70 GHz: 0.001616 rad.", py: "sig = fwhm / np.sqrt(8*np.log(2))", f: "sig = fwhm/sqrt(8*log(2.0_dp))" },
  FWHM: { n: "FWHM — full width at half maximum", t: "arcmin", s: "Resolution of the telescope. Planck: 13.08′ at 70 GHz, 9.66′ at 100 GHz. Must be given to healpy in radians.", py: "fwhm = np.radians(13.08/60)", f: "fwhm = 13.08_dp/60*pi/180" },
  Wl: { n: "W<sub>ℓ</sub> — pixel window", t: "0 … 1", s: "Low-pass effect of averaging over a pixel, ≈ sin x/x with x = ℓθ<sub>pix</sub>/2. Read from a table in Lab 2.", py: "W = hp.pixwin(nside, lmax=lmax)", f: "call pixel_window(pixlw, nside)" },
  Nl: { n: "N<sub>ℓ</sub> — noise power spectrum", t: "µK² sr", s: "White, uncorrelated noise: a constant N, not seen through the beam. Lab 2: 3.3×10⁻³ (70 GHz).", py: "N = cl_mask[(ell >= 2000) & (ell <= 2500)].mean()", f: "noise = sum(cl(2000:2500))/501" },
  Mll: { n: "M<sub>ℓℓ'</sub> — mode-coupling matrix", t: "depends on the mask", s: "How a mask mixes multipoles: ⟨C̃<sub>ℓ</sub>⟩ = Σ M<sub>ℓℓ'</sub>C<sub>ℓ'</sub>. Inverted by MASTER / NaMaster.", py: "# pymaster.NmtWorkspace().compute_coupling_matrix(...)", f: "! MASTER: invert M(l, l') in bins" }
};

(function () {
  const W = (window.WIDGETS = {});
  const DEG = Math.PI / 180, ARCMIN = DEG / 60;
  const el = UI.el;
  /* Planck 2018 best-fit D_l [µK²], means over bins of 10 starting at ℓ = 2 */
  const DTH = [872,865,1000,1166,1353,1557,1781,2027,2298,2591,2905,3236,3579,3926,4266,4590,4890,5156,5378,5550,5668,5723,5716,5647,5517,5332,5099,4825,4520,4195,3857,3519,3190,2880,2597,2348,2138,1970,1846,1766,1728,1729,1764,1827,1911,2010,2117,2225,2327,2418,2493,2548,2582,2593,2581,2548,2496,2428,2349,2262,2172,2085,2003,1932,1874,1833,1810,1806,1821,1853,1902,1964,2037,2116,2199,2279,2355,2422,2476,2515,2537,2540,2524,2487,2432,2360,2273,2172,2063,1946,1827,1707,1592,1482,1382,1293,1216,1154,1106,1072,1051,1044,1047,1060,1080,1104,1132,1160,1186,1209,1226,1237,1241,1237,1225,1206,1179,1146,1108,1066,1022,977,933,890,850,815,785,760,742,729,722,721,725,733,743,756,770,783,796,806,814,818,818,813,804,791,773,752,727,699,670,639,608,576,546,518,492,468,447,430,415,404,396,390,387,386,387,389,391,394,396,397,398,397,395,391,386,379,371,361,351,339,328,316,304,293,282,272,263,255,248,243,239,235,233,232,231,231,231,231,231,231,231,230,228,226,223,219,214,209,203,197,191,184,177,170,163,157,150,145,140,135,131,127,124,122,120,118,117,115,115,114,113,112,111,110,109,108,106,104,102,100,97,95,92,90,87,84,82,79];

  /* gamma(k, 1) variates (Marsaglia–Tsang), for chi-square draws */
  function gamma(k, rng) {
    if (k < 1) return gamma(k + 1, rng) * Math.pow(Math.random(), 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) { let x, v; do { x = rng.normal(); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = Math.random();
      if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; }
  }
  const chi2 = (nu, rng) => 2 * gamma(nu / 2, rng);

  /* normalised associated Legendre λ_lm(x), x = cos θ (stable recursion) */
  function lam(l, m, x) {
    const s = Math.sqrt(Math.max(0, 1 - x * x)); let p = Math.sqrt(1 / (4 * Math.PI));
    for (let k = 1; k <= m; k++) p *= -Math.sqrt((2 * k + 1) / (2 * k)) * s;
    if (l === m) return p;
    let p1 = x * Math.sqrt(2 * m + 3) * p; if (l === m + 1) return p1;
    let p0 = p;
    for (let L = m + 2; L <= l; L++) { const a = Math.sqrt((4 * L * L - 1) / (L * L - m * m)), b = Math.sqrt(((L - 1) * (L - 1) - m * m) / (4 * (L - 1) * (L - 1) - 1));
      const p2 = a * (x * p1 - b * p0); p0 = p1; p1 = p2; }
    return p1;
  }

  /* orthographic view: returns pixel → (θ, φ) or null; tilt about the x axis, spin about the pole */
  function ortho(u, v, tilt, spin) {
    const r2 = u * u + v * v; if (r2 > 1) return null;
    const z0 = Math.sqrt(1 - r2); // (u right, v up, z0 to the viewer)
    const ct = Math.cos(tilt), st = Math.sin(tilt);
    const X = u, Y = z0 * ct - v * st, Z = v * ct + z0 * st;  // tilt: the north pole leans towards the viewer
    const th = Math.acos(Math.max(-1, Math.min(1, Z))), ph = Math.atan2(Y, X) + spin;
    return [th, ph, z0];
  }
  function drawSphere(P, f, tilt, spin, o = {}) {
    const n = o.res || 180, Z = new Array(n * n).fill(NaN);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const u = (i + 0.5) / n * 2 - 1, v = 1 - (j + 0.5) / n * 2, q = ortho(u, v, tilt, spin);
      if (q) Z[j * n + i] = f(q[0], q[1]);
    }
    const lo = o.lo ?? -1, hi = o.hi ?? 1;
    const oc = document.createElement("canvas"); oc.width = n; oc.height = n; const g2 = oc.getContext("2d"), im = g2.createImageData(n, n);
    const cm = o.cmap || ((t) => { const a = [44, 90, 160], b = [247, 244, 236], c = [180, 50, 40]; return t < 0.5 ? a.map((x, k) => x + (b[k] - x) * t * 2) : b.map((x, k) => x + (c[k] - x) * (t - 0.5) * 2); });
    for (let k = 0; k < n * n; k++) { if (!isFinite(Z[k])) { im.data[4 * k + 3] = 0; continue; } const t = Math.max(0, Math.min(1, (Z[k] - lo) / (hi - lo))), c = cm(t); im.data[4 * k] = c[0]; im.data[4 * k + 1] = c[1]; im.data[4 * k + 2] = c[2]; im.data[4 * k + 3] = 255; }
    g2.putImageData(im, 0, 0);
    const g = P.ctx, S = Math.min(P.W, P.H) - 20, X0 = (P.W - S) / 2 + (o.dx || 0), Y0 = (P.H - S) / 2;
    g.imageSmoothingEnabled = true; g.drawImage(oc, X0, Y0, S, S);
    g.strokeStyle = col("ink"); g.lineWidth = 1.5; g.beginPath(); g.arc(X0 + S / 2, Y0 + S / 2, S / 2, 0, 7); g.stroke();
    return { X0, Y0, S };
  }
  /* draw a curve given on the sphere as (θ, φ) points, front side only */
  function sphereCurve(P, box, pts, tilt, spin, o = {}) {
    const g = P.ctx, ct = Math.cos(tilt), st = Math.sin(tilt); g.strokeStyle = col(o.c || "ink"); g.lineWidth = o.w || 1; g.globalAlpha = o.a ?? 0.8; g.beginPath(); let pen = false;
    for (const [th, ph0] of pts) {
      const ph = ph0 - spin, X = Math.sin(th) * Math.cos(ph), Y = Math.sin(th) * Math.sin(ph), Zc = Math.cos(th);
      // invert the view rotation of ortho(): Z = v ct + z0 st ; Y = z0 ct - v st  →  v = Z ct - Y st, z0 = Z st + Y ct
      const v = Zc * ct - Y * st, z0 = Zc * st + Y * ct, u = X;
      if (z0 < 0) { pen = false; continue; }
      const px = box.X0 + (u + 1) / 2 * box.S, py = box.Y0 + (1 - v) / 2 * box.S; pen ? g.lineTo(px, py) : g.moveTo(px, py); pen = true;
    }
    g.stroke(); g.globalAlpha = 1;
  }

  /* ------------------------------------------------------------------ 1. latitude–longitude grid */
  W.latlon = (root) => {
    UI.head(root, "A LATITUDE–LONGITUDE GRID · CELLS SHRINK TOWARDS THE POLES");
    const k = UI.kpis(root, [["n", "cells"], ["r", "largest / smallest cell"], ["p", "cells within 15° of a pole"], ["h", "HEALPix, same count"]]);
    const sp = el("div", "split"); root.append(sp);
    const A = UI.canvas(sp, 300, { pad: [10, 10, 10, 10] }), B = UI.canvas(sp, 300, { pad: [16, 14, 40, 58] });
    const c = UI.ctrls(root); let step = 10, tilt = 25 * DEG;
    UI.slider(c, { label: "grid step Δ", min: 2, max: 30, step: 1, value: step, fmt: (v) => v + "°", on: (v) => { step = v; draw(); } });
    UI.slider(c, { label: "view tilt", min: 0, max: 80, step: 1, value: 25, fmt: (v) => v + "°", on: (v) => { tilt = v * DEG; draw(); } });
    function draw() {
      A.begin({ xlim: [0, 1], ylim: [0, 1], axes: false });
      const box = drawSphere(A, (th) => Math.sin(th), tilt, 0.3, { lo: -0.4, hi: 1.6, res: 120, cmap: (t) => [240 - 40 * t, 236 - 30 * t, 226 - 50 * t] });
      for (let la = -90 + step; la < 90; la += step) { const th = (90 - la) * DEG, pts = []; for (let p = 0; p <= 360; p += 3) pts.push([th, p * DEG]); sphereCurve(A, box, pts, tilt, 0.3, { c: "blue", w: 1 }); }
      for (let lo = 0; lo < 360; lo += step) { const pts = []; for (let t = 0; t <= 180; t += 2) pts.push([t * DEG, lo * DEG]); sphereCurve(A, box, pts, tilt, 0.3, { c: "blue", w: 1 }); }
      const nth = Math.round(180 / step), nph = Math.round(360 / step), th = [], ar = [];
      for (let i = 0; i < nth; i++) { const a = i * step * DEG, b = Math.min(Math.PI, (i + 1) * step * DEG); th.push((a + b) / 2 / DEG); ar.push((Math.cos(a) - Math.cos(b)) * step * DEG / (4 * Math.PI / (nth * nph))); }
      B.begin({ xlim: [0, 180], ylim: [0, Math.max(1.7, ...ar) * 1.05], xlabel: "colatitude θ [deg]", ylabel: "cell area / mean area" });
      B.bars(Array.from({ length: nth + 1 }, (_, i) => Math.min(180, i * step)), ar, { c: "blue", a: 0.55 });
      B.hline(1, { c: "green", dash: [], w: 2.5, label: "HEALPix: all equal" });
      const n = nth * nph, pol = Math.round(n * 2 * Math.ceil(15 / step) / nth);
      k.set("n", n); k.set("r", (Math.max(...ar) / Math.min(...ar)).toFixed(1)); k.set("p", (100 * pol / n).toFixed(0) + "% of cells, 3.4% of sky");
      const ns = Math.max(1, Math.round(Math.sqrt(n / 12))); k.set("h", "N_side ≈ " + ns + " → ratio 1.0");
    }
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 2. sampling on the circle */
  W.circle = (root) => {
    UI.head(root, "SAMPLES ON A CIRCLE · N ≥ 2B + 1 OR ALIASING");
    const k = UI.kpis(root, [["need", "samples needed 2B + 1"], ["st", "status"], ["al", "k = B seen as"], ["err", "max reconstruction error"]]);
    const sp = el("div", "split"); root.append(sp);
    const A = UI.canvas(sp, 270), B = UI.canvas(sp, 270);
    const c = UI.ctrls(root); let kk = 3, N = 7, ph = 0;
    UI.slider(c, { label: "mode k of cos kφ", min: 1, max: 8, step: 1, value: kk, on: (v) => { kk = v; draw(); } });
    UI.slider(c, { label: "samples N", min: 3, max: 20, step: 1, value: N, on: (v) => { N = v; draw(); } });
    UI.slider(c, { label: "phase", min: 0, max: 6.28, step: 0.01, value: 0, fmt: (v) => v.toFixed(2), on: (v) => { ph = v; draw(); } });
    function draw() {
      const f = (p) => Math.cos(kk * p + ph), xs = M.linspace(0, 2 * Math.PI, 400);
      const s = Array.from({ length: N }, (_, n) => f(2 * Math.PI * n / N));
      const re = [], im = []; const K = []; for (let q = -Math.floor((N - 1) / 2); q <= Math.floor(N / 2); q++) { K.push(q); let a = 0, b = 0; s.forEach((v, n) => { a += v * Math.cos(q * 2 * Math.PI * n / N) / N; b -= v * Math.sin(q * 2 * Math.PI * n / N) / N; }); re.push(a); im.push(b); }
      const rec = (p) => { let v = 0; K.forEach((q, i) => { const w = (N % 2 === 0 && q === N / 2) ? 1 : 1; v += w * (re[i] * Math.cos(q * p) - im[i] * Math.sin(q * p)); }); return v; };
      A.begin({ xlim: [0, 2 * Math.PI], ylim: [-1.6, 1.6], xlabel: "φ [rad]", ylabel: "f(φ)" });
      A.line(xs, xs.map(f), { c: "ink", w: 2 }); A.line(xs, xs.map(rec), { c: "red", w: 2, dash: [6, 4] });
      A.pts(s.map((_, n) => 2 * Math.PI * n / N), s, { c: "blue", r: 4.5 });
      A.legendBox([["ink", "true cos kφ"], ["red", "DFT reconstruction", [6, 4]], ["blue", "samples"]], { bottom: true });
      B.begin({ xlim: [-10.5, 10.5], ylim: [0, 1.1], xlabel: "mode k", ylabel: "|f̂_k|" });
      K.forEach((q, i) => B.seg(q, 0, q, Math.hypot(re[i], im[i]), { c: "blue", w: 6 }));
      B.pts([kk, -kk], [0.5, 0.5], { c: "green", r: 5 }); B.text(kk + 0.4, 0.55, "true ±k", { c: "green" });
      const ok = N >= 2 * kk + 1; let al = ((kk % N) + N) % N; if (al > N / 2) al -= N;
      let e = 0; xs.forEach((p) => { e = Math.max(e, Math.abs(rec(p) - f(p))); });
      k.set("need", 2 * kk + 1); k.set("st", ok ? "exact" : "aliased"); k.set("al", ok ? "k = " + kk : "k = " + al); k.set("err", e < 1e-9 ? "< 10⁻⁹" : e.toFixed(2));
    }
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 3. spherical harmonics */
  W.ylm = (root) => {
    UI.head(root, "RE Y_ℓm ON THE SPHERE · DRAG THE SLIDERS");
    const k = UI.kpis(root, [["ty", "type"], ["sc", "angular scale 180°/ℓ"], ["nd", "nodal lines"], ["n", "coefficients up to ℓ"]]);
    const sp = el("div", "split"); root.append(sp);
    const A = UI.canvas(sp, 300, { pad: [6, 6, 6, 6] }), B = UI.canvas(sp, 300);
    const c = UI.ctrls(root); let L = 4, Mm = 2, tilt = 20 * DEG, spin = 0;
    const sm = UI.slider(c, { label: "m", min: 0, max: 4, step: 1, value: Mm, on: (v) => { Mm = v; draw(); } });
    UI.slider(c, { label: "ℓ", min: 0, max: 14, step: 1, value: L, on: (v) => { L = v; sm.max = L; if (Mm > L) { Mm = L; sm.set(L); } draw(); } });
    UI.slider(c, { label: "rotate", min: 0, max: 6.28, step: 0.02, value: 0, fmt: (v) => Math.round(v / DEG) + "°", on: (v) => { spin = v; draw(); } });
    UI.slider(c, { label: "tilt", min: -80, max: 80, step: 1, value: 20, fmt: (v) => v + "°", on: (v) => { tilt = v * DEG; draw(); } });
    function draw() {
      let mx = 1e-9; for (let t = 0; t <= 180; t++) mx = Math.max(mx, Math.abs(lam(L, Mm, Math.cos(t * DEG))));
      A.begin({ xlim: [0, 1], ylim: [0, 1], axes: false });
      drawSphere(A, (th, ph) => lam(L, Mm, Math.cos(th)) * Math.cos(Mm * ph) / mx, tilt, spin, { res: 170 });
      const ts = M.linspace(0, 180, 361);
      B.begin({ xlim: [0, 180], ylim: [-1.1, 1.1], xlabel: "colatitude θ [deg]", ylabel: "λ_ℓm(θ) / max" });
      B.hline(0, { c: "faint", dash: [] }); B.line(ts, ts.map((t) => lam(L, Mm, Math.cos(t * DEG)) / mx), { c: "purple", w: 2.5 });
      B.text(4, -1.05, `ℓ = ${L}, m = ${Mm}: ${L - Mm} zeros in 0 < θ < 180°`, { c: "muted", base: "bottom" });
      k.set("ty", Mm === 0 ? "zonal (bands)" : Mm === L ? "sectoral (slices)" : "tesseral (checkerboard)");
      k.set("sc", L ? (180 / L).toFixed(1) + "°" : "whole sky"); k.set("nd", `${L - Mm} latitude circles + ${Mm} great circles`); k.set("n", ((L + 1) ** 2) + " = (ℓ+1)²");
    }
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 4. cosmic variance */
  W.cosmicvar = (root) => {
    UI.head(root, "ONE SKY · Ĉ_ℓ / C_ℓ SCATTERS BY √(2/(2ℓ+1)) — WHATEVER THE INSTRUMENT");
    const k = UI.kpis(root, [["e2", "error at ℓ = 2"], ["e220", "bin error at ℓ ≈ 220"], ["in", "points inside ±1σ"], ["ks", "mean of 5Ĉ₂/C₂ (2000 skies)"]]);
    const sp = el("div", "split"); root.append(sp);
    const A = UI.canvas(sp, 290), B = UI.canvas(sp, 290);
    const c = UI.ctrls(root); let fsky = 1, dl = 1, seed = 1;
    UI.slider(c, { label: "f_sky", min: 0.1, max: 1, step: 0.01, value: 1, fmt: (v) => v.toFixed(2), on: (v) => { fsky = v; draw(); } });
    UI.select(c, { label: "bin width Δℓ", options: [["1", "1 (no binning)"], ["10", "10"], ["30", "30 (Lab 2)"]], on: (v) => { dl = +v; draw(); } });
    UI.button(UI.btns(root), "new sky", () => { seed++; draw(); }, "primary");
    function draw() {
      const rng = M.rng(seed * 17 + Math.round(fsky * 100)), lmax = 400, xs = [], ys = [], sg = [];
      for (let a = 2; a + dl - 1 <= lmax; a += dl) { let s = 0, nu = 0; for (let l = a; l < a + dl; l++) { const n = Math.max(1, (2 * l + 1) * fsky); s += chi2(n, rng) / n; nu += n; }
        const lc = a + (dl - 1) / 2; xs.push(lc); ys.push(s / dl); sg.push(Math.sqrt(2 / nu)); }
      A.begin({ xlim: [2, lmax], ylim: [0, 2.6], xlog: true, xlabel: "multipole ℓ", ylabel: "Ĉ_ℓ / C_ℓ" });
      A.fill(xs, xs.map((_, i) => 1 - sg[i]), xs.map((_, i) => 1 + sg[i]), { c: "green", a: 0.25 });
      A.hline(1, { c: "ink", dash: [] }); A.pts(xs, ys, { c: "blue", r: dl > 1 ? 3 : 2 });
      let inside = 0; ys.forEach((y, i) => { if (Math.abs(y - 1) < sg[i]) inside++; });
      const r2 = M.rng(seed * 7), v = Array.from({ length: 2000 }, () => chi2(5 * fsky, r2) * (1 / fsky));
      const h = M.hist(v, 0, 20, 40), xh = h.edges.slice(0, -1).map((e, i) => (e + h.edges[i + 1]) / 2);
      B.begin({ xlim: [0, 20], ylim: [0, 0.2], xlabel: "5 Ĉ₂ / C₂", ylabel: "density" });
      B.bars(h.edges, h.dens, { c: "blue" });
      const nu = 5 * fsky, xx = M.linspace(0.05, 20, 200), pdf = (x) => Math.exp((nu / 2 - 1) * Math.log(x * fsky) - x * fsky / 2 - (nu / 2) * Math.log(2) - M.lgamma(nu / 2)) * fsky;
      B.line(xx, xx.map(pdf), { c: "red", w: 2.5 }); B.text(10, 0.18, fsky < 1 ? "χ² with 5 f_sky d.o.f. (approx.)" : "χ²₅", { c: "red" });
      k.set("e2", (100 * Math.sqrt(2 / (5 * fsky))).toFixed(0) + "%"); k.set("e220", (100 * Math.sqrt(2 / (441 * fsky * dl))).toFixed(1) + "%");
      k.set("in", (100 * inside / ys.length).toFixed(0) + "% (expect 68%)"); k.set("ks", M.mean(v).toFixed(2) + " (expect 5)");
    }
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 5. HEALPix pixel centres */
  function healpixRings(ns) {
    const rings = [];
    for (let i = 1; i < 4 * ns; i++) {
      let z, n, s;
      if (i < ns) { z = 1 - i * i / (3 * ns * ns); n = 4 * i; s = 1; }
      else if (i <= 3 * ns) { z = 4 / 3 - 2 * i / (3 * ns); n = 4 * ns; s = (i - ns + 1) % 2; }
      else { const j = 4 * ns - i; z = -(1 - j * j / (3 * ns * ns)); n = 4 * j; s = 1; }
      const phis = []; for (let j = 1; j <= n; j++) phis.push(i < ns || i > 3 * ns ? (j - 0.5) * Math.PI / (n / 2) : (j - s / 2) * Math.PI / (2 * ns));
      rings.push({ i, z, n, phis });
    }
    return rings;
  }
  function moll(lat, lon) { // lon in (−π, π]; returns x in [−2√2, 2√2], y in [−√2, √2]
    let t = lat; if (Math.abs(lat) < Math.PI / 2 - 1e-9) { for (let k = 0; k < 30; k++) { const d = (2 * t + Math.sin(2 * t) - Math.PI * Math.sin(lat)) / (2 + 2 * Math.cos(2 * t)); t -= d; if (Math.abs(d) < 1e-10) break; } }
    return [-2 * Math.SQRT2 / Math.PI * lon * Math.cos(t), Math.SQRT2 * Math.sin(t)];
  }
  W.healpix = (root) => {
    UI.head(root, "HEALPIX PIXEL CENTRES · MOLLWEIDE (EQUAL-AREA) PROJECTION · CLICK A RING");
    const k = UI.kpis(root, [["np", "N_pix = 12 N_side²"], ["nr", "rings 4N_side − 1"], ["sz", "pixel size θ_pix"], ["ring", "selected ring"]]);
    const P = UI.canvas(root, (w) => Math.round(Math.min(420, w * 0.52)), { pad: [8, 8, 8, 8] });
    const c = UI.ctrls(root); let ns = 2, sel = 1, grid = false;
    UI.select(c, { label: "N_side", value: "2", options: [["1", "1 (12 pixels)"], ["2", "2 (48)"], ["4", "4 (192)"], ["8", "8 (768)"], ["16", "16 (3072)"]], on: (v) => { ns = +v; sel = Math.min(sel, 4 * ns - 1); draw(); } });
    UI.select(c, { label: "compare with", options: [["0", "nothing"], ["1", "lat–lon grid, same count"]], on: (v) => { grid = v === "1"; draw(); } });
    let rings = [];
    function draw() {
      rings = healpixRings(ns);
      P.begin({ xlim: [-2.9, 2.9], ylim: [-1.5, 1.68], axes: false });
      const g = P.ctx; g.strokeStyle = col("ink"); g.lineWidth = 1.5; g.beginPath(); g.ellipse(P.px(0), P.py(0), P.px(2 * Math.SQRT2) - P.px(0), P.py(0) - P.py(Math.SQRT2), 0, 0, 7); g.stroke();
      const r = Math.max(1.6, Math.min(6, 26 / ns));
      if (grid) {
        const n = 12 * ns * ns, nth = Math.max(2, Math.round(Math.sqrt(n / 2))), nph = Math.round(n / nth), X = [], Y = [];
        for (let a = 0; a < nth; a++) for (let b = 0; b < nph; b++) { const [x, y] = moll(Math.PI / 2 - (a + 0.5) * Math.PI / nth, (b + 0.5) * 2 * Math.PI / nph - Math.PI); X.push(x); Y.push(y); }
        P.pts(X, Y, { c: "red", r: r * 0.7, a: 0.6 });
      }
      rings.forEach((R) => { const lat = Math.asin(R.z), X = [], Y = []; R.phis.forEach((p) => { let l = p; if (l > Math.PI) l -= 2 * Math.PI; const [x, y] = moll(lat, l); X.push(x); Y.push(y); });
        P.pts(X, Y, { c: R.i === sel ? "orange" : R.i < ns || R.i > 3 * ns ? "purple" : "blue", r: R.i === sel ? r * 1.4 : r }); });
      P.text(-2.85, 1.66, "● polar-cap rings   ● equatorial rings   ● selected" + (grid ? "   ● lat–lon grid" : ""), { c: "muted", size: 11 });
      const R = rings[sel - 1];
      k.set("np", 12 * ns * ns); k.set("nr", 4 * ns - 1); k.set("sz", (58.63 / ns).toFixed(2) + "°");
      k.set("ring", `i = ${R.i}: ${R.n} px, z = ${R.z.toFixed(3)}, θ = ${(Math.acos(R.z) / DEG).toFixed(1)}°`);
    }
    P.c.addEventListener("click", (e) => { const b = P.c.getBoundingClientRect(), [x, y] = P.inv(e.clientX - b.left, e.clientY - b.top); let best = 1, d = 1e9;
      rings.forEach((R) => { const yy = moll(Math.asin(R.z), 0)[1]; if (Math.abs(yy - y) < d) { d = Math.abs(yy - y); best = R.i; } }); sel = best; draw(); });
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 6. cost of the transform */
  W.ringcost = (root) => {
    UI.head(root, "OPERATIONS FOR ONE TRANSFORM · DIRECT SUM ~L⁴ VS RINGS ~L³");
    const k = UI.kpis(root, [["d", "direct N_pix·L²"], ["r", "rings N_ring·L²"], ["s", "speed-up"], ["t", "ring time at 10⁹ op/s"]]);
    const sp = el("div", "split"); root.append(sp);
    const A = UI.canvas(sp, 290), B = UI.canvas(sp, 290);
    const c = UI.ctrls(root); let L = 2048, mode = "2";
    UI.slider(c, { label: "band limit L", min: 6, max: 13, step: 0.1, value: 11, fmt: (v) => Math.round(2 ** v), on: (v) => { L = Math.round(2 ** v); draw(); } });
    UI.select(c, { label: "round trip at", options: [["2", "ℓ_max = 2 N_side"], ["3", "ℓ_max = 3 N_side − 1"]], on: (v) => { mode = v; draw(); } });
    const ERR = { 2: [2.2e-3, 2.3e-4, 3e-5, 3.3e-6], 3: [0.15, 0.15, 0.15, 0.15] };
    function draw() {
      const Ls = M.logspace(1, 4, 60), dir = (l) => 1.33 * l ** 4, ring = (l) => 2 * l ** 3 + 1.33 * l * l * Math.log2(l);
      A.begin({ xlim: [10, 1e4], ylim: [1e3, 1e17], xlog: true, ylog: true, xlabel: "band limit L", ylabel: "operations" });
      A.line(Ls, Ls.map(dir), { c: "red", w: 2.5 }); A.line(Ls, Ls.map(ring), { c: "green", w: 2.5 }); A.vline(L, { c: "ink" });
      A.legendBox([["red", "direct ~ L⁴"], ["green", "FFT rings + Legendre ~ L³"]], { left: true });
      B.begin({ xlim: [-0.5, 3.5], ylim: [1e-7, 1], ylog: true, xlabel: "map2alm iterations", ylabel: "rms round-trip error" });
      ERR[mode].forEach((e, i) => B.seg(i, 1e-7, i, e, { c: mode === "2" ? "blue" : "red", w: 18 }));
      B.text(0.2, 0.3, mode === "2" ? "Jacobi iterations converge (N_side = 64)" : "stuck at 0.15: not resolved", { c: "muted" });
      const d = dir(L), r = ring(L);
      k.set("d", M.fmt(d, 2)); k.set("r", M.fmt(r, 2)); k.set("s", Math.round(d / r) + "×"); k.set("t", M.human ? M.human(r / 1e9) : (r / 1e9).toFixed(2) + " s");
    }
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 7. beam and pixel calculator */
  W.beamcalc = (root) => {
    UI.head(root, "BEAM AND PIXEL CALCULATOR · HOW MUCH SKY SURVIVES AT EACH ℓ");
    const k = UI.kpis(root, [["np", "pixels 12N_side²"], ["sz", "pixel size"], ["ppf", "pixels per FWHM"], ["b2", "B_ℓ² at ℓ"], ["w2", "W_ℓ² at ℓ (approx.)"], ["bw", "B_ℓ²W_ℓ²"], ["cv", "cosmic variance √(2/(2ℓ+1))"], ["as", "angular scale 180°/ℓ"]]);
    const P = UI.canvas(root, 290);
    const c = UI.ctrls(root); let fw = 13.08, ns = 1024, ell = 515;
    const sf = UI.slider(c, { label: "beam FWHM [arcmin]", min: 1, max: 120, step: 0.01, value: fw, fmt: (v) => v.toFixed(2) + "′", on: (v) => { fw = v; draw(); } });
    const sn = UI.select(c, { label: "N_side", value: "1024", options: [16, 32, 64, 128, 256, 512, 1024, 2048, 4096].map((v) => [String(v), String(v)]), on: (v) => { ns = +v; draw(); } });
    const sl = UI.slider(c, { label: "multipole ℓ", min: 2, max: 4000, step: 1, value: ell, on: (v) => { ell = v; draw(); } });
    const b = UI.btns(root);
    [["Planck 70 GHz", 13.08, 1024, 515], ["Planck 100 GHz", 9.66, 1024, 698], ["1° beam, N_side 64", 60, 64, 128], ["1° beam, N_side 128", 60, 128, 256]].forEach(([t, f, n, l]) =>
      UI.button(b, t, () => { fw = f; ns = n; ell = l; sf.set(f); sn.value = String(n); sl.set(l); draw(); }));
    const B2 = (l) => Math.exp(-l * (l + 1) * (fw * ARCMIN / Math.sqrt(8 * Math.log(2))) ** 2);
    const W2 = (l) => { const x = l * Math.sqrt(Math.PI / 3) / ns / 2; return x < 1e-9 ? 1 : (Math.sin(x) / x) ** 2; };
    function draw() {
      const lmax = Math.max(4 * ns, ell * 1.2, 600), ls = M.linspace(2, Math.min(lmax, 3 * ns * 1.3), 500);
      P.begin({ xlim: [0, ls[ls.length - 1]], ylim: [1e-4, 1.5], ylog: true, xlabel: "multipole ℓ", ylabel: "transfer (power)" });
      P.line(ls, ls.map(B2), { c: "blue", w: 2.5 }); P.line(ls, ls.map(W2), { c: "orange", w: 2.5 }); P.line(ls, ls.map((l) => B2(l) * W2(l)), { c: "ink", w: 2, dash: [6, 4] });
      P.hline(0.01, { c: "faint", label: "1% of the power left" }); P.vline(ell, { c: "red", label: "ℓ" });
      [2, 3].forEach((q) => P.vline(q * ns, { c: "purple", label: q + "N_side" }));
      P.legendBox([["blue", "B_ℓ² beam"], ["orange", "W_ℓ² pixel"], ["ink", "B_ℓ²W_ℓ²", [6, 4]]], { left: true, bottom: true });
      const th = Math.sqrt(4 * Math.PI / (12 * ns * ns)) / ARCMIN;
      k.set("np", (12 * ns * ns).toLocaleString("en")); k.set("sz", th.toFixed(2) + "′"); k.set("ppf", (fw / th).toFixed(1) + (fw / th < 2 ? " · aliasing risk" : fw / th > 5 ? " · oversampled" : " · good"));
      k.set("b2", B2(ell).toExponential(2)); k.set("w2", W2(ell).toFixed(3)); k.set("bw", (B2(ell) * W2(ell)).toExponential(2));
      k.set("cv", (100 * Math.sqrt(2 / (2 * ell + 1))).toFixed(1) + "%"); k.set("as", (180 / ell).toFixed(3) + "° = " + (10800 / ell).toFixed(1) + "′");
    }
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 8. deconvolution on a cut sky */
  W.deconv = (root) => {
    UI.head(root, "SIMULATED RECOVERY · (Ĉ_map − N) / B²W² WITH A SLIGHTLY WRONG NOISE");
    const k = UI.kpis(root, [["reach", "ℓ where noise/B² > CMB"], ["e220", "Knox error at ℓ ≈ 220"], ["bias", "bias in D_ℓ at ℓ = 1500"], ["ok", "bins within 2σ (ℓ < 800)"]]);
    const P = UI.canvas(root, 320);
    const c = UI.ctrls(root); let fw = 13.08, N = 3.3e-3, dN = 0, fsky = 0.778, seed = 3;
    UI.slider(c, { label: "beam FWHM", min: 5, max: 30, step: 0.01, value: fw, fmt: (v) => v.toFixed(2) + "′", on: (v) => { fw = v; draw(); } });
    UI.slider(c, { label: "noise N [µK² sr]", min: -4.5, max: -1.5, step: 0.01, value: Math.log10(N), fmt: (v) => (10 ** v).toExponential(1), on: (v) => { N = 10 ** v; draw(); } });
    UI.slider(c, { label: "error on N", min: -3, max: 3, step: 0.1, value: 0, fmt: (v) => (v > 0 ? "+" : "") + v.toFixed(1) + "%", on: (v) => { dN = v / 100; draw(); } });
    UI.slider(c, { label: "f_sky", min: 0.2, max: 1, step: 0.01, value: fsky, fmt: (v) => v.toFixed(2), on: (v) => { fsky = v; draw(); } });
    UI.button(UI.btns(root), "new sky", () => { seed++; draw(); });
    function draw() {
      const rng = M.rng(seed), dl = 30, xs = [], th = [], rec = [], er = [], sig = fw * ARCMIN / Math.sqrt(8 * Math.log(2));
      const B2 = (l) => Math.exp(-l * (l + 1) * sig * sig), W2 = (l) => { const x = l * 0.001 / 2; return (Math.sin(x) / x) ** 2; };
      let reach = null, ok = 0, tot = 0;
      for (let a = 2; a + dl <= 2500; a += dl) {
        const l = a + (dl - 1) / 2, Dt = DTH[Math.min(DTH.length - 1, Math.floor((l - 2) / 10))], Ct = 2 * Math.PI * Dt / (l * (l + 1)), bw = B2(l) * W2(l);
        const nu = (2 * l + 1) * fsky * dl, cmap = (Ct * bw + N) * (1 + rng.normal() * Math.sqrt(2 / nu));
        const cr = (cmap - N * (1 + dN)) / bw, D = l * (l + 1) / (2 * Math.PI), e = Math.sqrt(2 / nu) * (Ct + N / bw);
        xs.push(l); th.push(Dt); rec.push(cr * D); er.push(e * D);
        if (reach === null && l > 200 && N / bw > Ct) reach = l;
        if (l < 800) { tot++; if (Math.abs(cr * D - Dt) < 2 * e * D) ok++; }
      }
      P.begin({ xlim: [0, 2000], ylim: [-1500, 8000], xlabel: "multipole ℓ", ylabel: "D_ℓ [µK²]" });
      P.fill(xs, xs.map((_, i) => th[i] - er[i]), xs.map((_, i) => th[i] + er[i]), { c: "green", a: 0.22 });
      P.line(xs, th, { c: "ink", w: 2 }); P.pts(xs, rec, { c: "blue", r: 2.6 });
      if (reach) P.vline(reach, { c: "red", label: "noise/B² = CMB" });
      P.legendBox([["ink", "model"], ["green", "± expected error (Knox + noise)"], ["blue", "recovered, bins of 30"]]);
      const l = 1500, bias = N * dN / (B2(l) * W2(l)) * l * (l + 1) / (2 * Math.PI);
      k.set("reach", reach ? "ℓ ≈ " + Math.round(reach) : "> 2500"); k.set("e220", (100 * Math.sqrt(2 / (441 * fsky * 30))).toFixed(1) + "%");
      k.set("bias", (bias >= 0 ? "+" : "") + Math.round(bias) + " µK²"); k.set("ok", ok + " / " + tot);
    }
    root._redraw = draw; draw();
  };

  /* ------------------------------------------------------------------ 9. Lab 2 mind map */
  const LAB = "../Lab%202/index.html#", LES = "#";
  const MAP = [
    { id: "data", t: "DATA", c: "#2f84cb", sum: "Four files from the Planck 2018 release (PR3), reduced for the lab: temperature only, N<sub>side</sub> = 1024, RING order, µK<sub>CMB</sub>.", kids: [
      { t: "70 GHz map", r: "12.6 M px · rms 436 µK", w: "Planck LFI temperature map, the core of the lab. Galaxy along the equator, CMB speckles of ~100 µK elsewhere.", code: 'm = hp.read_map("data/planck_T070_n1024_uK.fits")', link: [LAB + "s2", "Step 0 on the Lab 2 page"] },
      { t: "100 GHz map", r: "1 unobserved pixel", w: "Planck HFI map, same format, for extensions A and C. Narrower beam (9.66′) and lower noise.", code: 'm100 = hp.read_map("data/planck_T100_n1024_uK.fits")', link: [LAB + "s7", "Extensions"] },
      { t: "Mask UT78", r: "f_sky = 0.778", w: "Planck common temperature mask: 1 = keep, 0 = cut. Removes the Galactic plane and bright sources.", f: "f_{\\rm sky}=\\langle\\text{mask}\\rangle", code: "fsky = mask.mean()", link: [LAB + "s4", "Step 2"] },
      { t: "Best-fit model", r: "D₂₂₀ = 5730 µK²", w: "The ΛCDM best-fit D<sub>ℓ</sub> of Planck 2018, ℓ = 2 … 2508: the yardstick for every step.", code: "Dth[2:2 + len(th)] = th[:, 1]", link: [LAB + "s3", "Step 1"] },
      { t: "Pixel window", r: "W_ℓ for N_side 1024", w: "The official W<sub>ℓ</sub> table, read from a file because hp.pixwin would download it.", code: 'W = np.atleast_2d(hp.read_cl("data/pixel_window_n1024.fits"))[0]', link: [LES + "s7", "Section 7 of this lesson"] }] },
    { id: "theory", t: "THEORY", c: "#7a4fc9", sum: "The sections of this lesson that each step relies on.", kids: [
      { t: "Y_ℓm and C_ℓ", r: "Ĉ_ℓ = Σ|a_ℓm|²/(2ℓ+1)", w: "anafast computes a<sub>ℓm</sub> by the ring transform and averages |a<sub>ℓm</sub>|² over m.", f: "\\hat C_\\ell=\\frac{1}{2\\ell+1}\\sum_m|a_{\\ell m}|^2", code: "cl = hp.anafast(m, lmax=3*nside - 1)", link: [LES + "s4", "Section 4"] },
      { t: "HEALPix", r: "12 N_side² equal pixels", w: "Equal-area pixels on rings: integrals are sums, one FFT per ring, ℓ<sub>max</sub> = 3N<sub>side</sub> − 1 = 3071.", f: "N_{\\rm pix}=12N_{\\rm side}^2", code: "nside = hp.npix2nside(m.size)", link: [LES + "s5", "Section 5"] },
      { t: "Cosmic variance · Knox", r: "1.4% at ℓ = 220", w: "Error bars of a bin of Δℓ multipoles on f<sub>sky</sub> of the sky, with the noise amplified by the beam.", f: "\\frac{\\Delta C}{C}=\\sqrt{\\frac{2}{(2\\ell+1)f_{\\rm sky}\\Delta\\ell}}", code: "err = np.sqrt(2/((2*lc+1)*fsky*dl)) * (Dthb + DNb)", link: [LES + "s4", "Section 4"] },
      { t: "Beam and pixel window", r: "B² = 0.5 at ℓ = 515", w: "Two low-pass filters: the power is multiplied by B<sub>ℓ</sub>²W<sub>ℓ</sub>².", f: "B_\\ell=e^{-\\ell(\\ell+1)\\sigma^2/2}", code: "B = hp.gauss_beam(np.radians(13.08/60), lmax=lmax)", link: [LES + "s7", "Section 7"] },
      { t: "Pseudo-C_ℓ, masks", r: "divide by f_sky", w: "A mask is a window: it lowers the power by f<sub>sky</sub>, couples neighbouring ℓ and leaks monopole/dipole power.", f: "\\langle\\tilde C_\\ell\\rangle\\approx f_{\\rm sky}C_\\ell", code: "cl_mask = hp.anafast(mm, lmax=lmax) / fsky", link: [LES + "s8", "Section 8"] },
      { t: "Deconvolution", r: "stop where N/B² > C", w: "Recovering the sky divides by B²W²: small errors are amplified enormously at high ℓ.", f: "\\hat C^{\\rm sky}_\\ell=\\frac{\\hat C_\\ell-N}{B_\\ell^2W_\\ell^2}", code: "cl_cmb = (cl_mask - N) / BW2", link: [LES + "s8", "Section 8"] }] },
    { id: "pipe", t: "PIPELINE", c: "#3ea255", sum: "Steps 0–4 of the notebook, in order. Each step removes one effect from the measurement equation.", kids: [
      { t: "Step 0 · read and look", r: "mean 86.5 µK, f_sky 0.778", w: "Read map and mask, find UNSEEN pixels (m &lt; −1e30), Mollweide plots with the colour cut at ±300 µK.", code: "bad = m < -1e30\nfsky = mask.mean()\nhp.mollview(m, min=-300, max=300)", link: [LAB + "s2", "Step 0"] },
      { t: "Step 1 · full-sky spectrum", r: "10× the model at ℓ = 220", w: "anafast on everything. The Galaxy dominates the large scales, noise (∝ ℓ² in D<sub>ℓ</sub>) the small ones: no peaks.", code: "cl_full = hp.anafast(m, lmax=lmax)", link: [LAB + "s3", "Step 1"] },
      { t: "Step 2 · cut the Galaxy", r: "monopole 35.9, dipole 4.7 µK", w: "Fit and remove monopole and dipole on the kept sky only, zero the cut, divide by f<sub>sky</sub>: the acoustic peaks appear.", code: "mm[mask == 0] = hp.UNSEEN\nmm = hp.remove_dipole(mm)\nmm[mask == 0] = 0.0\ncl_mask = hp.anafast(mm, lmax=lmax) / fsky", link: [LAB + "s4", "Step 2"] },
      { t: "Step 3 · noise bias", r: "N = 3.27×10⁻³ µK² sr", w: "At ℓ = 2000–2500 the beam has erased the sky: the flat C<sub>ℓ</sub> left is white noise. Check: C<sub>ℓ</sub>/N ≈ 1 in every band.", f: "N=\\langle\\hat C_\\ell\\rangle_{2000\\le\\ell\\le2500}", code: "N = cl_mask[(ell >= 2000) & (ell <= 2500)].mean()", link: [LAB + "s5", "Step 3"] },
      { t: "Step 4 · beam and pixels", r: "0.99 · 1.05 · 1.15 · 1.42", w: "Subtract N, divide by B²W², bin in Δℓ = 30 with Knox errors. Recovered/model in ℓ = 30–300, 300–600, 600–800, 800–1000.", f: "\\hat C^{\\rm CMB}_\\ell=\\frac{\\hat C_\\ell-N}{B_\\ell^2W_\\ell^2}", code: "cl_cmb = (cl_mask - N) / (B*W)**2", link: [LAB + "s6", "Step 4"] }] },
    { id: "ext", t: "EXTENSIONS", c: "#e6824c", sum: "Three questions that go beyond the core.", kids: [
      { t: "A · 100 GHz", r: "N = 5.1×10⁻⁴, reach ℓ ≈ 1200", w: "The same pipeline with FREQ = 100: narrower beam and less noise push the measurement to higher ℓ.", code: "FREQ = 100   # and rerun steps 0-4", link: [LAB + "s7", "Extensions"] },
      { t: "B · the leftover excess", r: "A = 4.8×10⁻³ µK² sr", w: "After the correction the spectrum is above the model by a constant in C<sub>ℓ</sub>: a white sky component seen through the beam — unresolved radio sources. χ² 5722 → 1532.", f: "\\hat C_\\ell-N=(C^{\\rm th}_\\ell+A)B_\\ell^2W_\\ell^2", code: "A = (x*y/var).sum() / (x*x/var).sum()", link: [LAB + "s7", "Extension B"] },
      { t: "C · cross 70×100", r: "1.00 at ℓ = 226", w: "Independent noise does not correlate: the cross-spectrum has no noise bias. Foregrounds are in both maps and survive.", f: "\\langle C^{70\\times100}_\\ell\\rangle=B^{70}_\\ell B^{100}_\\ell W_\\ell^2C_\\ell", code: "cl_x = hp.anafast(m70, map2=m100, lmax=lmax) / fsky_c", link: [LAB + "s7", "Extension C"] }] },
    { id: "res", t: "RESULTS", c: "#d1495b", sum: "What the lab measures, and where it stops.", kids: [
      { t: "Three acoustic peaks", r: "1–15% up to ℓ ≈ 800", w: "From a Galaxy-dominated sky to the first three peaks (ℓ ≈ 220, 540, 810) within 1–15% of the model at 70 GHz.", link: [LAB + "s8", "Results"] },
      { t: "The beam sets the reach", r: "70: ℓ ≈ 860 · 100: ℓ ≈ 1220", w: "Beyond the ℓ where N/B² exceeds the CMB, every small error is amplified: the beam, not the data, decides how far you can go.", link: [LAB + "s8", "Results"] },
      { t: "What we neglected", r: "mode coupling, beam shape…", w: "Mask coupling (MASTER), measured beams (RIMO), noise simulations and half-mission cross-spectra, foreground residuals, the degrading to N<sub>side</sub> 1024.", link: [LAB + "s8", "Results"] }] }
  ];

  W.mindmap = (root) => {
    UI.head(root, "LAB 2 · THE WHOLE PROCESS · CLICK A NODE");
    const bar = UI.btns(root);
    const host = el("div"); root.append(host);
    const panel = el("div", "mm-panel"); panel.style.marginTop = "10px"; root.append(panel);
    const open = new Set(MAP.map((b) => b.id)); let sel = ["pipe", 0], timer = null;
    const play = UI.button(bar, "▶ Play the pipeline", () => { if (timer) return stop(); let i = 0; const seq = [...MAP[2].kids.map((_, j) => ["pipe", j]), ...MAP[3].kids.map((_, j) => ["ext", j]), ...MAP[4].kids.map((_, j) => ["res", j])];
      open.add("pipe"); open.add("ext"); open.add("res"); sel = seq[0]; draw(); timer = setInterval(() => { i++; if (i >= seq.length) return stop(); sel = seq[i]; draw(); }, 2600); play.textContent = "■ Stop"; }, "primary");
    UI.button(bar, "expand all", () => { MAP.forEach((b) => open.add(b.id)); draw(); });
    UI.button(bar, "collapse all", () => { open.clear(); draw(); });
    function stop() { clearInterval(timer); timer = null; play.textContent = "▶ Play the pipeline"; }
    const esc = (s) => s.replace(/&(?!lt;|gt;|amp;)/g, "&amp;").replace(/</g, "&lt;");
    function draw() {
      const Wd = Math.max(300, host.clientWidth || root.clientWidth || 600), narrow = Wd < 560, rowH = 27, gap = 9;
      const rootW = narrow ? 0 : Math.min(140, Wd * 0.2), brX = narrow ? 6 : rootW + 22, brW = narrow ? 92 : 112, lfX = brX + brW + 22, lfW = Wd - lfX - 6;
      let y = narrow ? 44 : 10; const nodes = [], edges = [];
      MAP.forEach((b, bi) => {
        const kids = open.has(b.id) ? b.kids : [], y0 = y, h = Math.max(1, kids.length) * rowH;
        kids.forEach((kd, ki) => { const yy = y + ki * rowH; nodes.push({ x: lfX, y: yy + 2, w: lfW, h: rowH - 5, t: kd.t, r: kd.r, c: b.c, key: [b.id, ki], leaf: true });
          edges.push({ x1: brX + brW, y1: y0 + h / 2, x2: lfX, y2: yy + rowH / 2, key: [b.id, ki] }); });
        nodes.push({ x: brX, y: y0 + h / 2 - 13, w: brW, h: 26, t: (open.has(b.id) ? "▾ " : "▸ ") + b.t, c: b.c, key: [b.id, -1], branch: true });
        edges.push({ x1: narrow ? 0 : rootW, y1: null, x2: brX, y2: y0 + h / 2, key: [b.id, -1], fromRoot: true });
        y += h + gap;
      });
      const H = y + 4, ry = narrow ? 6 : H / 2 - 30;
      const rootNode = narrow ? { x: 6, y: 6, w: Wd - 12, h: 30 } : { x: 6, y: ry, w: rootW - 6, h: 60 };
      let s = `<svg class="mm-svg" viewBox="0 0 ${Wd} ${H}" width="${Wd}" height="${H}" xmlns="http://www.w3.org/2000/svg">`;
      edges.forEach((e) => { const hot = e.key[0] === sel[0] && (e.key[1] === sel[1] || e.key[1] === -1);
        if (e.fromRoot) { if (narrow) { const x = 14; s += `<path class="edge${hot ? " hot" : ""}" d="M${x} 36 L${x} ${e.y2} L${brX} ${e.y2}"/>`; } else { const y1 = ry + 30, mx = (rootW + brX) / 2; s += `<path class="edge${hot ? " hot" : ""}" d="M${rootW} ${y1} C${mx} ${y1} ${mx} ${e.y2} ${brX} ${e.y2}"/>`; } }
        else { const mx = (e.x1 + e.x2) / 2; s += `<path class="edge${hot ? " hot" : ""}" d="M${e.x1} ${e.y1} C${mx} ${e.y1} ${mx} ${e.y2} ${e.x2} ${e.y2}"/>`; } });
      s += `<g class="node"><rect x="${rootNode.x}" y="${rootNode.y}" width="${rootNode.w}" height="${rootNode.h}" rx="10" fill="var(--green)"/>` +
        (narrow ? `<text x="${rootNode.x + 10}" y="${rootNode.y + 19}" font-weight="800">LAB 2 · CMB power spectrum from Planck</text>` :
          `<text x="${rootNode.x + 8}" y="${rootNode.y + 18}" font-weight="800">LAB 2</text><text x="${rootNode.x + 8}" y="${rootNode.y + 34}">CMB C_ℓ from</text><text x="${rootNode.x + 8}" y="${rootNode.y + 49}">Planck maps</text>`) + `</g>`;
      nodes.forEach((n) => { const on = n.key[0] === sel[0] && n.key[1] === sel[1];
        const fill = n.branch ? n.c : on ? n.c + "33" : "var(--panel)";
        s += `<g class="node${on ? " on" : ""}" data-k="${n.key[0]}|${n.key[1]}"><rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${n.branch ? 8 : 6}" fill="${fill}" ${n.leaf ? `stroke="${n.c}"` : ""}/>`;
        if (n.branch) s += `<text x="${n.x + 8}" y="${n.y + 17}" font-weight="800" style="fill:#fff">${n.t}</text>`;
        else { const room = (n.t.length + (n.r || "").length) * 7.1 + 36 < n.w; s += `<rect x="${n.x}" y="${n.y}" width="5" height="${n.h}" fill="${n.c}" stroke="none"/><text x="${n.x + 12}" y="${n.y + 15}" font-weight="${on ? 800 : 600}">${esc(n.t)}</text>`;
          if (room && n.r) s += `<text x="${n.x + n.w - 8}" y="${n.y + 15}" text-anchor="end" style="fill:var(--muted)">${esc(n.r)}</text>`; }
        s += `</g>`; });
      host.innerHTML = s + "</svg>";
      host.querySelectorAll(".node[data-k]").forEach((g) => g.addEventListener("click", () => { stop(); const [b, i] = g.dataset.k.split("|"); if (+i === -1) { open.has(b) ? open.delete(b) : open.add(b); } sel = [b, +i]; draw(); }));
      // panel
      const br = MAP.find((b) => b.id === sel[0]);
      if (!br) { panel.innerHTML = ""; return; }
      if (sel[1] === -1) { panel.innerHTML = `<div class="tag" style="color:${br.c}">${br.t}</div><h4>${br.kids.length} nodes</h4><p>${br.sum}</p><p style="color:var(--muted)">Click the branch again to ${open.has(br.id) ? "collapse" : "expand"} it.</p>`; return; }
      const kd = br.kids[sel[1]];
      panel.innerHTML = `<div class="tag" style="color:${br.c}">${br.t} · ${sel[1] + 1}/${br.kids.length}</div><h4>${kd.t}</h4><p>${kd.w}</p>` +
        (kd.f ? `<p>\\(${kd.f}\\)</p>` : "") + (kd.code ? `<pre>${esc(kd.code)}</pre>` : "") +
        (kd.r ? `<div class="res"><b>Result:</b> ${esc(kd.r)}</div>` : "") + (kd.link ? `<p style="margin-top:8px"><a href="${kd.link[0]}">→ ${kd.link[1]}</a></p>` : "");
      if (window.APP && APP.renderMath) APP.renderMath(panel);
    }
    root._redraw = draw; draw();
  };
})();
