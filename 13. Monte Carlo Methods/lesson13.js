/* ==========================================================================
   Lesson 13 — Monte Carlo integration and sampling: symbols + widgets
   ========================================================================== */
window.SYMBOLS = {
  I: { n: "I — the integral we want", t: "target", s: "∫ f(x) dx over the domain: an area, a volume, a posterior mean…", py: "from scipy.integrate import quad\nI_exact = quad(f, a, b)[0]", f: "! exact value for testing, e.g. I = 8.0_dp/3.0_dp" },
  Abox: { n: "A<sub>box</sub> — area of the box", t: "known", s: "The simple region (square, cube) in which we throw uniform points. Must contain the figure.", py: "A_box = (x1 - x0) * (y1 - y0)", f: "A_box = (x1 - x0)*(y1 - y0)" },
  Nin: { n: "N<sub>in</sub> — number of hits", t: "count", s: "Points that fell inside the figure. Binomial: mean Np, variance Np(1−p).", py: "N_in = np.count_nonzero(inside)", f: "N_in = count(inside)" },
  N: { n: "N — number of random points", t: "count", s: "Sample size. Errors shrink like 1/√N in any dimension.", py: "n = 10**4", f: "integer, parameter :: n = 10000" },
  phat: { n: "p̂ — fraction of hits", t: "estimate", s: "N<sub>in</sub>/N, the estimate of p = A/A<sub>box</sub>. Used in place of the unknown p in the error formula.", py: "p_hat = N_in / n", f: "p_hat = real(N_in,dp)/n" },
  p: { n: "p — hit probability", t: "unknown", s: "Probability that a uniform point lands inside: the ratio of the two volumes.", py: "p = V_d / 2**d", f: "p = Vd / 2.0_dp**d" },
  dI: { n: "δI — Monte Carlo standard error", t: "error bar", s: "Standard deviation of the estimator, estimated from the same samples. Report it with every Monte Carlo number.", py: "err = (b - a) * y.std(ddof=1) / np.sqrt(n)", f: "err = (b-a)*sqrt(sum((fx-sum(fx)/n)**2)/(n-1))/sqrt(real(n,dp))" },
  fmean: { n: "⟨f⟩ — mean value of f", t: "average", s: "Average height of f over [a, b]: the integral divided by the width (mean-value theorem).", py: "f_mean = f(x).mean()", f: "fmean = sum(f(x))/n" },
  ab: { n: "b − a — width (volume V in d dims)", t: "known", s: "Size of the integration domain. In d dimensions: the volume V of the box.", py: "V = np.prod(upper - lower)", f: "V = product(upper - lower)" },
  IN: { n: "I<sub>N</sub> — the Monte Carlo estimate", t: "random", s: "A random variable: another set of points, another value. By the CLT it is Gaussian around I with width δI.", py: "I_N = (b - a) * f(rng.uniform(a, b, n)).mean()", f: "I_N = (b-a)*sum(fx)/n" },
  xi: { n: "x<sub>i</sub> — random points", t: "samples", s: "Independent uniform points in [a, b] (or in the box, for d dimensions: an n × d array).", py: "x = rng.uniform(a, b, (n, d))", f: "call random_number(x)   ! x(n,d) in [0,1)" },
  sigf: { n: "σ<sub>f</sub> — spread of f(x)", t: "std", s: "Standard deviation of f(x) when x is uniform. It sets the prefactor of the error; it can grow with d. Variance reduction = make it smaller.", py: "sigma_f = f(x).std(ddof=1)", f: "sigf = sqrt(sum((fx - sum(fx)/n)**2)/(n-1))" },
  kord: { n: "k — order of a grid rule", t: "integer", s: "Error ∝ h^k in 1-D: trapezoid k = 2, Simpson k = 4. On a product grid it becomes N^(−k/d).", py: "k = 4   # Simpson", f: "integer, parameter :: k = 4" },
  Vd: { n: "V<sub>d</sub> — volume of the unit d-ball", t: "geometry", s: "π^(d/2)/Γ(d/2+1): peaks at d ≈ 5, then goes to 0. The cube's volume 2^d explodes: the ball becomes a needle in a haystack.", py: "Vd = np.pi**(d/2) / scipy.special.gamma(d/2 + 1)", f: "Vd = pi**(d/2.0_dp)/gamma(d/2.0_dp + 1.0_dp)" },
  Gam: { n: "Γ — the Gamma function", t: "function", s: "Generalised factorial: Γ(n+1) = n!. Γ(1/2) = √π. Intrinsic GAMMA in Fortran 2008.", py: "scipy.special.gamma(x)", f: "gamma(x)" },
  Fcdf: { n: "F(x) — cumulative distribution", t: "function", s: "Probability that a sample is ≤ x. Goes from 0 to 1, monotonic.", py: "F = scipy.stats.expon(scale=1/lam).cdf", f: "F = 1.0_dp - exp(-lam*x)" },
  Finv: { n: "F⁻¹(u) — inverse CDF (quantile function)", t: "function", s: "Maps a uniform number to a sample of the target distribution.", py: "x = scipy.stats.expon(scale=1/lam).ppf(u)", f: "x = -log(1.0_dp - u)/lam" },
  u: { n: "u — uniform random number", t: "U(0,1)", s: "What the generator gives. Use 1 − u (in (0,1]) inside logarithms.", py: "u = rng.random(n)", f: "call random_number(u)" },
  lam: { n: "λ — rate of the exponential", t: "parameter", s: "Mean 1/λ. A periodogram bin is exponential with mean S, i.e. λ = 1/S.", py: "lam = 2.0", f: "lam = 2.0_dp" },
  z1: { n: "z₁ — first Gaussian deviate", t: "N(0,1)", s: "r cos(2πu₂) with r = √(−2 ln u₁).", py: "z1 = r * np.cos(2*np.pi*u2)", f: "z1 = r*cos(2*pi*u2)" },
  z2: { n: "z₂ — second Gaussian deviate", t: "N(0,1)", s: "r sin(2πu₂): independent of z₁, for free.", py: "z2 = r * np.sin(2*np.pi*u2)", f: "z2 = r*sin(2*pi*u2)" },
  u1: { n: "u₁ — uniform for the radius", t: "U(0,1]", s: "Must exclude 0: ln 0 = −∞.", py: "u1 = 1.0 - rng.random(m)", f: "u1 = 1.0_dp - u1" },
  u2: { n: "u₂ — uniform for the angle", t: "U(0,1)", s: "The angle 2πu₂ is uniform: a 2-D Gaussian is rotation-invariant.", py: "u2 = rng.random(m)", f: "call random_number(u2)" },
  Menv: { n: "M — envelope constant", t: "≥ 1", s: "Scales the proposal so that M g ≥ f everywhere. Acceptance = 1/M: as close to 1 as possible.", py: "M = np.max(f(x) / g(x))   # on a fine grid, then a bit more", f: "M = maxval(f/g) * 1.01_dp" },
  g: { n: "g(x) — proposal density", t: "density", s: "A distribution you can sample and evaluate. Must be > 0 wherever f ≠ 0 and have heavier tails than f.", py: "g = scipy.stats.norm(0, s)\nx = g.rvs(n, random_state=rng)", f: "x = s*z     ! z from Box-Muller" },
  w: { n: "w<sub>i</sub> — importance weight", t: "f/g or q/g", s: "Corrects for sampling from g instead of uniformly (or instead of the target). Watch the largest one.", py: "w = f(x) / g.pdf(x)", f: "w = f(x) / g(x)" },
  Eg: { n: "𝔼<sub>g</sub>[·] — expectation under g", t: "operator", s: "Average when x is drawn from the proposal g. I = 𝔼<sub>g</sub>[f/g].", py: "np.mean(f(x) / g.pdf(x))", f: "sum(f(x)/g(x))/n" },
  q: { n: "q(θ) = 𝓛(θ)π(θ) — unnormalised posterior", t: "function", s: "What we can evaluate. Its normalisation Z (the evidence) is unknown — and cancels in self-normalised weights.", py: "log_q = log_like(th) + log_prior(th)", f: "log_q = lnL + lnprior" },
  Neff: { n: "N<sub>eff</sub> — effective sample size", t: "diagnostic", s: "(Σw)²/Σw². How many equally weighted samples your weighted sample is worth.", py: "neff = w.sum()**2 / np.sum(w**2)", f: "neff = sum(w)**2/sum(w**2)" },
  Egw2: { n: "𝔼<sub>g</sub>[w²] — second moment of the weights", t: "number ≥ 1", s: "With normalised densities N<sub>eff</sub>/N ≈ 1/𝔼<sub>g</sub>[w²]. Multiplies across independent dimensions → exponential collapse.", py: "np.mean(w**2) / np.mean(w)**2", f: "sum(w**2)/n / (sum(w)/n)**2" },
  s: { n: "s — proposal width (relative)", t: "parameter", s: "Ratio of the proposal's width to the target's. s slightly > 1 is safe in 1-D, deadly in 100-D.", py: "g = scipy.stats.norm(0, s)", f: "x = s*z" },
  NIR: { n: "N — near-infrared reflectance", t: "band ~850 nm", s: "Vegetation reflects strongly here (leaf cell structure).", py: "NIR = cube[..., band(wl, 850)]", f: "nir = cube(band850, :, :)" },
  Rb: { n: "R — red reflectance", t: "band ~660 nm", s: "Chlorophyll absorbs red light: vegetation is dark here.", py: "R = cube[..., band(wl, 660)]", f: "red = cube(band660, :, :)" },
  post: { n: "p(θ | x) — posterior", t: "density", s: "The distribution we want to integrate over (Lecture 12).", py: "post = np.exp(lp - lp.max())", f: "post = exp(lp - maxval(lp))" }
};

(function () {
  const W = (window.WIDGETS = {});
  const bumps = (x) => Math.exp(-0.5 * ((x + 1.5) / 0.3) ** 2) + 2 * Math.exp(-0.5 * (x / 0.25) ** 2) + 1.5 * Math.exp(-0.5 * ((x - 1.2) / 0.35) ** 2);
  const BUMPS_I = Math.sqrt(2 * Math.PI) * (0.3 + 0.5 + 0.525);

  /* ---------------------------------------------------------- 1. darts */
  W.darts = (root) => {
    UI.head(root, "HIT-OR-MISS · π FROM A QUARTER CIRCLE");
    const k = UI.kpis(root, [["n", "points N"], ["in", "hits Nin"], ["pi", "π̂ ± δπ"], ["z", "(π̂ − π)/δπ"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 300, { pad: [10, 10, 30, 40], h: (w) => Math.min(320, w) }), B = UI.canvas(box, 300);
    UI.legend(root, [["green", "hit"], ["red", "miss"], ["blue", "running estimate"], ["purple", "±δπ band (from the same points)"]]);
    let rng = M.rng(13), N = 0, hin = 0, pts = [], hist = [];
    const b = UI.btns(root);
    [100, 1500, 10000, 100000].forEach((n) => UI.button(b, "+" + n.toLocaleString(), () => add(n), n === 1500 ? "go" : ""));
    UI.button(b, "reset", () => { N = 0; hin = 0; pts = []; hist = []; rng = M.rng((Math.random() * 1e9) | 0); draw(); }, "ghost");
    function add(n) {
      for (let i = 0; i < n; i++) { const x = rng(), y = rng(), h = x * x + y * y < 1; N++; if (h) hin++; if (pts.length < 6000) pts.push([x, y, h]); if (N < 50 || N % Math.ceil(N / 120) === 0) { const p = hin / N; hist.push([N, 4 * p, 4 * Math.sqrt(p * (1 - p) / N)]); } }
      draw();
    }
    function draw() {
      A.begin({ xlim: [0, 1], ylim: [0, 1], xlabel: "x", ylabel: "y" });
      const t = M.linspace(0, Math.PI / 2, 100); A.line(t.map(Math.cos), t.map(Math.sin), { c: "ink", w: 2 });
      A.pts(pts.map((p) => p[0]), pts.map((p) => p[1]), { c: pts.map((p) => (p[2] ? "green" : "red")), r: pts.length > 2000 ? 1.2 : 2.2, a: 0.8 });
      const nm = Math.max(10, N);
      B.begin({ xlim: [1, nm], ylim: [2.6, 3.7], xlog: true, xlabel: "N", ylabel: "π̂" });
      if (hist.length) { B.fill(hist.map((h) => h[0]), hist.map((h) => h[1] - h[2]), hist.map((h) => h[1] + h[2]), { c: "purple", a: 0.2 }); B.line(hist.map((h) => h[0]), hist.map((h) => h[1]), { c: "blue" }); }
      B.hline(Math.PI, { c: "ink", label: "π" });
      const p = N ? hin / N : 0, d = N ? 4 * Math.sqrt(p * (1 - p) / N) : 0;
      k.set("n", N.toLocaleString()); k.set("in", hin.toLocaleString()); k.set("pi", N ? `${(4 * p).toFixed(4)} ± ${d.toFixed(4)}` : "—"); k.set("z", N && d ? ((4 * p - Math.PI) / d).toFixed(2) + " σ" : "—");
    }
    root._redraw = draw; draw();
  };

  /* ---------------------------------------------------------- 2. sample mean integrator */
  const FUNCS = {
    x2: { f: (x) => x * x, a: 0, b: 2, I: 8 / 3, lbl: "x² on [0, 2]" },
    sin: { f: Math.sin, a: 0, b: Math.PI, I: 2, lbl: "sin x on [0, π]" },
    g: { f: (x) => Math.exp(-x * x), a: 0, b: 3, I: 0.886207348, lbl: "e^(−x²) on [0, 3]" },
    bumps: { f: bumps, a: -4, b: 4, I: BUMPS_I, lbl: "three bumps on [−4, 4]" }
  };
  W.integ = (root) => {
    UI.head(root, "SAMPLE-MEAN MONTE CARLO · AREA = WIDTH × MEAN HEIGHT");
    const k = UI.kpis(root, [["est", "estimate ± δI"], ["ex", "exact"], ["z", "deviation"], ["sf", "σ_f (from the samples)"]]);
    const P = UI.canvas(root, 290);
    UI.legend(root, [["ink", "f(x)"], ["orange", "f at the random points"], ["purple", "(b − a) × mean height"]]);
    const c = UI.ctrls(root);
    let F = FUNCS.x2, n = 20, rng = M.rng(20), xs = [];
    const gen = () => { xs = []; for (let i = 0; i < n; i++) xs.push(F.a + (F.b - F.a) * rng()); };
    UI.select(c, { label: "integrand", options: Object.entries(FUNCS).map(([kk, v]) => [kk, v.lbl]), on: (v) => { F = FUNCS[v]; gen(); draw(); } });
    UI.slider(c, { label: "number of points N", min: 1, max: 4, step: 0.05, value: Math.log10(20), fmt: (v) => Math.round(Math.pow(10, v)).toLocaleString(), on: (v) => { n = Math.round(Math.pow(10, v)); gen(); draw(); } });
    const b = UI.btns(root); UI.button(b, "new random points", () => { gen(); draw(); }, "go");
    function draw() {
      const t = M.linspace(F.a, F.b, 300), ft = t.map(F.f), ymax = Math.max(...ft) * 1.15, fx = xs.map(F.f);
      const m = M.mean(fx), sf = xs.length > 1 ? M.std(fx) : 0, I = (F.b - F.a) * m, e = (F.b - F.a) * sf / Math.sqrt(n);
      P.begin({ xlim: [F.a, F.b], ylim: [0, ymax], xlabel: "x", ylabel: "f(x)" });
      P.fill(t, 0, ft, { c: "faint", a: 0.25 });
      P.rect(F.a, 0, F.b, m, { c: "purple", a: 0.18, stroke: "purple" });
      if (n <= 400) xs.forEach((x, i) => P.seg(x, 0, x, fx[i], { c: "orange", w: 1.2, a: 0.7 }));
      P.pts(xs.slice(0, 2000), fx.slice(0, 2000), { c: "orange", r: n > 400 ? 1.3 : 3 });
      P.line(t, ft, { c: "ink", w: 2.5 });
      k.set("est", `${I.toFixed(4)} ± ${e.toFixed(4)}`); k.set("ex", F.I.toFixed(4)); k.set("z", e ? ((I - F.I) / e).toFixed(2) + " σ" : "—"); k.set("sf", sf.toFixed(4));
    }
    gen(); root._redraw = draw; draw();
  };

  /* ---------------------------------------------------------- 2b. rms vs N */
  W.rms = (root) => {
    UI.head(root, "THE 1/√N LAW, MEASURED · 200 REPETITIONS PER N");
    const k = UI.kpis(root, [["sm", "sample mean: rms·√N"], ["hm", "hit-or-miss: rms·√N"], ["r", "points ratio (hm/sm)²"]]);
    const P = UI.canvas(root, 280);
    UI.legend(root, [["blue", "sample mean (theory 2.385/√N)"], ["red", "hit-or-miss (theory 3.771/√N)"]]);
    const b = UI.btns(root); let rng = M.rng(400), res = null;
    UI.button(b, "run the experiment", () => { run(); draw(); }, "go");
    function run() {
      const Ns = M.logspace(1, 4.3, 9).map(Math.round), sm = [], hm = [];
      for (const n of Ns) {
        let s1 = 0, s2 = 0;
        for (let r = 0; r < 200; r++) {
          let a = 0, h = 0; for (let i = 0; i < n; i++) { const x = 2 * rng(); a += x * x; if (4 * rng() < x * x) h++; }
          s1 += (2 * a / n - 8 / 3) ** 2; s2 += (8 * h / n - 8 / 3) ** 2;
        }
        sm.push(Math.sqrt(s1 / 200)); hm.push(Math.sqrt(s2 / 200));
      }
      res = { Ns, sm, hm };
    }
    function draw() {
      P.begin({ xlim: [8, 3e4], ylim: [3e-3, 2], xlog: true, ylog: true, xlabel: "N", ylabel: "rms error" });
      const g = M.logspace(0.9, 4.5, 50); P.line(g, g.map((n) => 2.385 / Math.sqrt(n)), { c: "blue", w: 1.5, dash: [5, 4] }); P.line(g, g.map((n) => 3.771 / Math.sqrt(n)), { c: "red", w: 1.5, dash: [5, 4] });
      if (res) {
        P.pts(res.Ns, res.sm, { c: "blue", r: 5 }); P.pts(res.Ns, res.hm, { c: "red", r: 5 });
        const ks = M.mean(res.sm.map((v, i) => v * Math.sqrt(res.Ns[i]))), kh = M.mean(res.hm.map((v, i) => v * Math.sqrt(res.Ns[i])));
        k.set("sm", ks.toFixed(3)); k.set("hm", kh.toFixed(3)); k.set("r", ((kh / ks) ** 2).toFixed(2));
      }
    }
    root._redraw = draw; draw();
  };

  /* ---------------------------------------------------------- 3. grid vs MC */
  function simpson1d(n) { let s = 0; for (let i = 0; i < n; i++) { const x = i / (n - 1), w = i === 0 || i === n - 1 ? 1 : i % 2 ? 4 : 2; s += w * Math.PI / 2 * Math.sin(Math.PI * x); } return s / (3 * (n - 1)); }
  W.gridmc = (root) => {
    UI.head(root, "SIMPSON GRID vs MONTE CARLO · ∫ ∏ (π/2) sin(πxᵢ) dx = 1 ON [0,1]^d");
    const k = UI.kpis(root, [["sf", "σ_f in this d"], ["cross", "who wins at N = 10⁶"], ["min", "smallest Simpson grid"]]);
    const P = UI.canvas(root, 310);
    UI.legend(root, [["purple", "Simpson product rule (error ∝ N^(−4/d))"], ["red", "Monte Carlo σ_f/√N"]]);
    const c = UI.ctrls(root); let d = 4;
    UI.slider(c, { label: "dimension d", min: 1, max: 12, step: 1, value: d, on: (v) => { d = v; draw(); } });
    function draw() {
      const xs = [], ys = [];
      for (let n = 3; ; n += 2) { const N = Math.pow(n, d); if (N > 1e9) break; xs.push(N); ys.push(Math.abs(Math.pow(simpson1d(n), d) - 1)); if (n > 2001) break; }
      const sf = Math.sqrt(Math.pow(Math.PI * Math.PI / 8, d) - 1), g = M.logspace(0, 9, 60);
      P.begin({ xlim: [1, 1e9], ylim: [1e-14, 10], xlog: true, ylog: true, xlabel: "number of function evaluations N", ylabel: "error" });
      P.line(g, g.map((N) => sf / Math.sqrt(N)), { c: "red", w: 2.5, dash: [7, 4] });
      P.line(xs, ys, { c: "purple", w: 2.5 }); P.pts(xs, ys, { c: "purple", r: 3 });
      const n6 = Math.floor(Math.pow(1e6, 1 / d)), n6o = n6 % 2 ? n6 : n6 - 1;
      const eg = n6o >= 3 ? Math.abs(Math.pow(simpson1d(n6o), d) - 1) : Infinity, em = sf / 1000;
      k.set("sf", sf.toFixed(3)); k.set("cross", eg < em ? `Simpson (${eg.toExponential(1)} vs ${em.toExponential(1)})` : `Monte Carlo (${em.toExponential(1)} vs ${isFinite(eg) ? eg.toExponential(1) : "no grid"})`);
      k.set("min", "3^" + d + " = " + Math.pow(3, d).toLocaleString());
    }
    root._redraw = draw; draw();
  };

  /* ---------------------------------------------------------- 3b. ball in cube */
  W.ball = (root) => {
    UI.head(root, "THE EMPTY CUBE · VOLUME OF THE UNIT d-BALL BY HIT-OR-MISS");
    const k = UI.kpis(root, [["h", "hits"], ["est", "estimate of V_d"], ["ex", "exact V_d"], ["w", "Wilson 95% on p"]]);
    const P = UI.canvas(root, 270);
    UI.legend(root, [["ink", "exact p = V_d / 2^d"], ["orange", "your measurements"], ["red", "upper limit (0 hits)"]]);
    const c = UI.ctrls(root); let d = 10, N = 1e5, meas = []; const rng = M.rng(99);
    UI.slider(c, { label: "dimension d", min: 2, max: 24, step: 1, value: d, on: (v) => (d = v) });
    UI.select(c, { label: "points N", options: [["10000", "10⁴"], ["100000", "10⁵"], ["1000000", "10⁶ (slow at high d)"]], value: "100000", on: (v) => (N = +v) });
    const b = UI.btns(root); UI.button(b, "throw the points", () => run(), "go"); UI.button(b, "clear", () => { meas = []; draw(); }, "ghost");
    const Vd = (dd) => Math.pow(Math.PI, dd / 2) / M.gamma(dd / 2 + 1);
    function wilson(h, n) { const z = 1.96, z2 = z * z, p = h / n, c0 = (p + z2 / (2 * n)) / (1 + z2 / n), hw = z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n); return [Math.max(0, c0 - hw), c0 + hw]; }
    function run() {
      let h = 0; for (let i = 0; i < N; i++) { let r = 0; for (let j = 0; j < d; j++) { const x = 2 * rng() - 1; r += x * x; if (r > 1) break; } if (r < 1) h++; }
      const w = wilson(h, N); meas.push({ d, h, N, w });
      k.set("h", h.toLocaleString()); k.set("est", (Math.pow(2, d) * h / N).toFixed(4)); k.set("ex", Vd(d).toFixed(4)); k.set("w", `[${w[0].toExponential(2)}, ${w[1].toExponential(2)}]`);
      draw();
    }
    function draw() {
      const ds = M.linspace(2, 24, 23);
      P.begin({ xlim: [1.5, 24.5], ylim: [1e-11, 1.5], ylog: true, xlabel: "dimension d", ylabel: "p = ball / cube" });
      P.line(ds, ds.map((x) => Vd(x) / Math.pow(2, x)), { c: "ink", w: 2 });
      meas.forEach((m) => { if (m.h > 0) { P.seg(m.d, m.w[0], m.d, m.w[1], { c: "orange", w: 2 }); P.pts([m.d], [m.h / m.N], { c: "orange", r: 5, stroke: "ink" }); } else { P.seg(m.d, 1e-11, m.d, m.w[1], { c: "red", w: 2, dash: [3, 3] }); P.pts([m.d], [m.w[1]], { c: "red", r: 4 }); } });
      [1e-4, 1e-5, 1e-6].forEach((v) => P.hline(v, { c: "faint", label: "1/N = " + v.toExponential(0) }));
    }
    root._redraw = draw; draw();
  };

  /* ---------------------------------------------------------- 4. samplers */
  W.sampler = (root) => {
    UI.head(root, "SAMPLER LAB · FROM UNIFORM NUMBERS TO ANY DISTRIBUTION");
    const k = UI.kpis(root, [["n", "samples"], ["m", "mean (exact)"], ["v", "variance (exact)"], ["acc", "acceptance"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 280), B = UI.canvas(box, 280);
    const c = UI.ctrls(root);
    let mode = "inv", lam = 2, rng = M.rng(2027), s = [], u = [], extra = [], ntry = 0;
    UI.select(c, { label: "method", options: [["inv", "inversion: Exp(λ)"], ["bm", "Box–Muller: N(0,1)"], ["rej", "rejection: half-normal from Exp(1)"]], on: (v) => { mode = v; reset(); } });
    UI.slider(c, { label: "λ (inversion only)", min: 0.5, max: 4, step: 0.1, value: lam, fmt: (v) => v.toFixed(1), on: (v) => { lam = v; reset(); } });
    const b = UI.btns(root); [20, 500, 20000].forEach((n) => UI.button(b, "+" + n.toLocaleString(), () => add(n), n === 500 ? "go" : "")); UI.button(b, "reset", () => reset(), "ghost");
    const MR = Math.sqrt(2 * Math.E / Math.PI), half = (x) => Math.sqrt(2 / Math.PI) * Math.exp(-x * x / 2);
    function reset() { s = []; u = []; extra = []; ntry = 0; draw(); }
    function add(n) {
      for (let i = 0; i < n; i++) {
        if (mode === "inv") { const uu = rng(); s.push(-Math.log(1 - uu) / lam); u.push(uu); }
        else if (mode === "bm") { const u1 = 1 - rng(), u2 = rng(), r = Math.sqrt(-2 * Math.log(u1)); s.push(r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)); u.push([u1, u2]); }
        else { const x = -Math.log(1 - rng()), v = rng(); ntry++; const ok = v < half(x) / (MR * Math.exp(-x)); if (ok) s.push(x); if (extra.length < 3000) extra.push([x, v * MR * Math.exp(-x), ok]); }
      }
      draw();
    }
    function draw() {
      if (mode === "inv") {
        const t = M.linspace(0, 3, 200);
        A.begin({ xlim: [0, 3], ylim: [0, 1], xlabel: "x", ylabel: "u = F(x)", title: "u on the y-axis → x = F⁻¹(u)" });
        A.line(t, t.map((x) => 1 - Math.exp(-lam * x)), { c: "ink", w: 2.5 });
        s.slice(-25).forEach((x, i) => { const uu = u[u.length - 25 + i] ?? u[i]; A.seg(0, uu, Math.min(x, 3), uu, { c: "orange", w: 1, a: 0.7 }); A.seg(Math.min(x, 3), uu, Math.min(x, 3), 0, { c: "orange", w: 1, a: 0.7 }); });
        const h = M.hist(s, 0, 3, 40);
        B.begin({ xlim: [0, 3], ylim: [0, lam * 1.2], xlabel: "x", ylabel: "density", title: "histogram vs λe^(−λx)" });
        B.bars(h.edges, h.dens, { c: "orange" }); B.line(t, t.map((x) => lam * Math.exp(-lam * x)), { c: "ink" });
        k.set("m", s.length ? `${M.mean(s).toFixed(4)} (${(1 / lam).toFixed(4)})` : "—"); k.set("v", s.length > 1 ? `${(M.std(s) ** 2).toFixed(4)} (${(1 / lam ** 2).toFixed(4)})` : "—"); k.set("acc", "100% (no rejection)");
      } else if (mode === "bm") {
        A.begin({ xlim: [0, 1], ylim: [0, 1], xlabel: "u₁", ylabel: "u₂", title: "pairs of uniforms" });
        A.pts(u.slice(-1500).map((p) => p[0]), u.slice(-1500).map((p) => p[1]), { c: "blue", r: 1.8, a: 0.6 });
        const h = M.hist(s, -4, 4, 50), t = M.linspace(-4, 4, 200);
        B.begin({ xlim: [-4, 4], ylim: [0, 0.5], xlabel: "z", ylabel: "density", title: "z₁, z₂ vs N(0,1)" });
        B.bars(h.edges, h.dens, { c: "blue" }); B.line(t, t.map((x) => M.npdf(x)), { c: "ink" });
        k.set("m", s.length ? `${M.mean(s).toFixed(4)} (0)` : "—"); k.set("v", s.length > 1 ? `${(M.std(s) ** 2).toFixed(4)} (1)` : "—"); k.set("acc", "100% (two per pair)");
      } else {
        const t = M.linspace(0, 4, 200);
        A.begin({ xlim: [0, 4], ylim: [0, MR * 1.05], xlabel: "x", ylabel: "v · M g(x)", title: "accept under f, reject between f and Mg" });
        A.pts(extra.map((p) => p[0]), extra.map((p) => p[1]), { c: extra.map((p) => (p[2] ? "green" : "red")), r: 2, a: 0.7 });
        A.line(t, t.map((x) => MR * Math.exp(-x)), { c: "orange", w: 2.5, dash: [6, 4] }); A.line(t, t.map(half), { c: "ink", w: 2.5 });
        const h = M.hist(s, 0, 4, 40);
        B.begin({ xlim: [0, 4], ylim: [0, 1], xlabel: "x", ylabel: "density", title: "accepted samples vs half-normal" });
        B.bars(h.edges, h.dens, { c: "green" }); B.line(t, t.map(half), { c: "ink" });
        k.set("m", s.length ? `${M.mean(s).toFixed(4)} (0.7979)` : "—"); k.set("v", s.length > 1 ? `${(M.std(s) ** 2).toFixed(4)} (0.3634)` : "—"); k.set("acc", ntry ? `${(s.length / ntry).toFixed(4)} (1/M = 0.7602)` : "—");
      }
      k.set("n", s.length.toLocaleString());
    }
    root._redraw = draw; reset(); add(500);
  };

  /* ---------------------------------------------------------- 5. three bumps, importance sampling */
  W.bumps = (root) => {
    UI.head(root, "IMPORTANCE SAMPLING · THREE BUMPS ON [−4, 4], N = 1000");
    const k = UI.kpis(root, [["one", "one run: I ± δI"], ["pr", "rel. rms error, proposal"], ["un", "rel. rms error, uniform"], ["vr", "variance reduction"]]);
    const P = UI.canvas(root, 290);
    UI.legend(root, [["ink", "f(x)"], ["green", "proposal g (scaled)"], ["orange", "samples (size ∝ weight)"]]);
    const c = UI.ctrls(root); let s = 1, kind = "gauss"; const rng = M.rng(5), n = 1000;
    UI.select(c, { label: "proposal", options: [["gauss", "Gaussian N(0, s²)"], ["unif", "uniform on [−4, 4]"]], on: (v) => { kind = v; draw(); } });
    UI.slider(c, { label: "proposal width s", min: 0.3, max: 3, step: 0.05, value: s, fmt: (v) => v.toFixed(2), on: (v) => { s = v; draw(); } });
    const draw1 = (kd) => { const x = [], w = []; for (let i = 0; i < n; i++) { const xi = kd === "unif" ? -4 + 8 * rng() : s * rng.normal(); const g = kd === "unif" ? 1 / 8 : M.npdf(xi, 0, s); x.push(xi); w.push(bumps(xi) / g); } return { x, w }; };
    const relrms = (kd) => { let q = 0; for (let r = 0; r < 200; r++) { const { w } = draw1(kd); q += (M.mean(w) - BUMPS_I) ** 2; } return Math.sqrt(q / 200) / BUMPS_I; };
    function draw() {
      const { x, w } = draw1(kind), I = M.mean(w), e = M.std(w) / Math.sqrt(n), t = M.linspace(-4, 4, 400);
      P.begin({ xlim: [-4, 4], ylim: [0, 2.4], xlabel: "x", ylabel: "f(x)" });
      const gs = (xx) => (kind === "unif" ? 1 / 8 : M.npdf(xx, 0, s)), gmax = Math.max(...t.map(gs));
      P.fill(t, 0, t.map((xx) => gs(xx) / gmax * 2.1), { c: "green", a: 0.15 }); P.line(t, t.map((xx) => gs(xx) / gmax * 2.1), { c: "green", w: 1.5 });
      P.line(t, t.map(bumps), { c: "ink", w: 2.5 });
      const wm = Math.max(...w); P.pts(x.slice(0, 300), x.slice(0, 300).map(() => 0.06), { c: "orange", r: w.slice(0, 300).map((v) => 1 + 6 * Math.sqrt(v / wm)), a: 0.6 });
      const rp = relrms(kind), ru = relrms("unif");
      k.set("one", `${I.toFixed(3)} ± ${e.toFixed(3)} (exact ${BUMPS_I.toFixed(3)})`); k.set("pr", (100 * rp).toFixed(2) + "%"); k.set("un", (100 * ru).toFixed(2) + "%"); k.set("vr", ((ru / rp) ** 2).toFixed(1) + "×");
    }
    root._redraw = draw; draw();
    UI.note(root, "Try s = 0.4: the proposal's tails are lighter than the outer bumps — the error becomes worse than uniform sampling. Around s ≈ 0.7–1 the variance drops several times.");
  };

  /* ---------------------------------------------------------- 5b. tails */
  W.tails = (root) => {
    UI.head(root, "A PROPOSAL THAT FAILS SILENTLY · ∫ t₃(x) dx = 1");
    const k = UI.kpis(root, [["est", "final estimate"], ["err", "claimed error"], ["wmax", "largest weight / total"]]);
    const P = UI.canvas(root, 280);
    const c = UI.ctrls(root); let prop = "gauss", seed = 1;
    UI.select(c, { label: "proposal", options: [["gauss", "Gaussian N(0,1): lighter tails ✗"], ["cauchy", "Cauchy: heavier tails ✓"]], on: (v) => { prop = v; draw(); } });
    const b = UI.btns(root); UI.button(b, "new run (2×10⁵ points)", () => { seed++; draw(); }, "go");
    const t3 = (x) => (6 * Math.sqrt(3)) / (Math.PI * 9) * Math.pow(1 + x * x / 3, -2); // Γ(2)/(√(3π)Γ(3/2)) (1+x²/3)^-2
    function draw() {
      const rng = M.rng(seed * 7919), N = 200000, xs = [], ys = []; let sw = 0, sw2 = 0, wmax = 0;
      for (let i = 1; i <= N; i++) {
        let x, g; if (prop === "gauss") { x = rng.normal(); g = M.npdf(x); } else { x = Math.tan(Math.PI * (rng() - 0.5)); g = 1 / (Math.PI * (1 + x * x)); }
        const w = t3(x) / g; sw += w; sw2 += w * w; if (w > wmax) wmax = w;
        if (i % Math.ceil(i / 150) === 0) { xs.push(i); ys.push(sw / i); }
      }
      P.begin({ xlim: [10, N], ylim: [0.7, 1.3], xlog: true, xlabel: "N", ylabel: "running estimate" });
      P.hline(1, { c: "ink", label: "exact = 1" }); P.line(xs, ys, { c: prop === "gauss" ? "red" : "green", w: 2.5 });
      const m = sw / N, e = Math.sqrt((sw2 / N - m * m) / N);
      k.set("est", m.toFixed(4)); k.set("err", "± " + e.toFixed(4)); k.set("wmax", (100 * wmax / sw).toFixed(3) + "%");
    }
    root._redraw = draw; draw();
    UI.note(root, "With the Gaussian proposal run several seeds: most runs look fine and slightly low, then a far point arrives and the estimate jumps. The variance of the weights is infinite — the error bar lies.");
  };

  /* ---------------------------------------------------------- 6. self-normalised IS on a banana posterior */
  W.snis = (root) => {
    UI.head(root, "SELF-NORMALISED IMPORTANCE SAMPLING · A SKEWED 2-D POSTERIOR, N = 5000");
    const k = UI.kpis(root, [["mx", "E[θ₂] (truth 0.300)"], ["sy", "std θ₂ (truth)"], ["ne", "N_eff / N"], ["wm", "largest weight"]]);
    const P = UI.canvas(root, 330, { h: (w) => Math.min(380, w * 0.6) });
    UI.legend(root, [["purple", "posterior (exact, on a grid)"], ["orange", "samples from the proposal (size ∝ weight)"], ["green", "proposal 1σ, 2σ"]]);
    const c = UI.ctrls(root); let sc = 1, kind = "gauss", seed = 1;
    UI.slider(c, { label: "proposal width × Laplace", min: 0.6, max: 3, step: 0.05, value: 1, fmt: (v) => v.toFixed(2), on: (v) => { sc = v; draw(); } });
    UI.select(c, { label: "proposal family", options: [["gauss", "Gaussian (Laplace)"], ["t5", "Student t₅"]], on: (v) => { kind = v; draw(); } });
    const b = UI.btns(root); UI.button(b, "new samples", () => { seed++; draw(); }, "go");
    // banana: theta1 ~ N(0,1), theta2 | theta1 ~ N(0.3 theta1^2, 0.35^2). Mode (0,0), Laplace sd (1, 0.35).
    const logq = (x, y) => -0.5 * x * x - 0.5 * ((y - 0.3 * x * x) / 0.35) ** 2;
    const trueSy = Math.sqrt(0.35 * 0.35 + 0.09 * 2); // Var(0.3 x²) = 0.09·2
    function draw() {
      const rng = M.rng(seed * 31), n = 5000, th = [], lw = [];
      for (let i = 0; i < n; i++) {
        let z1 = rng.normal(), z2 = rng.normal(), lg;
        if (kind === "t5") { const chi = [0, 1, 2, 3, 4].reduce((s) => s + rng.normal() ** 2, 0), f = Math.sqrt(5 / chi); z1 *= f; z2 *= f; lg = -3.5 * Math.log(1 + (z1 * z1 + z2 * z2) / 5); }
        else lg = -0.5 * (z1 * z1 + z2 * z2);
        const x = sc * z1, y = sc * 0.35 * z2; th.push([x, y]); lw.push(logq(x, y) - lg);
      }
      const mx = Math.max(...lw), w = lw.map((v) => Math.exp(v - mx)), S = M.sum(w), S2 = w.reduce((a, v) => a + v * v, 0);
      const Ey = th.reduce((a, t, i) => a + w[i] * t[1], 0) / S, Ey2 = th.reduce((a, t, i) => a + w[i] * t[1] * t[1], 0) / S;
      P.begin({ xlim: [-3.5, 3.5], ylim: [-1.5, 3.5], xlabel: "θ₁", ylabel: "θ₂" });
      const gx = M.linspace(-3.5, 3.5, 70), gy = M.linspace(-1.5, 3.5, 50);
      for (const x of gx) for (const y of gy) { const p = Math.exp(logq(x, y)); if (p > 0.01) P.rect(x - 0.05, y - 0.05, x + 0.05, y + 0.05, { c: "purple", a: 0.6 * p }); }
      const wm = Math.max(...w);
      P.pts(th.slice(0, 1500).map((t) => t[0]), th.slice(0, 1500).map((t) => t[1]), { c: "orange", r: w.slice(0, 1500).map((v) => 0.6 + 5 * Math.sqrt(v / wm)), a: 0.55 });
      const a = M.linspace(0, 2 * Math.PI, 100); [1, 2].forEach((r) => P.line(a.map((t) => r * sc * Math.cos(t)), a.map((t) => r * sc * 0.35 * Math.sin(t)), { c: "green", w: 1.5 }));
      k.set("mx", Ey.toFixed(3)); k.set("sy", `${Math.sqrt(Math.max(0, Ey2 - Ey * Ey)).toFixed(3)} (${trueSy.toFixed(3)})`); k.set("ne", (S * S / S2 / n).toFixed(3)); k.set("wm", (100 * wm / S).toFixed(2) + "%");
    }
    root._redraw = draw; draw();
    UI.note(root, "The Laplace Gaussian is built at the mode (0, 0): it cannot see the banana bending upward, so with width ×1 the mean of θ₂ comes out low and a few points carry large weights. Widen it or use a Student t — then compare N_eff and the largest weight.");
  };

  /* ---------------------------------------------------------- 6b. ESS vs dimension */
  W.essd = (root) => {
    UI.head(root, "ESS / N vs DIMENSION · TARGET N(0, I), PROPOSAL N(0, s² I)");
    const k = UI.kpis(root, [["t", "theory at d"], ["m", "measured (N = 2×10⁴)"]]);
    const P = UI.canvas(root, 280);
    const c = UI.ctrls(root); let s = 1.2, d = 20; const rng = M.rng(8);
    UI.slider(c, { label: "proposal width s", min: 1.02, max: 3, step: 0.01, value: s, fmt: (v) => v.toFixed(2), on: (v) => { s = v; draw(); } });
    UI.slider(c, { label: "dimension d (for the measurement)", min: 1, max: 100, step: 1, value: d, on: (v) => { d = v; draw(); } });
    const b = UI.btns(root); UI.button(b, "measure at this d", () => { measure(); }, "go");
    const th = (dd) => Math.pow(s * s / Math.sqrt(2 * s * s - 1), -dd);
    let meas = [];
    function measure() {
      const n = 20000, lw = [];
      for (let i = 0; i < n; i++) { let a = 0; for (let j = 0; j < d; j++) { const z = rng.normal(), x = s * z; a += -0.5 * x * x + 0.5 * z * z + Math.log(s); } lw.push(a); }
      const mx = Math.max(...lw), w = lw.map((v) => Math.exp(v - mx)), S = M.sum(w), S2 = w.reduce((q, v) => q + v * v, 0);
      meas.push([d, S * S / S2 / n, s]); draw();
    }
    function draw() {
      const ds = M.linspace(1, 100, 100);
      P.begin({ xlim: [1, 100], ylim: [1e-8, 1.5], ylog: true, xlabel: "dimension d", ylabel: "ESS / N" });
      P.line(ds, ds.map(th), { c: "red", w: 2.5 }); P.hline(1 / 20000, { c: "faint", label: "ESS = 1 for N = 2×10⁴" });
      meas.filter((m) => Math.abs(m[2] - s) < 1e-9).forEach((m) => P.pts([m[0]], [m[1]], { c: "orange", r: 5, stroke: "ink" }));
      P.vline(d, { c: "orange" });
      k.set("t", th(d).toExponential(2)); const last = meas.filter((m) => m[0] === d && Math.abs(m[2] - s) < 1e-9).pop(); k.set("m", last ? last[1].toExponential(2) : "press measure");
    }
    root._redraw = draw; draw();
  };

  /* ---------------------------------------------------------- 7. central limit theorem */
  W.clt = (root) => {
    UI.head(root, "WHY THE ERROR BAR WORKS · 3000 REPEATED MONTE CARLO INTEGRALS OF x² ON [0, 2]");
    const k = UI.kpis(root, [["m", "mean of the estimates"], ["s", "their std (measured)"], ["p", "predicted 2.385/√N"]]);
    const P = UI.canvas(root, 280); const c = UI.ctrls(root); let N = 30; const rng = M.rng(3);
    UI.slider(c, { label: "points per integral N", min: 0.5, max: 3.5, step: 0.05, value: Math.log10(30), fmt: (v) => Math.round(Math.pow(10, v)).toLocaleString(), on: (v) => { N = Math.max(3, Math.round(Math.pow(10, v))); draw(); } });
    function draw() {
      const est = []; for (let r = 0; r < 3000; r++) { let s = 0; for (let i = 0; i < N; i++) { const x = 2 * rng(); s += x * x; } est.push(2 * s / N); }
      const pr = 2.385 / Math.sqrt(N), lo = 8 / 3 - 4.5 * pr, hi = 8 / 3 + 4.5 * pr, h = M.hist(est, lo, hi, 50), t = M.linspace(lo, hi, 200);
      P.begin({ xlim: [lo, hi], ylim: [0, Math.max(...h.dens, M.npdf(0, 0, pr)) * 1.15], xlabel: "I_N (one Monte Carlo estimate)", ylabel: "density" });
      P.bars(h.edges, h.dens, { c: "blue" }); P.line(t, t.map((x) => M.npdf(x, 8 / 3, pr)), { c: "ink", w: 2.5 }); P.vline(8 / 3, { c: "red", label: "8/3", dash: [] });
      k.set("m", M.mean(est).toFixed(4)); k.set("s", M.std(est).toFixed(4)); k.set("p", pr.toFixed(4));
    }
    root._redraw = draw; draw();
    UI.note(root, "Even for N = 3 the mean is right (unbiased) but the histogram is skewed; from N ≈ 30 it is a Gaussian of width σ_f/√N — the central limit theorem behind every Monte Carlo error bar.");
  };

})();
