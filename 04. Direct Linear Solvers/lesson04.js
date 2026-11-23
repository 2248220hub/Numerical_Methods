/* Lesson 4 — stability, Gaussian elimination, LU, Cholesky: symbols and widgets */
window.SYMBOLS = {
  A: { n: "A — coefficient matrix", t: "n × n", s: "The linear operator of the problem: a discretised differential operator, a design matrix, a covariance…", py: "A = np.array(...)", f: "real(dp) :: A(n,n)" },
  L: { n: "L — unit lower triangular factor", t: "n × n, l<sub>ii</sub> = 1", s: "Holds the elimination multipliers l<sub>ik</sub> = m<sub>ik</sub> below the diagonal. With partial pivoting |l<sub>ik</sub>| ≤ 1.", py: "L = np.tril(lu, -1) + np.eye(n)", f: "! strict lower triangle of A after dgetrf" },
  U: { n: "U — upper triangular factor", t: "n × n", s: "The matrix left at the end of the forward reduction. Its diagonal gives the determinant.", py: "U = np.triu(lu)", f: "! upper triangle of A after dgetrf" },
  P: { n: "P — permutation matrix", t: "n × n, one 1 per row and column", s: "Records the row swaps of partial pivoting. Stored as an integer array, never as a matrix. det P = (−1)<sup>swaps</sup>.", py: "lu, piv = scipy.linalg.lu_factor(A)", f: "integer :: ipiv(n)" },
  Lc: { n: "L — Cholesky factor", t: "lower triangular, l<sub>ii</sub> > 0", s: "The unique 'square root' of an SPD matrix: A = LLᵀ.", py: "L = np.linalg.cholesky(A)", f: "call dpotrf('L', n, A, n, info)" },
  Sig: { n: "Σ — covariance matrix", t: "SPD", s: "Σ<sub>ij</sub> = Cov(Y<sub>i</sub>, Y<sub>j</sub>). Always symmetric positive (semi-)definite.", py: "Sigma = np.cov(samples)", f: "real(dp) :: Sig(d,d)" },
  Zv: { n: "Z — standard normal vector", t: "𝒩(0, I)", s: "Independent unit Gaussians; L Z has covariance L Lᵀ = Σ.", py: "Z = rng.standard_normal((d, N))", f: "! Box–Muller from random_number" },
  mu: { n: "μ — mean vector", t: "ℝ<sup>d</sup>", s: "Centre of the Gaussian.", py: "mu = np.array([1.0, -1.0])", f: "real(dp) :: mu(d)" },
  fx: { n: "f(x) — function whose root we want", t: "scalar", s: "Here x² − 10x + 9, roots 1 and 9.", py: "f = lambda x: x*x - 10*x + 9", f: "f = x*x - 10*x + 9" },
  xk: { n: "x<sub>k</sub> — iterate", t: "scalar", s: "Successive approximations to the root; stop on residual, relative correction or iteration count.", py: "x = x - f(x)/df(x)", f: "x = x - f(x)/df(x)" },
  mik: { n: "m<sub>ik</sub> — elimination multiplier", t: "scalar", s: "a<sub>ik</sub>/a<sub>kk</sub>: how many times row k is subtracted from row i. Huge if the pivot is tiny — hence pivoting.", py: "m = A[i, k] / A[k, k]", f: "m = A(i,k)/A(k,k)" },
  uii: { n: "u<sub>ii</sub> — pivot / diagonal of U", t: "scalar ≠ 0", s: "Back substitution divides by it; det A = (−1)<sup>p</sup> Π u<sub>ii</sub>.", py: "np.diag(U)", f: "[(A(i,i), i = 1, n)]" },
  yv: { n: "y — right-hand side of the triangular system", t: "ℝⁿ", s: "Pb after the forward substitution Ly = Pb.", py: "y = solve_triangular(L, b[piv], lower=True, unit_diagonal=True)", f: "! forward substitution" }
};

(function () {
  const W = (window.WIDGETS = {});
  const f32 = Math.fround;
  const fmtE = (v) => (v === 0 ? "0" : v.toExponential(1));

  /* 1. catastrophic cancellation */
  W.quadcancel = (root) => {
    UI.head(root, "ROOTS OF x² + bx + 1 = 0 · NAIVE vs STABLE FORMULA");
    const k = UI.kpis(root, [["n32", "naive, float32"], ["s32", "stable, float32"], ["n64", "naive, float64"], ["s64", "stable, float64"]]);
    const P = UI.canvas(root, 300); const c = UI.ctrls(root); let lb = 4;
    UI.slider(c, { label: "log₁₀ b", min: 1, max: 9, step: 0.05, value: lb, fmt: (v) => "b = " + Math.pow(10, v).toExponential(2), on: (v) => { lb = v; draw(); } });
    const errs = (b) => {
      const exact = 1 / (b / 2 + Math.sqrt(b * b / 4 - 1));               // small root of x² − bx + 1, b > 0
      const disc64 = Math.sqrt(b * b - 4), naive64 = (b - disc64) / 2, st64 = 1 / ((b + disc64) / 2);
      const B = f32(b), d32 = f32(Math.sqrt(f32(f32(B * B) - 4))), naive32 = f32(f32(B - d32) / 2), st32 = f32(1 / f32(f32(B + d32) / 2));
      const e = (v) => Math.abs(v - exact) / exact;
      return { n32: e(naive32), s32: e(st32), n64: e(naive64), s64: e(st64) };
    };
    const grid = M.linspace(1, 9, 161), tab = grid.map((g) => errs(Math.pow(10, g)));
    function draw() {
      P.begin({ xlim: [10, 1e9], ylim: [1e-17, 10], xlog: true, ylog: true, xlabel: "|b|", ylabel: "relative error of the small root" });
      P.hline(1.19e-7, { c: "red", label: "ε₃₂" }); P.hline(2.22e-16, { c: "blue", label: "ε₆₄" });
      const xs = grid.map((g) => Math.pow(10, g)), fl = (v) => Math.max(v, 1.5e-17);
      P.line(xs, tab.map((t) => fl(t.n32)), { c: "red", w: 2, dash: [6, 3] }); P.line(xs, tab.map((t) => fl(t.s32)), { c: "red", w: 2.5 });
      P.line(xs, tab.map((t) => fl(t.n64)), { c: "blue", w: 2, dash: [6, 3] }); P.line(xs, tab.map((t) => fl(t.s64)), { c: "blue", w: 2.5 });
      P.vline(Math.pow(10, lb), { c: "ink", dash: [] });
      const e = errs(Math.pow(10, lb)); ["n32", "s32", "n64", "s64"].forEach((kk) => k.set(kk, fmtE(e[kk])));
    }
    root._redraw = draw; draw();
    UI.legend(root, [["red", "float32"], ["blue", "float64"]]);
    UI.note(root, "Dashed: (b − √(b²−4))/2, a difference of two almost equal numbers. Solid: 1/x₁ with x₁ = (b + √(b²−4))/2, only sums. The naive error grows like b²·ε until every digit is gone (100%), first in single then in double precision; the stable one stays at the rounding level. Float32 is emulated with Math.fround.");
  };

  /* 2. cost of Cramer vs elimination */
  W.cramer = (root) => {
    UI.head(root, "COST OF A DIRECT SOLVE · (n+1)! vs n³");
    const k = UI.kpis(root, [["c", "Cramer at 1 Gflop/s"], ["g", "Gauss / LU"], ["ch", "Cholesky"]]);
    const P = UI.canvas(root, 290); const c = UI.ctrls(root); let n = 10;
    UI.slider(c, { label: "n", min: 2, max: 40, step: 1, value: n, on: (v) => { n = v; draw(); } });
    const fact = (m) => { let f = 1; for (let i = 2; i <= m; i++) f *= i; return f; };
    function draw() {
      const ns = M.linspace(2, 40, 39);
      P.begin({ xlim: [2, 40], ylim: [1, 1e50], ylog: true, xlabel: "n", ylabel: "floating-point operations" });
      P.line(ns, ns.map((m) => fact(m + 1)), { c: "red", w: 2.5 }); P.line(ns, ns.map((m) => (2 / 3) * m ** 3), { c: "blue", w: 2.5 }); P.line(ns, ns.map((m) => m ** 3 / 3), { c: "green", w: 2.5, dash: [5, 3] });
      P.hline(1e9 * 3.15e7 * 1.4e10, { c: "ink", label: "1 Gflop/s × age of the Universe" });
      P.vline(n, { c: "orange", dash: [] });
      k.set("c", M.human(fact(n + 1) / 1e9)); k.set("g", M.human((2 / 3) * n ** 3 / 1e9)); k.set("ch", M.human(n ** 3 / 3 / 1e9));
    }
    root._redraw = draw; draw();
    UI.legend(root, [["red", "Cramer ≈ (n+1)!"], ["blue", "Gauss / LU ≈ (2/3)n³"], ["green", "Cholesky ≈ (1/3)n³"]]);
  };

  /* shared: draw a matrix as a table of numbers */
  const drawMatrix = (P, Mx, o = {}) => {
    const n = Mx.length, m = Mx[0].length;
    P.begin({ xlim: [0, m + (o.extra || 0)], ylim: [0, n], axes: false });
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const c = o.color ? o.color(i, j) : null;
      if (c) P.rect(j + 0.04, n - i - 0.96, j + 0.96, n - i - 0.04, { c, a: 0.28 });
      if (o.box && o.box(i, j)) P.rect(j + 0.04, n - i - 0.96, j + 0.96, n - i - 0.04, { fill: false, stroke: "red", w: 2.2 });
      const v = Mx[i][j], s = typeof v === "string" ? v : Math.abs(v) < 1e-12 ? "0" : Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3 ? v.toExponential(1) : (+v.toFixed(3)).toString();
      P.text(j + 0.5, n - i - 0.5, s, { align: "center", base: "middle", size: o.size || 13, c: o.ink ? o.ink(i, j) : "ink" });
    }
  };

  /* 3. back substitution */
  W.backsub = (root) => {
    UI.head(root, "BACK SUBSTITUTION · FROM THE BOTTOM ROW UP");
    const k = UI.kpis(root, [["s", "step"], ["op", "operation"], ["f", "flops so far"]]);
    const P = UI.canvas(root, 280); const b = UI.btns(root);
    const U = [[2, 1, -1, 3, 1], [0, 4, 2, -1, 2], [0, 0, 3, 1, -2], [0, 0, 0, 5, 1], [0, 0, 0, 0, 2]], xT = [1, -2, 3, 0.5, -1];
    const y = U.map((r) => r.reduce((s, v, j) => s + v * xT[j], 0)), n = 5; let step = 0;
    UI.button(b, "◀ back", () => { step = Math.max(0, step - 1); draw(); }); UI.button(b, "next ▶", () => { step = Math.min(n, step + 1); draw(); }, "primary"); UI.button(b, "reset", () => { step = 0; draw(); });
    function draw() {
      const known = (i) => i >= n - step, cur = n - step;   // rows already solved: i >= n-step
      const tab = U.map((r, i) => r.map((v) => v).concat(["│", y[i], known(i) ? "x=" + (+xT[i].toFixed(3)) : "x" + (i + 1) + " ?"]));
      drawMatrix(P, tab, { color: (i, j) => (j < n && j > i && known(j) && !known(i) && i === cur - 1 ? "orange" : j < n && i === j && known(i) ? "green" : j < n && j > i ? null : null), box: (i, j) => i === cur && j === i && step > 0 && i < n, size: 13,
        ink: (i, j) => (j < n && U[i][j] === 0 ? "faint" : j === n + 2 ? (known(i) ? "green" : "muted") : "ink") });
      if (step === 0) { k.set("s", "0 / 5"); k.set("op", "start: row 5 has one unknown"); k.set("f", 0); return; }
      const i = n - step;
      const terms = U[i].slice(i + 1).map((v, jj) => (v ? `${v}·${+xT[i + 1 + jj].toFixed(3)}` : null)).filter(Boolean).join(" + ");
      k.set("s", step + " / 5"); k.set("op", `x${i + 1} = (${y[i]}${terms ? " − (" + terms + ")" : ""}) / ${U[i][i]} = ${+xT[i].toFixed(3)}`);
      let f = 0; for (let s = 1; s <= step; s++) f += 2 * (s - 1) + 1; k.set("f", f);
    }
    root._redraw = draw; draw();
    UI.note(root, "Green: diagonal entries already used (solved unknowns). Orange: the products u<sub>ij</sub>x<sub>j</sub> of the row in progress. Row i costs 2(n − i) + 1 flops: n² in total.");
  };

  /* 4. Gaussian elimination with/without pivoting */
  W.pivot = (root) => {
    UI.head(root, "GAUSSIAN ELIMINATION · STEP BY STEP, WITH AND WITHOUT PIVOTING");
    const k = UI.kpis(root, [["s", "step k"], ["m", "largest |multiplier|"], ["g", "growth max|a⁽ᵏ⁾| / max|a|"], ["e", "solution error (end)"], ["q", "unknown order"]]);
    const P = UI.canvas(root, 240); const c = UI.ctrls(root); const bt = UI.btns(root);
    let le = -10, mode = "none", step = 0;
    UI.select(c, { label: "pivoting", options: [["none", "none (naive)"], ["partial", "partial (row swaps)"], ["full", "full (rows and columns)"]], on: (v) => { mode = v; draw(); } });
    UI.slider(c, { label: "first pivot a₁₁ = 10^", min: -18, max: 0, step: 1, value: le, on: (v) => { le = v; draw(); } });
    UI.button(bt, "◀", () => { step = Math.max(0, step - 1); draw(); }); UI.button(bt, "next step ▶", () => { step = Math.min(3, step + 1); draw(); }, "primary"); UI.button(bt, "reset", () => { step = 0; draw(); });
    function run(upto) {
      const eps = Math.pow(10, le), A = [[eps, 2, 3, 1], [1, 1, 1, 1], [2, 1, 4, 2], [1, 3, 2, 5]], xt = [1, 1, 1, 1];
      const b = A.map((r) => r.reduce((s, v, j) => s + v * xt[j], 0)), q = [0, 1, 2, 3], n = 4;
      const a0 = Math.max(...A.flat().map(Math.abs)); let mmax = 0, gmax = a0, piv = null;
      for (let kk = 0; kk < upto && kk < n - 1; kk++) {
        let p = kk, r = kk;
        if (mode === "partial") for (let i = kk; i < n; i++) if (Math.abs(A[i][kk]) > Math.abs(A[p][kk])) p = i;
        if (mode === "full") for (let i = kk; i < n; i++) for (let j = kk; j < n; j++) if (Math.abs(A[i][j]) > Math.abs(A[p][r])) { p = i; r = j; }
        [A[kk], A[p]] = [A[p], A[kk]]; [b[kk], b[p]] = [b[p], b[kk]];
        if (r !== kk) { A.forEach((row) => ([row[kk], row[r]] = [row[r], row[kk]])); [q[kk], q[r]] = [q[r], q[kk]]; }
        for (let i = kk + 1; i < n; i++) { const m = A[i][kk] / A[kk][kk]; mmax = Math.max(mmax, Math.abs(m)); for (let j = kk; j < n; j++) A[i][j] -= m * A[kk][j]; b[i] -= m * b[kk]; A[i][kk] = 0; }
        gmax = Math.max(gmax, ...A.flat().map(Math.abs)); piv = kk;
      }
      return { A, b, q, mmax, g: gmax / a0, piv, xt };
    }
    function draw() {
      const R = run(step), n = 4;
      const tab = R.A.map((r, i) => r.concat(["│", R.b[i]]));
      drawMatrix(P, tab, { size: 12.5, color: (i, j) => (j < n && i > j && step > j ? "green" : null), box: (i, j) => step < 3 && i === step && j === step,
        ink: (i, j) => (j < n && Math.abs(R.A[i][j]) > 1e4 ? "red" : "ink") });
      k.set("q", "x" + R.q.map((v) => v + 1).join(", x"));
      const F = run(3); const x = new Array(n).fill(0);
      for (let i = n - 1; i >= 0; i--) { let s = F.b[i]; for (let j = i + 1; j < n; j++) s -= F.A[i][j] * x[j]; x[i] = s / F.A[i][i]; }
      const xs = new Array(n); F.q.forEach((qq, i) => (xs[qq] = x[i]));
      k.set("s", step + " / 3"); k.set("m", R.mmax ? R.mmax.toExponential(1) : "—"); k.set("g", R.g.toExponential(1));
      k.set("e", Math.max(...xs.map((v) => Math.abs(v - 1))).toExponential(1));
    }
    root._redraw = draw; draw();
    UI.note(root, "The exact solution is x = (1, 1, 1, 1) for every a₁₁. Red box: the pivot of the next step; green: zeros created; red numbers: entries above 10⁴ produced by huge multipliers. Without pivoting, a₁₁ = 10⁻¹⁰ already costs 6–7 digits and 10⁻¹⁶ destroys the solution; with pivoting the error stays ~10⁻¹⁶.");
  };

  /* 5. in-place LU with partial pivoting */
  W.lufact = (root) => {
    UI.head(root, "PA = LU IN PLACE · MULTIPLIERS FILL THE ZEROS THEY CREATE");
    const k = UI.kpis(root, [["s", "step"], ["p", "pivot vector p"], ["r", "‖PA − LU‖ (at the end)"], ["d", "det A = (−1)ˢ Π uᵢᵢ"]]);
    const P = UI.canvas(root, 250); const bt = UI.btns(root);
    const A0 = [[2, 1, 1, 0], [4, 3, 3, 1], [8, 7, 9, 5], [6, 7, 9, 8]], n = 4; let step = 0;
    UI.button(bt, "◀", () => { step = Math.max(0, step - 1); draw(); }); UI.button(bt, "next step ▶", () => { step = Math.min(3, step + 1); draw(); }, "primary"); UI.button(bt, "reset", () => { step = 0; draw(); });
    function lu(upto) {
      const A = A0.map((r) => r.slice()), p = [0, 1, 2, 3]; let sw = 0;
      for (let kk = 0; kk < upto; kk++) {
        let r = kk; for (let i = kk; i < n; i++) if (Math.abs(A[i][kk]) > Math.abs(A[r][kk])) r = i;
        if (r !== kk) { [A[kk], A[r]] = [A[r], A[kk]]; [p[kk], p[r]] = [p[r], p[kk]]; sw++; }
        for (let i = kk + 1; i < n; i++) { A[i][kk] /= A[kk][kk]; for (let j = kk + 1; j < n; j++) A[i][j] -= A[i][kk] * A[kk][j]; }
      }
      return { A, p, sw };
    }
    function draw() {
      const R = lu(step);
      drawMatrix(P, R.A, { size: 14, color: (i, j) => (j < step && i > j ? "blue" : i <= j && i < step ? "orange" : null), box: (i, j) => step < 3 && i === step && j === step });
      k.set("s", step + " / 3"); k.set("p", "[" + R.p.map((v) => v + 1).join(", ") + "]");
      if (step === 3) {
        const L = R.A.map((r, i) => r.map((v, j) => (j < i ? v : j === i ? 1 : 0))), U = R.A.map((r, i) => r.map((v, j) => (j >= i ? v : 0)));
        let err = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let s = 0; for (let t = 0; t < n; t++) s += L[i][t] * U[t][j]; err = Math.max(err, Math.abs(s - A0[R.p[i]][j])); }
        k.set("r", err.toExponential(1)); k.set("d", ((R.sw % 2 ? -1 : 1) * R.A.reduce((s, r, i) => s * r[i], 1)).toFixed(3));
      } else { k.set("r", "—"); k.set("d", "—"); }
    }
    root._redraw = draw; draw();
    UI.legend(root, [["blue", "L: multipliers lᵢₖ (unit diagonal not stored)"], ["orange", "U: finished rows"]]);
    UI.note(root, "A = [[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]]. Each step swaps the largest entry of the column to the pivot position, divides the column below it (the multipliers, all ≤ 1 in size) and updates the trailing block. At the end one array holds L and U and the integer vector p holds P.");
  };

  /* 6. Cholesky and correlated Gaussians */
  W.cholgauss = (root) => {
    UI.head(root, "Σ = LLᵀ · FROM INDEPENDENT TO CORRELATED GAUSSIAN NOISE");
    const k = UI.kpis(root, [["L", "L = [[l₁₁, 0], [l₂₁, l₂₂]]"], ["d", "det Σ = (l₁₁l₂₂)²"], ["c", "sample covariance"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 280), B = UI.canvas(box, 280); const c = UI.ctrls(root);
    let s1 = 2, s2 = 1.5, rho = 0.8;
    UI.slider(c, { label: "σ₁", min: 0.2, max: 3, step: 0.05, value: s1, on: (v) => { s1 = v; draw(); } });
    UI.slider(c, { label: "σ₂", min: 0.2, max: 3, step: 0.05, value: s2, on: (v) => { s2 = v; draw(); } });
    UI.slider(c, { label: "correlation ρ", min: -0.99, max: 0.99, step: 0.01, value: rho, fmt: (v) => v.toFixed(2), on: (v) => { rho = v; draw(); } });
    const rng = M.rng(5), N = 1500, Z = Array.from({ length: N }, () => [rng.normal(), rng.normal()]);
    function draw() {
      const S = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]];
      const l11 = Math.sqrt(S[0][0]), l21 = S[1][0] / l11, l22 = Math.sqrt(S[1][1] - l21 * l21);
      const Y = Z.map(([z1, z2]) => [l11 * z1, l21 * z1 + l22 * z2]);
      A.begin({ xlim: [-6, 6], ylim: [-6, 6], xlabel: "Z₁", ylabel: "Z₂", title: "Z ~ 𝒩(0, I)" }); A.pts(Z.map((z) => z[0]), Z.map((z) => z[1]), { c: "slate", r: 1.6, a: 0.5 });
      const th = M.linspace(0, 2 * Math.PI, 100);
      [1, 2].forEach((r) => A.line(th.map((t) => r * Math.cos(t)), th.map((t) => r * Math.sin(t)), { c: "ink", w: 1.4 }));
      B.begin({ xlim: [-6, 6], ylim: [-6, 6], xlabel: "Y₁", ylabel: "Y₂", title: "Y = L Z ~ 𝒩(0, Σ)" }); B.pts(Y.map((y) => y[0]), Y.map((y) => y[1]), { c: "blue", r: 1.6, a: 0.5 });
      [1, 2].forEach((r) => B.line(th.map((t) => r * l11 * Math.cos(t)), th.map((t) => r * (l21 * Math.cos(t) + l22 * Math.sin(t))), { c: "red", w: 1.8 }));
      const m1 = M.mean(Y.map((y) => y[0])), m2 = M.mean(Y.map((y) => y[1])), c12 = M.mean(Y.map((y) => (y[0] - m1) * (y[1] - m2)));
      k.set("L", `[[${l11.toFixed(3)}, 0], [${l21.toFixed(3)}, ${l22.toFixed(3)}]]`); k.set("d", ((l11 * l22) ** 2).toFixed(4));
      k.set("c", `${(M.std(Y.map((y) => y[0]), 0) ** 2).toFixed(2)}, ${c12.toFixed(2)}, ${(M.std(Y.map((y) => y[1]), 0) ** 2).toFixed(2)} (target ${S[0][0].toFixed(2)}, ${S[0][1].toFixed(2)}, ${S[1][1].toFixed(2)})`);
    }
    root._redraw = draw; draw();
    UI.note(root, "L maps the circles of the independent Z into the 1σ and 2σ ellipses of Σ. As |ρ| → 1, l₂₂ = σ₂√(1 − ρ²) → 0: Σ becomes singular (not positive definite) and Cholesky would fail at the second column.");
  };

  /* 7. tridiagonal systems */
  W.thomas = (root) => {
    UI.head(root, "−u″ = f ON (0, 1) · THOMAS ALGORITHM, O(n)");
    const k = UI.kpis(root, [["e", "max error"], ["fl", "Thomas ≈ 8n flops"], ["dn", "dense LU (2/3)n³"]]);
    const A = UI.canvas(root, 260); const c = UI.ctrls(root); let ln = 1.5, fk = "one";
    UI.slider(c, { label: "interior points n", min: 0.6, max: 5, step: 0.05, value: ln, fmt: (v) => Math.round(Math.pow(10, v)).toLocaleString(), on: (v) => { ln = v; draw(); } });
    UI.select(c, { label: "load f(x)", options: [["one", "f = 1  →  u = x(1−x)/2"], ["sin", "f = π² sin(πx)  →  u = sin(πx)"], ["exp", "f = −eˣ (u = eˣ − 1 − (e−1)x)"]], on: (v) => { fk = v; draw(); } });
    const fs = { one: [(x) => 1, (x) => (x * (1 - x)) / 2], sin: [(x) => Math.PI * Math.PI * Math.sin(Math.PI * x), (x) => Math.sin(Math.PI * x)], exp: [(x) => -Math.exp(x), (x) => Math.exp(x) - 1 - (Math.E - 1) * x] };
    function draw() {
      const n = Math.round(Math.pow(10, ln)), h = 1 / (n + 1), [f, ex] = fs[fk];
      const cp = new Float64Array(n), dp = new Float64Array(n), x = new Float64Array(n);
      cp[0] = -1 / 2; dp[0] = (h * h * f(h)) / 2;
      for (let i = 1; i < n; i++) { const den = 2 + cp[i - 1]; cp[i] = -1 / den; dp[i] = (h * h * f((i + 1) * h) + dp[i - 1]) / den; }
      x[n - 1] = dp[n - 1]; for (let i = n - 2; i >= 0; i--) x[i] = dp[i] - cp[i] * x[i + 1];
      let em = 0; for (let i = 0; i < n; i++) em = Math.max(em, Math.abs(x[i] - ex((i + 1) * h)));
      const xs = M.linspace(0, 1, 300), st = Math.max(1, Math.floor(n / 60)), px = [], py = [];
      for (let i = 0; i < n; i += st) { px.push((i + 1) * h); py.push(x[i]); }
      const top = Math.max(...xs.map(ex)) * 1.15;
      A.begin({ xlim: [0, 1], ylim: [Math.min(0, ...xs.map(ex)) * 1.15, top], xlabel: "x", ylabel: "u(x)" });
      A.line(xs, xs.map(ex), { c: "ink", w: 2 }); A.pts(px, py, { c: "orange", r: 3.2 });
      k.set("e", em.toExponential(2)); k.set("fl", (8 * n).toExponential(1)); k.set("dn", ((2 / 3) * n ** 3).toExponential(1));
    }
    root._redraw = draw; draw();
    UI.note(root, "The second-difference matrix tridiag(−1, 2, −1)/h² is tridiagonal: elimination never creates entries outside the band. The error first falls as h² (discretisation) then grows again for very large n as rounding accumulates — try n = 10⁵.");
  };
})();
