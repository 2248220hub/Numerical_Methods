/* Lesson 5 — norms, conditioning, refinement, EVD, SVD, least squares: symbols and widgets */
window.SYMBOLS = {
  n2: { n: "‖x‖₂ — Euclidean norm", t: "≥ 0", s: "Length of a vector, rms × √n, energy. The norm that orthogonal matrices preserve.", py: "np.linalg.norm(x)", f: "norm2(x)" },
  nA: { n: "‖A‖ — induced matrix norm", t: "≥ 0", s: "The largest factor by which A stretches a vector: max ‖Ax‖/‖x‖.", py: "np.linalg.norm(A, 2)", f: "! s(1) from dgesvd" },
  smax: { n: "σ<sub>max</sub> — largest singular value", t: "≥ 0", s: "‖A‖₂: the longest semi-axis of the image of the unit sphere.", py: "np.linalg.svd(A, compute_uv=False)[0]", f: "s(1)" },
  kappa: { n: "κ(A) — condition number", t: "≥ 1", s: "‖A‖‖A⁻¹‖ = σ<sub>max</sub>/σ<sub>min</sub> (2-norm): worst-case amplification of relative errors from data to solution.", py: "np.linalg.cond(A)", f: "s(1)/s(n)" },
  epsm: { n: "ε<sub>mach</sub> — machine epsilon", t: "dimensionless", s: "Gap between 1 and the next float: 2.22×10⁻¹⁶ (double), 1.19×10⁻⁷ (single). Rounding errors are ≤ ε/2.", py: "np.finfo(float).eps", f: "epsilon(1.0_dp)" },
  Qm: { n: "Q — orthogonal eigenvector matrix", t: "QᵀQ = I", s: "Columns are the orthonormal eigenvectors of a symmetric matrix.", py: "w, Q = np.linalg.eigh(A)", f: "call dsyev('V', 'U', n, A, n, w, work, lwork, info)" },
  Lam: { n: "Λ — diagonal of eigenvalues", t: "real for symmetric A", s: "The stretch factors along the eigenvectors.", py: "np.diag(w)", f: "w(1:n)" },
  A: { n: "A — matrix of the problem", t: "m × n", s: "Square or rectangular, full rank or not: the SVD exists for all.", py: "A = np.array(...)", f: "real(dp) :: A(m,n)" },
  Um: { n: "U — left singular vectors", t: "m × m orthogonal", s: "Orthonormal basis of the output space; u₁…u<sub>r</sub> span the column space.", py: "U, s, Vt = np.linalg.svd(A)", f: "U(m,m) from dgesvd" },
  Sigm: { n: "Σ — singular values", t: "m × n diagonal, σᵢ ≥ 0", s: "Stretch factors, sorted in decreasing order. Zeros mark the null space.", py: "s", f: "s(min(m,n))" },
  Vm: { n: "V — right singular vectors", t: "n × n orthogonal", s: "Orthonormal basis of the input space; v<sub>r+1</sub>…v<sub>n</sub> span the null space.", py: "V = Vt.T", f: "VT(n,n) from dgesvd" },
  sigi: { n: "σᵢ — i-th singular value", t: "≥ 0", s: "√λᵢ(AᵀA). A vᵢ = σᵢ uᵢ.", py: "s[i]", f: "s(i)" },
  Ap: { n: "A⁺ — Moore–Penrose pseudoinverse", t: "n × m", s: "VΣ⁺Uᵀ: least-squares solution for m > n, minimum-norm solution for m < n, safe for rank-deficient A.", py: "np.linalg.pinv(A, rcond=1e-13)", f: "call dgelsd(...)" }
};

(function () {
  const W = (window.WIDGETS = {});
  const f32 = Math.fround;

  /* ---- small dense linear algebra in JS ---- */
  const luSolve = (A0, b0, single = false) => {           // Gaussian elimination with partial pivoting
    const r = single ? f32 : (v) => v, n = b0.length, A = A0.map((row) => row.map(r)), b = b0.map(r);
    for (let k = 0; k < n; k++) {
      let p = k; for (let i = k + 1; i < n; i++) if (Math.abs(A[i][k]) > Math.abs(A[p][k])) p = i;
      [A[k], A[p]] = [A[p], A[k]]; [b[k], b[p]] = [b[p], b[k]];
      for (let i = k + 1; i < n; i++) { const m = r(A[i][k] / A[k][k]); for (let j = k; j < n; j++) A[i][j] = r(A[i][j] - r(m * A[k][j])); b[i] = r(b[i] - r(m * b[k])); }
    }
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let j = i + 1; j < n; j++) s = r(s - r(A[i][j] * x[j])); x[i] = r(s / A[i][i]); }
    return x;
  };
  const luFactor = (A0, single) => {                       // returns {A, piv} with L\U stored in place
    const r = single ? f32 : (v) => v, n = A0.length, A = A0.map((row) => row.map(r)), piv = [];
    for (let k = 0; k < n; k++) {
      let p = k; for (let i = k + 1; i < n; i++) if (Math.abs(A[i][k]) > Math.abs(A[p][k])) p = i;
      piv.push(p); [A[k], A[p]] = [A[p], A[k]];
      for (let i = k + 1; i < n; i++) { A[i][k] = r(A[i][k] / A[k][k]); for (let j = k + 1; j < n; j++) A[i][j] = r(A[i][j] - r(A[i][k] * A[k][j])); }
    }
    return { A, piv, r };
  };
  const luApply = (F, b0) => {
    const { A, piv, r } = F, n = A.length, b = b0.map(r);
    piv.forEach((p, k) => { [b[k], b[p]] = [b[p], b[k]]; });
    for (let i = 1; i < n; i++) { let s = b[i]; for (let j = 0; j < i; j++) s = r(s - r(A[i][j] * b[j])); b[i] = s; }
    for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let j = i + 1; j < n; j++) s = r(s - r(A[i][j] * b[j])); b[i] = r(s / A[i][i]); }
    return b;
  };
  const matvec = (A, x) => A.map((row) => row.reduce((s, v, j) => s + v * x[j], 0));
  const nrm = (x) => Math.sqrt(x.reduce((s, v) => s + v * v, 0));
  const randOrth = (n, rng) => {                           // modified Gram–Schmidt on a random matrix
    const Q = Array.from({ length: n }, () => Array.from({ length: n }, () => rng.normal()));
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < j; i++) { let d = 0; for (let k = 0; k < n; k++) d += Q[k][i] * Q[k][j]; for (let k = 0; k < n; k++) Q[k][j] -= d * Q[k][i]; }
      let s = 0; for (let k = 0; k < n; k++) s += Q[k][j] ** 2; s = Math.sqrt(s); for (let k = 0; k < n; k++) Q[k][j] /= s;
    }
    return Q;
  };
  /* one-sided Jacobi SVD of an m × n matrix (n small): returns {s, U, V} with s descending */
  const svd = (A0) => {
    const m = A0.length, n = A0[0].length, A = A0.map((r) => r.slice()), V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
        let a = 0, b = 0, c = 0; for (let i = 0; i < m; i++) { a += A[i][p] ** 2; b += A[i][q] ** 2; c += A[i][p] * A[i][q]; }
        if (Math.abs(c) <= 1e-15 * Math.sqrt(a * b)) continue; off = Math.max(off, Math.abs(c) / Math.sqrt(a * b));
        const z = (b - a) / (2 * c), t = Math.sign(z || 1) / (Math.abs(z) + Math.sqrt(1 + z * z)), cs = 1 / Math.sqrt(1 + t * t), sn = cs * t;
        for (let i = 0; i < m; i++) { const x = A[i][p], y = A[i][q]; A[i][p] = cs * x - sn * y; A[i][q] = sn * x + cs * y; }
        for (let i = 0; i < n; i++) { const x = V[i][p], y = V[i][q]; V[i][p] = cs * x - sn * y; V[i][q] = sn * x + cs * y; }
      }
      if (off < 1e-15) break;
    }
    const s = Array.from({ length: n }, (_, j) => Math.sqrt(A.reduce((t, r) => t + r[j] ** 2, 0)));
    const ord = s.map((v, i) => i).sort((i, j) => s[j] - s[i]);
    return { s: ord.map((i) => s[i]), U: A.map((r) => ord.map((i) => (s[i] > 0 ? r[i] / s[i] : 0))), V: V.map((r) => ord.map((i) => r[i])) };
  };

  /* 1. unit balls and a matrix acting on the 2-norm circle */
  W.normball = (root) => {
    UI.head(root, "UNIT BALLS OF ℓ₁, ℓ₂, ℓ∞ · AND THE STRETCH OF A MATRIX");
    const k = UI.kpis(root, [["n1", "‖A‖₁ col sums"], ["ni", "‖A‖∞ row sums"], ["n2", "‖A‖₂ = σ_max"], ["nf", "‖A‖_F"]]);
    const box = UI.el("div", "split"); root.append(box); const L = UI.canvas(box, 290), R = UI.canvas(box, 290); const c = UI.ctrls(root);
    let a = 2, b = 1, cc = 0.5, d = 1;
    [["a₁₁", () => a, (v) => (a = v)], ["a₁₂", () => b, (v) => (b = v)], ["a₂₁", () => cc, (v) => (cc = v)], ["a₂₂", () => d, (v) => (d = v)]].forEach(([lab, g, s]) =>
      UI.slider(c, { label: lab, min: -3, max: 3, step: 0.05, value: g(), fmt: (v) => v.toFixed(2), on: (v) => { s(v); draw(); } }));
    function draw() {
      const th = M.linspace(0, 2 * Math.PI, 400);
      L.begin({ xlim: [-1.6, 1.6], ylim: [-1.6, 1.6], xlabel: "x₁", ylabel: "x₂", title: "‖x‖ = 1" });
      L.line([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], { c: "red", w: 2.2 }); L.line(th.map(Math.cos), th.map(Math.sin), { c: "blue", w: 2.2 }); L.line([1, -1, -1, 1, 1], [1, 1, -1, -1, 1], { c: "green", w: 2.2 });
      const Am = [[a, b], [cc, d]], S = svd(Am), lim = Math.max(1.5, S.s[0] * 1.2);
      R.begin({ xlim: [-lim, lim], ylim: [-lim, lim], xlabel: "(Ax)₁", ylabel: "(Ax)₂", title: "A × unit circle" });
      R.line(th.map(Math.cos), th.map(Math.sin), { c: "faint", w: 1.2, dash: [4, 3] });
      R.line(th.map((t) => a * Math.cos(t) + b * Math.sin(t)), th.map((t) => cc * Math.cos(t) + d * Math.sin(t)), { c: "blue", w: 2.5 });
      [0, 1].forEach((i) => R.seg(0, 0, S.s[i] * S.U[0][i], S.s[i] * S.U[1][i], { c: i ? "green" : "red", w: 2.5 }));
      k.set("n1", Math.max(Math.abs(a) + Math.abs(cc), Math.abs(b) + Math.abs(d)).toFixed(3)); k.set("ni", Math.max(Math.abs(a) + Math.abs(b), Math.abs(cc) + Math.abs(d)).toFixed(3));
      k.set("n2", S.s[0].toFixed(3)); k.set("nf", Math.sqrt(a * a + b * b + cc * cc + d * d).toFixed(3));
    }
    root._redraw = draw; draw();
    UI.legend(root, [["red", "ℓ₁ diamond / σ₁u₁"], ["blue", "ℓ₂ circle / its image"], ["green", "ℓ∞ square / σ₂u₂"]]);
    UI.note(root, "Right: the unit circle mapped by A is an ellipse; its longest semi-axis is ‖A‖₂ = σ₁, its shortest σ₂ (= 1/‖A⁻¹‖₂). Always ‖A‖₂ ≤ ‖A‖_F ≤ √2 ‖A‖₂ here.");
  };

  /* 2. two lines, nearly parallel */
  W.condlines = (root) => {
    UI.head(root, "SOLVING A 2×2 SYSTEM = CROSSING TWO LINES");
    const k = UI.kpis(root, [["k", "κ∞(A)"], ["x", "solution x"], ["xp", "perturbed solution"], ["amp", "relative change: x / b"]]);
    const P = UI.canvas(root, 320); const c = UI.ctrls(root); let le = -1, db = 0.05;
    UI.slider(c, { label: "ε in A = [[1, 1], [1, 1+ε]]", min: -6, max: 0.5, step: 0.05, value: le, fmt: (v) => "10^" + v.toFixed(2), on: (v) => { le = v; draw(); } });
    UI.slider(c, { label: "perturbation δb₂ (in units of ε)", min: -1, max: 1, step: 0.01, value: db, fmt: (v) => v.toFixed(2), on: (v) => { db = v; draw(); } });
    function draw() {
      const e = Math.pow(10, le), b = [2, 2 + e], bp = [2, 2 + e + db * e];
      const sol = (bb) => { const x2 = (bb[1] - bb[0]) / e; return [bb[0] - x2, x2]; };
      const x = sol(b), xp = sol(bp);
      P.begin({ xlim: [-1.5, 3.5], ylim: [-1, 3], xlabel: "x₁", ylabel: "x₂" });
      const xs = [-1.5, 3.5];
      P.line(xs, xs.map((t) => 2 - t), { c: "blue", w: 2.5 }); P.line(xs, xs.map((t) => (b[1] - t) / (1 + e)), { c: "red", w: 2.5 }); P.line(xs, xs.map((t) => (bp[1] - t) / (1 + e)), { c: "red", w: 1.6, dash: [6, 4] });
      P.pts([x[0]], [x[1]], { c: "ink", r: 6 }); P.pts([xp[0]], [xp[1]], { c: "orange", r: 6, stroke: "ink" });
      const kap = (2 + e) * (2 + e) / e, rb = Math.abs(db * e) / (2 + e), rx = Math.max(Math.abs(xp[0] - x[0]), Math.abs(xp[1] - x[1])) / Math.max(Math.abs(x[0]), Math.abs(x[1]));
      k.set("k", kap.toExponential(2)); k.set("x", `(${x[0].toFixed(3)}, ${x[1].toFixed(3)})`); k.set("xp", `(${xp[0].toFixed(3)}, ${xp[1].toFixed(3)})`); k.set("amp", rb ? (rx / rb).toExponential(2) + " ≤ κ" : "—");
    }
    root._redraw = draw; draw();
    UI.legend(root, [["blue", "x₁ + x₂ = 2"], ["red", "x₁ + (1+ε)x₂ = 2+ε (dashed: + δb₂)"], ["orange", "perturbed solution"]]);
    UI.note(root, "As ε → 0 the lines become parallel: the same small shift of the right-hand side moves the crossing further and further. The amplification is always below κ∞ ≈ 4/ε, and close to it for this perturbation.");
  };

  /* 3. Hilbert matrices */
  W.hilbert = (root) => {
    UI.head(root, "HILBERT MATRICES · ERROR ≈ κ × ε_mach");
    const k = UI.kpis(root, [["k", "κ₂(Hₙ)"], ["e64", "error float64"], ["e32", "error float32 (emulated)"], ["dl", "digits left (double)"]]);
    const P = UI.canvas(root, 300); const c = UI.ctrls(root); let n = 8;
    UI.slider(c, { label: "size n", min: 2, max: 14, step: 1, value: n, on: (v) => { n = v; draw(); } });
    const KAP = { 2: 19.3, 3: 524, 4: 1.55e4, 5: 4.77e5, 6: 1.5e7, 7: 4.75e8, 8: 1.53e10, 9: 4.93e11, 10: 1.6e13, 11: 5.22e14, 12: 1.68e16, 13: 5.6e17, 14: 1.8e19 };
    const res = {};
    for (let m = 2; m <= 14; m++) {
      const H = Array.from({ length: m }, (_, i) => Array.from({ length: m }, (_, j) => 1 / (i + j + 1))), b = H.map((r) => r.reduce((s, v) => s + v, 0));
      const err = (x) => nrm(x.map((v) => v - 1)) / Math.sqrt(m);
      res[m] = { e64: err(luSolve(H, b)), e32: err(luSolve(H, b, true)) };
    }
    function draw() {
      const ns = M.linspace(2, 14, 13);
      P.begin({ xlim: [2, 14], ylim: [1e-17, 1e3], ylog: true, xlabel: "n", ylabel: "relative error" });
      P.line(ns, ns.map((m) => KAP[m] * 2.2e-16), { c: "blue", w: 1.5, dash: [5, 4] }); P.line(ns, ns.map((m) => KAP[m] * 1.19e-7), { c: "red", w: 1.5, dash: [5, 4] });
      P.line(ns, ns.map((m) => Math.max(res[m].e64, 1e-17)), { c: "blue", w: 2.5 }); P.pts(ns, ns.map((m) => Math.max(res[m].e64, 1e-17)), { c: "blue", r: 3.5 });
      P.line(ns, ns.map((m) => Math.max(res[m].e32, 1e-17)), { c: "red", w: 2.5 }); P.pts(ns, ns.map((m) => Math.max(res[m].e32, 1e-17)), { c: "red", r: 3.5 });
      P.vline(n, { c: "orange", dash: [] }); P.hline(1, { c: "ink", label: "100% error" });
      k.set("k", KAP[n].toExponential(2)); k.set("e64", res[n].e64.toExponential(1)); k.set("e32", res[n].e32.toExponential(1)); k.set("dl", Math.max(0, 16 - Math.log10(KAP[n])).toFixed(1));
    }
    root._redraw = draw; draw();
    UI.legend(root, [["blue", "float64: error (solid), κ·ε₆₄ (dashed)"], ["red", "float32: error (solid), κ·ε₃₂ (dashed)"]]);
    UI.note(root, "H x = H·1 solved by Gaussian elimination with partial pivoting; float32 emulated by rounding every operation with Math.fround. The observed error follows κ ε<sub>mach</sub> (usually a bit below the bound). Single precision is useless from n ≈ 7, double from n ≈ 12.");
  };

  /* 4. mixed-precision iterative refinement */
  W.refine = (root) => {
    UI.head(root, "ITERATIVE REFINEMENT · LU IN SINGLE PRECISION, RESIDUAL IN DOUBLE");
    const k = UI.kpis(root, [["e0", "float32 LU alone"], ["ef", "after 8 corrections"], ["rate", "κ ε₃₂ (contraction)"]]);
    const P = UI.canvas(root, 290); const c = UI.ctrls(root); let lk = 4, res64 = true, n = 40;
    UI.slider(c, { label: "condition number κ", min: 0, max: 9, step: 0.25, value: lk, fmt: (v) => "10^" + v.toFixed(2), on: (v) => { lk = v; draw(); } });
    UI.select(c, { label: "residual r = b − Ax computed in", options: [["64", "double precision"], ["32", "single precision"]], on: (v) => { res64 = v === "64"; draw(); } });
    const rng = M.rng(3), Q1 = randOrth(n, rng), Q2 = randOrth(n, rng), xt = Array.from({ length: n }, () => rng.normal());
    function draw() {
      const s = M.linspace(0, -lk, n).map((v) => Math.pow(10, v));
      const A = Q1.map((row) => Q2.map((r2) => row.reduce((t, q, kk) => t + q * s[kk] * r2[kk], 0)));
      const b = matvec(A, xt), F = luFactor(A, true);
      let x = luApply(F, b); const errs = [nrm(x.map((v, i) => v - xt[i])) / nrm(xt)];
      for (let it = 0; it < 8; it++) {
        let r = res64 ? b.map((bi, i) => bi - A[i].reduce((t, a, j) => t + a * x[j], 0)) : b.map((bi, i) => f32(f32(bi) - A[i].reduce((t, a, j) => f32(t + f32(f32(a) * f32(x[j]))), 0)));
        const d = luApply(F, r); x = x.map((v, i) => v + d[i]); errs.push(nrm(x.map((v, i) => v - xt[i])) / nrm(xt));
      }
      const its = M.linspace(0, 8, 9);
      P.begin({ xlim: [0, 8], ylim: [1e-17, 1e4], ylog: true, xlabel: "refinement step (0 = plain float32 LU)", ylabel: "relative error" });
      P.hline(Math.pow(10, lk) * 2.2e-16, { c: "blue", label: "κ ε₆₄" }); P.hline(1.19e-7, { c: "red", label: "ε₃₂" });
      P.line(its, errs.map((e) => Math.max(e, 1e-17)), { c: "purple", w: 2.5 }); P.pts(its, errs.map((e) => Math.max(e, 1e-17)), { c: "purple", r: 4 });
      k.set("e0", errs[0].toExponential(1)); k.set("ef", errs[8].toExponential(1)); k.set("rate", (Math.pow(10, lk) * 1.19e-7).toExponential(1));
    }
    root._redraw = draw; draw();
    UI.note(root, "A = Q₁ diag(1 … κ⁻¹) Q₂ᵀ, n = 40. With the residual in double precision the error drops by ≈ κε₃₂ per step until it reaches the double-precision limit κε₆₄. Residual in single precision: the corrections stall at the float32 level. κ above ≈ 10⁷ (κε₃₂ > 1): refinement diverges.");
  };

  /* 5. eigenvectors and the ellipse */
  W.evd = (root) => {
    UI.head(root, "EIGENVECTORS: DIRECTIONS THAT ARE ONLY STRETCHED");
    const k = UI.kpis(root, [["l", "eigenvalues λ"], ["ang", "angle between eigenvectors"], ["s", "singular values σ"], ["kk", "κ₂ = σ₁/σ₂"]]);
    const P = UI.canvas(root, 330); const c = UI.ctrls(root);
    let a = 2, b = 1, cc = 1, d = 3, sym = true; const sl = {};
    UI.select(c, { label: "matrix", options: [["1", "symmetric (a₂₁ = a₁₂)"], ["0", "general"]], on: (v) => { sym = v === "1"; if (sym) { cc = b; sl.c.set(cc); } draw(); } });
    [["a", "a₁₁", () => a, (v) => (a = v)], ["b", "a₁₂", () => b, (v) => { b = v; if (sym) { cc = v; sl.c.set(v); } }], ["c", "a₂₁", () => cc, (v) => { cc = v; if (sym) { b = v; sl.b.set(v); } }], ["d", "a₂₂", () => d, (v) => (d = v)]].forEach(([id, lab, g, s]) =>
      (sl[id] = UI.slider(c, { label: lab, min: -3, max: 3, step: 0.05, value: g(), fmt: (v) => v.toFixed(2), on: (v) => { s(v); draw(); } })));
    function draw() {
      const tr = a + d, det = a * d - b * cc, disc = tr * tr / 4 - det, th = M.linspace(0, 2 * Math.PI, 300), S = svd([[a, b], [cc, d]]), lim = Math.max(1.6, S.s[0] * 1.2);
      P.begin({ xlim: [-lim * 1.6, lim * 1.6], ylim: [-lim, lim], xlabel: "", ylabel: "" });
      P.line(th.map(Math.cos), th.map(Math.sin), { c: "faint", w: 1.2, dash: [4, 3] });
      P.line(th.map((t) => a * Math.cos(t) + b * Math.sin(t)), th.map((t) => cc * Math.cos(t) + d * Math.sin(t)), { c: "blue", w: 2.5 });
      if (disc >= 0) {
        const ls = [tr / 2 + Math.sqrt(disc), tr / 2 - Math.sqrt(disc)], vs = ls.map((l) => { let v = Math.abs(b) > 1e-12 ? [b, l - a] : Math.abs(cc) > 1e-12 ? [l - d, cc] : l === a ? [1, 0] : [0, 1]; const nn = Math.hypot(...v); return [v[0] / nn, v[1] / nn]; });
        vs.forEach((v, i) => { P.seg(-v[0] * 3 * lim, -v[1] * 3 * lim, v[0] * 3 * lim, v[1] * 3 * lim, { c: i ? "green" : "red", w: 1, dash: [3, 3] }); P.seg(0, 0, ls[i] * v[0], ls[i] * v[1], { c: i ? "green" : "red", w: 3 }); });
        const ang = Math.acos(Math.min(1, Math.abs(vs[0][0] * vs[1][0] + vs[0][1] * vs[1][1]))) * 180 / Math.PI;
        k.set("l", ls.map((l) => l.toFixed(3)).join(", ")); k.set("ang", ang.toFixed(1) + "°");
      } else { k.set("l", `${(tr / 2).toFixed(3)} ± ${Math.sqrt(-disc).toFixed(3)} i (complex)`); k.set("ang", "no real eigenvectors"); }
      [0, 1].forEach((i) => P.seg(0, 0, S.s[i] * S.U[0][i], S.s[i] * S.U[1][i], { c: "purple", w: 1.6, dash: [6, 3] }));
      k.set("s", S.s.map((v) => v.toFixed(3)).join(", ")); k.set("kk", S.s[1] > 1e-12 ? (S.s[0] / S.s[1]).toFixed(3) : "∞");
    }
    root._redraw = draw; draw();
    UI.legend(root, [["blue", "image of the unit circle"], ["red", "λ₁ v₁"], ["green", "λ₂ v₂"], ["purple", "ellipse axes σᵢ uᵢ"]]);
    UI.note(root, "Symmetric: eigenvectors at 90°, and they coincide with the axes of the ellipse (|λ| = σ). General: the eigenvectors lean towards each other (try a₁₂ = 3, a₂₁ = 0), no longer match the ellipse axes, and for a rotation-like matrix (a₁₂ = −a₂₁) they become complex.");
  };

  /* 6. SVD in three steps */
  W.svdgeom = (root) => {
    UI.head(root, "A = U Σ Vᵀ · ROTATE, STRETCH, ROTATE");
    const k = UI.kpis(root, [["s", "σ₁, σ₂"], ["r", "numerical rank"], ["k", "κ₂"], ["v", "v₁ (input direction)"]]);
    const P = UI.canvas(root, 320); const c = UI.ctrls(root);
    let a = 1.5, b = 1, cc = -0.5, d = 0.8, stage = 3;
    UI.select(c, { label: "show", options: [["3", "A x = U Σ Vᵀ x (full)"], ["0", "x (input)"], ["1", "Vᵀ x (rotated to the axes)"], ["2", "Σ Vᵀ x (stretched)"]], on: (v) => { stage = +v; draw(); } });
    [["a₁₁", () => a, (v) => (a = v)], ["a₁₂", () => b, (v) => (b = v)], ["a₂₁", () => cc, (v) => (cc = v)], ["a₂₂", () => d, (v) => (d = v)]].forEach(([lab, g, s]) =>
      UI.slider(c, { label: lab, min: -2, max: 2, step: 0.05, value: g(), fmt: (v) => v.toFixed(2), on: (v) => { s(v); draw(); } }));
    const th = M.linspace(0, 2 * Math.PI, 200), shape = th.map((t) => [Math.cos(t), Math.sin(t)]).concat([[NaN, NaN], [0, 0], [0.6, 0], [NaN, NaN], [0, 0], [0, 0.35]]);
    function draw() {
      const S = svd([[a, b], [cc, d]]), U = S.U, V = S.V, s = S.s;
      const Vt = (p) => [V[0][0] * p[0] + V[1][0] * p[1], V[0][1] * p[0] + V[1][1] * p[1]], Sg = (p) => [s[0] * p[0], s[1] * p[1]], Um = (p) => [U[0][0] * p[0] + U[0][1] * p[1], U[1][0] * p[0] + U[1][1] * p[1]];
      const T = (p) => (stage === 0 ? p : stage === 1 ? Vt(p) : stage === 2 ? Sg(Vt(p)) : Um(Sg(Vt(p))));
      const lim = Math.max(1.5, s[0] * 1.2);
      P.begin({ xlim: [-lim * 1.6, lim * 1.6], ylim: [-lim, lim] });
      P.line(th.map(Math.cos), th.map(Math.sin), { c: "faint", w: 1, dash: [4, 3] });
      const pts = shape.map((p) => (isNaN(p[0]) ? p : T(p)));
      P.line(pts.map((p) => p[0]), pts.map((p) => p[1]), { c: "blue", w: 2.5 });
      [0, 1].forEach((i) => { const vi = [V[0][i], V[1][i]], t = T(vi); P.seg(0, 0, t[0], t[1], { c: i ? "green" : "red", w: 3 }); });
      const tol = 2 * 2.2e-16 * s[0];
      k.set("s", s.map((v) => v.toFixed(4)).join(", ")); k.set("r", s.filter((v) => v > tol).length); k.set("k", s[1] > tol ? (s[0] / s[1]).toFixed(3) : "∞ (singular)"); k.set("v", `(${V[0][0].toFixed(3)}, ${V[1][0].toFixed(3)})`);
    }
    root._redraw = draw; draw();
    UI.legend(root, [["red", "image of v₁"], ["green", "image of v₂"], ["blue", "unit circle with an x-axis (long) and y-axis (short) tick"]]);
    UI.note(root, "Step through the stages: Vᵀ moves v₁, v₂ onto the axes (a rotation, possibly with a reflection), Σ stretches them by σ₁, σ₂, U rotates them to u₁, u₂. Make the rows proportional (e.g. a₂₁ = a₁₁/2, a₂₂ = a₁₂/2): σ₂ = 0, the circle collapses to a segment and v₂ spans the null space.");
  };

  /* 7. least squares: normal equations vs QR vs SVD */
  W.lsqfit = (root) => {
    UI.head(root, "POLYNOMIAL FIT y = Σ cₖ tᵏ · NORMAL EQUATIONS vs QR vs SVD");
    const k = UI.kpis(root, [["k", "κ(A) / κ(AᵀA)"], ["ne", "coef. error: normal eq."], ["qr", "QR"], ["sv", "SVD"], ["kept", "σ kept (SVD)"]]);
    const P = UI.canvas(root, 300); const c = UI.ctrls(root); let deg = 6, meth = "svd", lr = -14, noise = 0.5;
    UI.slider(c, { label: "polynomial degree", min: 1, max: 12, step: 1, value: deg, on: (v) => { deg = v; draw(); } });
    UI.select(c, { label: "method for the plotted fit", options: [["svd", "SVD (pseudoinverse)"], ["qr", "QR (Householder)"], ["ne", "normal equations (Cholesky)"]], on: (v) => { meth = v; draw(); } });
    UI.slider(c, { label: "SVD cutoff rcond", min: -16, max: -1, step: 0.25, value: lr, fmt: (v) => "10^" + v.toFixed(2), on: (v) => { lr = v; draw(); } });
    UI.slider(c, { label: "noise σ", min: 0, max: 1.5, step: 0.05, value: noise, on: (v) => { noise = v; draw(); } });
    const rng = M.rng(42), tt = M.linspace(0, 10, 100), z = tt.map(() => rng.normal()), truth = (t) => 2.5 + 1.2 * t - 0.05 * t * t;
    const solveNE = (A, y) => { const n = A[0].length, G = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => A.reduce((s, r) => s + r[i] * r[j], 0))), g = Array.from({ length: n }, (_, i) => A.reduce((s, r, m) => s + r[i] * y[m], 0));
      const L = G.map((r) => r.map(() => 0)); for (let j = 0; j < n; j++) { let s = G[j][j]; for (let q = 0; q < j; q++) s -= L[j][q] ** 2; if (!(s > 0)) return null; L[j][j] = Math.sqrt(s); for (let i = j + 1; i < n; i++) { let t = G[i][j]; for (let q = 0; q < j; q++) t -= L[i][q] * L[j][q]; L[i][j] = t / L[j][j]; } }
      const w = new Array(n); for (let i = 0; i < n; i++) { let s = g[i]; for (let q = 0; q < i; q++) s -= L[i][q] * w[q]; w[i] = s / L[i][i]; }
      const x = new Array(n); for (let i = n - 1; i >= 0; i--) { let s = w[i]; for (let q = i + 1; q < n; q++) s -= L[q][i] * x[q]; x[i] = s / L[i][i]; } return x; };
    const solveQR = (A0, y0) => { const m = A0.length, n = A0[0].length, A = A0.map((r) => r.slice()), y = y0.slice();
      for (let j = 0; j < n; j++) { let nr = 0; for (let i = j; i < m; i++) nr += A[i][j] ** 2; nr = Math.sqrt(nr); const al = A[j][j] > 0 ? -nr : nr, v = new Array(m).fill(0); for (let i = j; i < m; i++) v[i] = A[i][j]; v[j] -= al; const vv = v.reduce((s, q) => s + q * q, 0); if (vv === 0) continue;
        for (let q = j; q < n; q++) { let s = 0; for (let i = j; i < m; i++) s += v[i] * A[i][q]; s = 2 * s / vv; for (let i = j; i < m; i++) A[i][q] -= s * v[i]; }
        let s = 0; for (let i = j; i < m; i++) s += v[i] * y[i]; s = 2 * s / vv; for (let i = j; i < m; i++) y[i] -= s * v[i]; }
      const x = new Array(n); for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let q = i + 1; q < n; q++) s -= A[i][q] * x[q]; x[i] = s / A[i][i]; } return x; };
    const solveSVD = (A, y, rc) => { const S = svd(A), n = S.s.length, x = new Array(n).fill(0); let kept = 0;
      for (let i = 0; i < n; i++) { if (S.s[i] <= rc * S.s[0]) continue; kept++; const ub = S.U.reduce((t, r, m) => t + r[i] * y[m], 0) / S.s[i]; for (let q = 0; q < n; q++) x[q] += S.V[q][i] * ub; } return { x, kept, s: S.s }; };
    function draw() {
      const n = deg + 1, A = tt.map((t) => Array.from({ length: n }, (_, j) => Math.pow(t, j))), ct = Array.from({ length: n }, (_, j) => [2.5, 1.2, -0.05][j] || 0);
      const yExact = tt.map(truth), yN = yExact.map((v, i) => v + noise * z[i]), rc = Math.pow(10, lr);
      const ce = (x) => (x ? nrm(x.map((v, i) => v - ct[i])) / nrm(ct) : NaN);
      const ne0 = solveNE(A, yExact), qr0 = solveQR(A, yExact), sv0 = solveSVD(A, yExact, rc);
      const fit = meth === "ne" ? solveNE(A, yN) : meth === "qr" ? solveQR(A, yN) : solveSVD(A, yN, rc).x;
      P.begin({ xlim: [0, 10], ylim: [-2, 12], xlabel: "t", ylabel: "y" });
      P.pts(tt, yN, { c: "slate", r: 2.5, a: 0.7 }); P.line(tt, yExact, { c: "ink", w: 1.5, dash: [5, 4] });
      if (fit) { const ff = M.linspace(0, 10, 300); P.line(ff, ff.map((t) => fit.reduce((s, cc2, j) => s + cc2 * Math.pow(t, j), 0)), { c: "orange", w: 2.5 }); }
      const kap = sv0.s[0] / sv0.s[sv0.s.length - 1];
      k.set("k", `${kap.toExponential(1)} / ${(kap * kap).toExponential(1)}`); k.set("ne", ne0 ? ce(ne0).toExponential(1) : "Cholesky failed"); k.set("qr", ce(qr0).toExponential(1)); k.set("sv", ce(sv0.x).toExponential(1)); k.set("kept", sv0.kept + " of " + n);
    }
    root._redraw = draw; draw();
    UI.note(root, "The coefficient errors in the KPIs come from fitting the noiseless quadratic: they measure the algorithm, not the noise. The monomial basis tᵏ on [0, 10] is terribly conditioned: κ(A) ≈ 10¹⁰ at degree 8. The normal equations square it and fail first (Cholesky breaks down at degree ≳ 10); QR stays accurate; the SVD truncates directions below rcond·σ₁, trading exactness for stability.");
  };
})();
