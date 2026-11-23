/* Lesson 6 — sparse storage, CG, preconditioning, matrix-free map-making: symbols and widgets */
window.SYMBOLS = {
  A: { n: "A — sparse SPD matrix", t: "n × n, nnz ≪ n²", s: "Accessed only through products Av by iterative methods.", py: "A = scipy.sparse.csr_matrix(...)", f: "type(csr_matrix) :: a" },
  val: { n: "val — non-zero values", t: "array of length nnz", s: "The stored entries, row by row (CSR) or column by column (CSC).", py: "A.data", f: "a%val(:)" },
  nnz: { n: "nnz — number of non-zeros", t: "integer", s: "Memory and the cost of y = Ax are O(nnz) instead of O(n²).", py: "A.nnz", f: "a%nnz" },
  rk: { n: "r<sub>k</sub> — residual", t: "ℝⁿ", s: "b − Ax<sub>k</sub> = −∇f(x<sub>k</sub>): how far from solving the system, and the steepest-descent direction.", py: "r = b - A @ x", f: "r = b - ap" },
  alpha: { n: "α<sub>k</sub> — step length", t: "> 0", s: "Exact minimiser of f along the search direction.", py: "alpha = rr / (p @ Ap)", f: "alpha = rz / dot_product(p, ap)" },
  pk: { n: "p<sub>k</sub> — search direction", t: "ℝⁿ", s: "In CG, A-conjugate to all previous directions: p<sub>i</sub>ᵀAp<sub>j</sub> = 0.", py: "p = r + beta * p", f: "p = z + beta*p" },
  beta: { n: "β<sub>k</sub> — direction update", t: "≥ 0", s: "r<sub>k+1</sub>ᵀr<sub>k+1</sub> / r<sub>k</sub>ᵀr<sub>k</sub>: how much of the old direction to keep.", py: "beta = rr_new / rr", f: "beta = rz_new / rz" },
  Mpre: { n: "M — preconditioner", t: "SPD, M ≈ A", s: "Cheap to invert; replaces κ(A) with κ(M⁻¹A). Jacobi: diag(A); IC(0): incomplete Cholesky.", py: "M = LinearOperator(A.shape, matvec=lambda r: r / d)", f: "z = dinv * r" }
};

(function () {
  const W = (window.WIDGETS = {});
  /* 2D Poisson (k×k) in CSR, 0-based */
  const poisson = (k) => { const n = k * k, val = [], col = [], ptr = [0];
    for (let j = 0; j < k; j++) for (let i = 0; i < k; i++) { const r = j * k + i;
      if (j > 0) { col.push(r - k); val.push(-1); } if (i > 0) { col.push(r - 1); val.push(-1); } col.push(r); val.push(4);
      if (i < k - 1) { col.push(r + 1); val.push(-1); } if (j < k - 1) { col.push(r + k); val.push(-1); } ptr.push(val.length); }
    return { n, val, col, ptr }; };
  const spmv = (A, x) => { const y = new Float64Array(A.n); for (let i = 0; i < A.n; i++) { let s = 0; for (let p = A.ptr[i]; p < A.ptr[i + 1]; p++) s += A.val[p] * x[A.col[p]]; y[i] = s; } return y; };
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const pcg = (mv, b, Msolve, tol, maxit) => {
    const n = b.length, x = new Float64Array(n), r = Float64Array.from(b); let z = Msolve ? Msolve(r) : r, p = Float64Array.from(z), rz = dot(r, z); const nb = Math.sqrt(dot(r, r)), hist = [1];
    for (let k = 0; k < maxit; k++) { const Ap = mv(p), al = rz / dot(p, Ap); for (let i = 0; i < n; i++) { x[i] += al * p[i]; r[i] -= al * Ap[i]; }
      const rel = Math.sqrt(dot(r, r)) / nb; hist.push(rel); if (rel < tol) break; z = Msolve ? Msolve(r) : r; const rzn = dot(r, z); for (let i = 0; i < n; i++) p[i] = z[i] + (rzn / rz) * p[i]; rz = rzn; }
    return { x, hist };
  };
  const sd = (mv, b, tol, maxit) => { const n = b.length, x = new Float64Array(n), r = Float64Array.from(b), nb = Math.sqrt(dot(r, r)), hist = [1];
    for (let k = 0; k < maxit; k++) { const Ar = mv(r), al = dot(r, r) / dot(r, Ar); for (let i = 0; i < n; i++) { x[i] += al * r[i]; r[i] -= al * Ar[i]; } const rel = Math.sqrt(dot(r, r)) / nb; hist.push(rel); if (rel < tol) break; }
    return { x, hist }; };

  /* 1. formats */
  W.sparsefmt = (root) => {
    UI.head(root, "CLICK THE MATRIX · CSR, CSC AND COO UPDATE");
    const P = UI.canvas(root, 250); const c = UI.ctrls(root); const out = UI.el("div", "note"); root.append(out);
    let base = 1; const Mx = [[0, 0, 3, 0, 0], [22, 0, 0, 0, 17], [0, 0, 0, 0, 0], [0, 8, 0, 0, 0], [0, 0, 0, 5, 0]]; let next = 1;
    UI.select(c, { label: "indices", options: [["1", "1-based (Fortran, the notes)"], ["0", "0-based (C, Python, scipy)"]], on: (v) => { base = +v; draw(); } });
    UI.button(UI.btns(root), "reset to the example", () => { [[0, 0, 3, 0, 0], [22, 0, 0, 0, 17], [0, 0, 0, 0, 0], [0, 8, 0, 0, 0], [0, 0, 0, 5, 0]].forEach((r, i) => (Mx[i] = r.slice())); draw(); });
    P.c.style.cursor = "pointer";
    P.c.addEventListener("click", (e) => { const rc = P.c.getBoundingClientRect(), [x, y] = P.inv(e.clientX - rc.left, e.clientY - rc.top), j = Math.floor(x), i = 4 - Math.floor(y); if (i >= 0 && i < 5 && j >= 0 && j < 5) { Mx[i][j] = Mx[i][j] ? 0 : next++; draw(); } });
    function draw() {
      P.begin({ xlim: [0, 9], ylim: [0, 5], axes: false });
      for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) { const v = Mx[i][j]; P.rect(j + 0.05, 4 - i + 0.05, j + 0.95, 4 - i + 0.95, { c: v ? "blue" : "grid", a: v ? 0.3 : 0.25 }); P.text(j + 0.5, 4.5 - i, v ? String(v) : "0", { align: "center", base: "middle", size: 14, c: v ? "ink" : "faint" }); }
      for (let i = 0; i < 5; i++) P.text(-0.05, 4.5 - i, "", {});
      const nnz = Mx.flat().filter(Boolean).length; P.text(5.4, 4.5, `n = 5, nnz = ${nnz} of 25`, { size: 12, base: "middle" }); P.text(5.4, 3.7, "click a cell to add/remove", { size: 11, c: "muted", base: "middle" });
      const vR = [], cR = [], pR = [base]; for (let i = 0; i < 5; i++) { for (let j = 0; j < 5; j++) if (Mx[i][j]) { vR.push(Mx[i][j]); cR.push(j + base); } pR.push(vR.length + base); }
      const vC = [], rC = [], pC = [base]; for (let j = 0; j < 5; j++) { for (let i = 0; i < 5; i++) if (Mx[i][j]) { vC.push(Mx[i][j]); rC.push(i + base); } pC.push(vC.length + base); }
      const rO = [], cO = [], vO = []; for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) if (Mx[i][j]) { rO.push(i + base); cO.push(j + base); vO.push(Mx[i][j]); }
      const L = (a) => "[" + a.join(", ") + "]";
      out.innerHTML = `<b>CSR</b> val ${L(vR)} · col_ind ${L(cR)} · row_ptr ${L(pR)}<br><b>CSC</b> val ${L(vC)} · row_ind ${L(rC)} · col_ptr ${L(pC)}<br><b>COO</b> row ${L(rO)} · col ${L(cO)} · val ${L(vO)}<br>Storage: dense 25 numbers; CSR ${2 * nnz + 6} numbers; COO ${3 * nnz}. Row i of CSR = val[row_ptr[i] … row_ptr[i+1] − 1]; an empty row has row_ptr[i] = row_ptr[i+1].`;
    }
    root._redraw = draw; draw();
  };

  /* 2. fill-in of Cholesky for different orderings */
  W.fillin = (root) => {
    UI.head(root, "SPARSITY OF A AND OF ITS CHOLESKY FACTOR L");
    const k = UI.kpis(root, [["a", "nnz(tril A)"], ["l", "nnz(L)"], ["f", "fill-in factor"]]);
    const box = UI.el("div", "split"); root.append(box); const A = UI.canvas(box, 300, { pad: [8, 8, 26, 30] }), B = UI.canvas(box, 300, { pad: [8, 8, 26, 30] }); const c = UI.ctrls(root);
    let g = 12, ord = "natural";
    UI.slider(c, { label: "grid k (n = k²)", min: 4, max: 22, step: 1, value: g, on: (v) => { g = v; draw(); } });
    UI.select(c, { label: "ordering of the unknowns", options: [["natural", "natural (row by row)"], ["rcm", "reverse Cuthill–McKee"], ["random", "random"], ["rb", "red–black (checkerboard)"]], on: (v) => { ord = v; draw(); } });
    function order(kk) {
      const n = kk * kk, idx = [...Array(n).keys()];
      if (ord === "random") { const r = M.rng(9); for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; } return idx; }
      if (ord === "rb") return idx.filter((v) => ((v % kk) + Math.floor(v / kk)) % 2 === 0).concat(idx.filter((v) => ((v % kk) + Math.floor(v / kk)) % 2 === 1));
      if (ord === "rcm") { // Cuthill–McKee from a corner, then reversed
        const deg = (v) => [v - 1, v + 1, v - kk, v + kk].filter((u) => u >= 0 && u < n && (Math.abs((u % kk) - (v % kk)) <= 1)).length;
        const nb = (v) => [v - kk, v - 1, v + 1, v + kk].filter((u) => u >= 0 && u < n && !((u === v - 1 && v % kk === 0) || (u === v + 1 && v % kk === kk - 1)));
        const seen = new Array(n).fill(false), out = [0]; seen[0] = true;
        for (let h = 0; h < out.length; h++) nb(out[h]).sort((a, b) => deg(a) - deg(b)).forEach((u) => { if (!seen[u]) { seen[u] = true; out.push(u); } });
        return out.reverse();
      }
      return idx;
    }
    function draw() {
      const n = g * g, p = order(g), inv = new Array(n); p.forEach((v, i) => (inv[v] = i));
      const D = Array.from({ length: n }, () => new Float64Array(n));
      for (let v = 0; v < n; v++) { const i = inv[v]; D[i][i] = 4; [v - 1, v + 1, v - g, v + g].forEach((u) => { if (u < 0 || u >= n) return; if ((u === v - 1 && v % g === 0) || (u === v + 1 && v % g === g - 1)) return; D[i][inv[u]] = -1; }); }
      const L = Array.from({ length: n }, () => new Float64Array(n)); let nl = 0, na = 0;
      for (let j = 0; j < n; j++) { let s = D[j][j]; for (let q = 0; q < j; q++) s -= L[j][q] * L[j][q]; L[j][j] = Math.sqrt(s);
        for (let i = j + 1; i < n; i++) { let t = D[i][j]; for (let q = 0; q < j; q++) t -= L[i][q] * L[j][q]; L[i][j] = t / L[j][j]; } }
      const ZA = new Array(n * n), ZL = new Array(n * n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const a = D[i][j] !== 0, l = j <= i && Math.abs(L[i][j]) > 1e-13; ZA[i * n + j] = a ? 1 : 0; ZL[i * n + j] = l ? (j <= i && D[i][j] !== 0 ? 1 : 0.55) : 0; if (a && j <= i) na++; if (l) nl++; }
      A.begin({ xlim: [0, n], ylim: [0, n], title: "A" }); A.image(ZA, n, n, 0, n, 0, n, { cmap: "gray", lo: 1, hi: 0 });
      B.begin({ xlim: [0, n], ylim: [0, n], title: "L (grey: fill-in)" }); B.image(ZL, n, n, 0, n, 0, n, { cmap: "gray", lo: 1, hi: 0 });
      k.set("a", na); k.set("l", nl); k.set("f", (nl / na).toFixed(2) + "×");
    }
    root._redraw = draw; draw();
    UI.note(root, "Black: entries of A and of L that were non-zero in A; grey: fill-in created by the elimination. Natural ordering fills the band of width k; reverse Cuthill–McKee narrows the profile; a random ordering fills almost everything. Real sparse solvers use minimum degree or nested dissection orderings.");
  };

  /* 3. steepest descent vs CG in 2D */
  W.descent = (root) => {
    UI.head(root, "MINIMISE f(x) = ½xᵀAx − bᵀx · CLICK TO SET THE START");
    const k = UI.kpis(root, [["k", "κ(A)"], ["sd", "steepest descent steps (to 10⁻¹⁰)"], ["cg", "CG steps"]]);
    const P = UI.canvas(root, 360); const c = UI.ctrls(root); let lk = 1, ang = 30, x0 = [-3, 2.5];
    UI.slider(c, { label: "κ", min: 0, max: 3, step: 0.05, value: lk, fmt: (v) => Math.pow(10, v).toFixed(1), on: (v) => { lk = v; draw(); } });
    UI.slider(c, { label: "orientation of the valley [deg]", min: 0, max: 180, step: 1, value: ang, on: (v) => { ang = v; draw(); } });
    P.c.style.cursor = "crosshair"; P.c.addEventListener("click", (e) => { const r = P.c.getBoundingClientRect(); x0 = P.inv(e.clientX - r.left, e.clientY - r.top); draw(); });
    function draw() {
      const kap = Math.pow(10, lk), t = ang * Math.PI / 180, cs = Math.cos(t), sn = Math.sin(t), l1 = 1, l2 = 1 / kap;   // eigenvalues 1 and 1/κ (scaled for display)
      const A = [[l1 * cs * cs + l2 * sn * sn, (l1 - l2) * cs * sn], [(l1 - l2) * cs * sn, l1 * sn * sn + l2 * cs * cs]], xs = [0.5, 0.3], b = [A[0][0] * xs[0] + A[0][1] * xs[1], A[1][0] * xs[0] + A[1][1] * xs[1]];
      const mv = (v) => [A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1]], f = (x, y) => 0.5 * (A[0][0] * x * x + 2 * A[0][1] * x * y + A[1][1] * y * y) - b[0] * x - b[1] * y;
      const run = (cg) => { let x = x0.slice(), r = [b[0] - mv(x)[0], b[1] - mv(x)[1]], p = r.slice(), rr = r[0] * r[0] + r[1] * r[1]; const path = [x.slice()], r0 = Math.sqrt(rr);
        for (let it = 0; it < 4000; it++) { const d = cg ? p : r, Ad = mv(d), al = (r[0] * r[0] + r[1] * r[1]) / (d[0] * Ad[0] + d[1] * Ad[1]); x = [x[0] + al * d[0], x[1] + al * d[1]]; r = [r[0] - al * Ad[0], r[1] - al * Ad[1]]; path.push(x.slice());
          const rn = r[0] * r[0] + r[1] * r[1]; if (Math.sqrt(rn) < 1e-10 * r0) break; if (cg) { p = [r[0] + (rn / rr) * p[0], r[1] + (rn / rr) * p[1]]; rr = rn; } }
        return path; };
      const ps = run(false), pc = run(true);
      P.begin({ xlim: [-4.5, 4.5], ylim: [-3, 3], xlabel: "x₁", ylabel: "x₂" });
      const nx = 120, ny = 80, Z = new Array(nx * ny); for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) Z[j * nx + i] = Math.log(f(-4.5 + 9 * (i + 0.5) / nx, 3 - 6 * (j + 0.5) / ny) - f(xs[0], xs[1]) + 1e-3);
      P.image(Z, nx, ny, -4.5, 4.5, -3, 3, { cmap: "magma", smooth: true });
      P.line(ps.map((q) => q[0]), ps.map((q) => q[1]), { c: "#ffffff", w: 1.5 }); P.pts(ps.slice(0, 200).map((q) => q[0]), ps.slice(0, 200).map((q) => q[1]), { c: "#ffffff", r: 2.2 });
      P.line(pc.map((q) => q[0]), pc.map((q) => q[1]), { c: "green", w: 3 }); P.pts(pc.map((q) => q[0]), pc.map((q) => q[1]), { c: "green", r: 4.5, stroke: "ink" });
      P.pts([xs[0]], [xs[1]], { c: "yellow", r: 6, stroke: "ink" });
      k.set("k", kap.toFixed(1)); k.set("sd", ps.length - 1); k.set("cg", pc.length - 1);
    }
    root._redraw = draw; draw();
    UI.legend(root, [["#ffffff", "steepest descent (white)"], ["green", "conjugate gradient"], ["yellow", "minimum = solution"]]);
    UI.note(root, "Background: log of f (dark = low). Steepest descent always moves perpendicular to the contour and turns 90° at each step: in a narrow valley it zig-zags, and its step count grows like κ. CG reaches the minimum in 2 steps whatever κ and the start.");
  };

  /* 4. convergence curves on 2D Poisson */
  W.cgconv = (root) => {
    UI.head(root, "2D POISSON · ‖r‖/‖r₀‖ PER ITERATION");
    const k = UI.kpis(root, [["n", "n = k²"], ["kap", "κ"], ["sd", "steepest descent iterations"], ["cg", "CG iterations"]]);
    const P = UI.canvas(root, 300); const c = UI.ctrls(root); let g = 32;
    UI.slider(c, { label: "grid k", min: 8, max: 96, step: 8, value: g, on: (v) => { g = v; draw(); } });
    function draw() {
      const A = poisson(g), b = new Float64Array(A.n).fill(1), mv = (v) => spmv(A, v);
      const s = sd(mv, b, 1e-6, 20000), cg = pcg(mv, b, null, 1e-6, 5000);
      const kap = (Math.cos(Math.PI / (2 * (g + 1))) / Math.sin(Math.PI / (2 * (g + 1)))) ** 2, rho = (Math.sqrt(kap) - 1) / (Math.sqrt(kap) + 1);
      const xmax = Math.max(s.hist.length, cg.hist.length);
      P.begin({ xlim: [1, xmax], ylim: [1e-7, 2], xlog: true, ylog: true, xlabel: "iteration", ylabel: "‖r‖ / ‖r₀‖" });
      const it = (h) => h.map((_, i) => i + 1);
      P.line(it(s.hist), s.hist, { c: "red", w: 2 }); P.line(it(cg.hist), cg.hist, { c: "blue", w: 2.5 });
      const kk = M.logspace(0, Math.log10(xmax), 60); P.line(kk, kk.map((q) => 2 * Math.pow(rho, q)), { c: "blue", w: 1.2, dash: [5, 4] });
      k.set("n", A.n); k.set("kap", kap.toFixed(0)); k.set("sd", s.hist.length - 1 >= 20000 ? "> 20 000" : s.hist.length - 1); k.set("cg", cg.hist.length - 1);
    }
    root._redraw = draw; draw();
    UI.legend(root, [["red", "steepest descent"], ["blue", "CG (dashed: bound 2((√κ−1)/(√κ+1))ᵏ, for the A-norm error)"]]);
    UI.note(root, "Right-hand side f = 1. Doubling k quadruples κ: steepest descent needs ≈ 4× more iterations, CG ≈ 2×. The CG residual is not monotone (CG minimises the A-norm of the error, not ‖r‖).");
  };

  /* 5. matrix-free map-making */
  W.mapmaking = (root) => {
    UI.head(root, "A TOY MAP-MAKER · MAP + PER-SCAN OFFSETS, SOLVED MATRIX-FREE");
    const k = UI.kpis(root, [["it", "CG iterations"], ["en", "naive binned map error"], ["ec", "destriped map error (mean removed)"], ["mo", "lost monopole"]]);
    const A = UI.canvas(root, 250), B = UI.canvas(root, 170); const c = UI.ctrls(root); let slen = 16, sig = 2, seed = 1;
    UI.slider(c, { label: "scan length [samples]", min: 4, max: 128, step: 4, value: slen, on: (v) => { slen = v; draw(); } });
    UI.slider(c, { label: "offset σ (drift amplitude)", min: 0, max: 5, step: 0.1, value: sig, on: (v) => { sig = v; draw(); } });
    UI.button(UI.btns(root), "new realisation", () => { seed++; draw(); });
    function draw() {
      const rng = M.rng(seed * 17), npix = 64, nsamp = 16384, nscan = Math.floor(nsamp / slen), ns = nscan * slen;
      const sky = Array.from({ length: npix }, (_, i) => Math.sin(2 * Math.PI * 3 * i / npix) + 0.5 * Math.cos(2 * Math.PI * 7 * i / npix));
      const pix = new Int32Array(ns); for (let s = 0; s < nscan; s++) { const st = Math.floor(rng() * npix), dir = rng() < 0.5 ? 1 : -1; for (let t = 0; t < slen; t++) pix[s * slen + t] = (((st + dir * t) % npix) + npix) % npix; }
      const offs = Array.from({ length: nscan }, () => sig * rng.normal()), d = new Float64Array(ns); for (let i = 0; i < ns; i++) d[i] = sky[pix[i]] + offs[Math.floor(i / slen)] + 0.1 * rng.normal();
      const nu = npix + nscan;
      const Az = (z) => { const t = new Float64Array(ns); for (let i = 0; i < ns; i++) t[i] = z[pix[i]] + z[npix + Math.floor(i / slen)]; return t; };
      const ATt = (t) => { const z = new Float64Array(nu); for (let i = 0; i < ns; i++) { z[pix[i]] += t[i]; z[npix + Math.floor(i / slen)] += t[i]; } return z; };
      const R = pcg((z) => ATt(Az(z)), ATt(d), null, 1e-10, 2000), m = Array.from(R.x.slice(0, npix));
      const hits = new Float64Array(npix), sum = new Float64Array(npix); for (let i = 0; i < ns; i++) { hits[pix[i]]++; sum[pix[i]] += d[i]; }
      const naive = Array.from(sum, (v, i) => v / Math.max(1, hits[i]));
      const dm = M.mean(m) - M.mean(sky), err = (a) => Math.sqrt(M.mean(a.map((v, i) => (v - M.mean(a) - (sky[i] - M.mean(sky))) ** 2)));
      const ii = M.linspace(0, npix - 1, npix), lo = Math.min(...sky, ...naive) - 0.5, hi = Math.max(...sky, ...naive) + 0.5;
      A.begin({ xlim: [0, npix - 1], ylim: [lo, hi], xlabel: "pixel", ylabel: "map" });
      A.line(ii, sky, { c: "ink", w: 3 }); A.line(ii, naive, { c: "red", w: 1.6 }); A.line(ii, m.map((v) => v - dm), { c: "green", w: 2, dash: [5, 3] });
      B.begin({ xlim: [0, 600], ylim: [-sig * 3 - 2, sig * 3 + 2], xlabel: "time sample (first 600)", ylabel: "TOD" });
      B.line(M.linspace(0, 599, 600), Array.from(d.slice(0, 600)), { c: "slate", w: 1 });
      k.set("it", R.hist.length - 1); k.set("en", err(naive).toFixed(3)); k.set("ec", err(m).toFixed(4)); k.set("mo", dm.toFixed(3));
    }
    root._redraw = draw; draw();
    UI.legend(root, [["ink", "true sky"], ["red", "naive binned map"], ["green", "CG solution (mean shifted to the truth)"], ["slate", "time-ordered data"]]);
    UI.note(root, "Every sample sees one pixel plus the offset of its scan. A z (map → time stream) and Aᵀ t (time stream → map and offsets) are loops over the samples: the 16 384 × (64 + 1024) matrix is never stored. Longer scans tie the offsets better and the naive map improves too; the monopole (a constant moved between map and offsets) stays undetermined.");
  };

  /* 6. preconditioned CG on a variable-coefficient diffusion problem */
  W.pcg = (root) => {
    UI.head(root, "PRECONDITIONED CG · −∇·(c∇u) = f WITH JUMPING COEFFICIENTS");
    const k = UI.kpis(root, [["n", "no preconditioner"], ["j", "Jacobi"], ["ic", "IC(0)"]]);
    const P = UI.canvas(root, 300); const c = UI.ctrls(root); let g = 32, spread = 2, seed = 4;
    UI.slider(c, { label: "grid k", min: 8, max: 64, step: 4, value: g, on: (v) => { g = v; draw(); } });
    UI.slider(c, { label: "coefficient spread (decades)", min: 0, max: 6, step: 0.5, value: spread, on: (v) => { spread = v; draw(); } });
    UI.button(UI.btns(root), "new coefficients", () => { seed++; draw(); });
    function draw() {
      const n = g * g, rng = M.rng(seed), cc = Array.from({ length: n }, () => Math.pow(10, spread * (rng() - 0.5)));
      const id = (i, j) => j * g + i, diag = new Float64Array(n), W_ = new Float64Array(n), S_ = new Float64Array(n);   // couplings to west (i-1) and south (j-1)
      for (let j = 0; j < g; j++) for (let i = 0; i < g; i++) { const p = id(i, j); let dsum = 0;
        [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]].forEach(([a, b]) => { const inside = a >= 0 && a < g && b >= 0 && b < g, cf = 0.5 * (cc[p] + (inside ? cc[id(a, b)] : cc[p])); dsum += cf; if (inside && a === i - 1) W_[p] = -cf; if (inside && b === j - 1) S_[p] = -cf; });
        diag[p] = dsum; }
      const mv = (x) => { const y = new Float64Array(n); for (let p = 0; p < n; p++) { const i = p % g, j = Math.floor(p / g); let s = diag[p] * x[p]; if (i > 0) s += W_[p] * x[p - 1]; if (i < g - 1) s += W_[p + 1] * x[p + 1]; if (j > 0) s += S_[p] * x[p - g]; if (j < g - 1) s += S_[p + g] * x[p + g]; y[p] = s; } return y; };
      // IC(0) for the 5-point stencil: L has diagonal d, west lw, south ls
      const d = new Float64Array(n), lw = new Float64Array(n), ls = new Float64Array(n);
      for (let p = 0; p < n; p++) { const i = p % g, j = Math.floor(p / g); lw[p] = i > 0 ? W_[p] / d[p - 1] : 0; ls[p] = j > 0 ? S_[p] / d[p - g] : 0; d[p] = Math.sqrt(Math.max(1e-300, diag[p] - lw[p] ** 2 - ls[p] ** 2)); }
      const icSolve = (r) => { const y = new Float64Array(n), z = new Float64Array(n);
        for (let p = 0; p < n; p++) { const i = p % g, j = Math.floor(p / g); let s = r[p]; if (i > 0) s -= lw[p] * y[p - 1]; if (j > 0) s -= ls[p] * y[p - g]; y[p] = s / d[p]; }
        for (let p = n - 1; p >= 0; p--) { const i = p % g, j = Math.floor(p / g); let s = y[p]; if (i < g - 1) s -= lw[p + 1] * z[p + 1]; if (j < g - 1) s -= ls[p + g] * z[p + g]; z[p] = s / d[p]; } return z; };
      const xt = Array.from({ length: n }, () => rng.normal()), b = mv(xt);
      const r0 = pcg(mv, b, null, 1e-6, 20000), rj = pcg(mv, b, (r) => r.map((v, p) => v / diag[p]), 1e-6, 20000), ri = pcg(mv, b, icSolve, 1e-6, 20000);
      const xmax = Math.max(r0.hist.length, rj.hist.length, ri.hist.length);
      P.begin({ xlim: [0, xmax], ylim: [1e-7, 10], ylog: true, xlabel: "iteration", ylabel: "‖r‖ / ‖r₀‖" });
      P.line(r0.hist.map((_, i) => i), r0.hist, { c: "red", w: 2 }); P.line(rj.hist.map((_, i) => i), rj.hist, { c: "orange", w: 2.2 }); P.line(ri.hist.map((_, i) => i), ri.hist, { c: "green", w: 2.5 });
      k.set("n", r0.hist.length - 1 + " it"); k.set("j", rj.hist.length - 1 + " it"); k.set("ic", ri.hist.length - 1 + " it");
    }
    root._redraw = draw; draw();
    UI.legend(root, [["red", "CG"], ["orange", "Jacobi-PCG"], ["green", "IC(0)-PCG"]]);
    UI.note(root, "Coefficients c drawn log-uniformly over the chosen number of decades. With spread 0 (constant c) Jacobi is a uniform scaling and gains nothing; as the spread grows, the diagonal varies wildly, plain CG slows down and Jacobi recovers most of the loss. IC(0) needs the fewest iterations at every spread, at the price of two triangular solves per iteration.");
  };
})();
