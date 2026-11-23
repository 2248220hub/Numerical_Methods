"""
sparse_iterative.py  --  Lecture 6: sparse storage and iterative solvers.

  1. CSR, CSC, COO of the 5x5 example, by hand and with scipy.sparse (0-based indices)
  2. Memory and the cost of y = A x for the 2D Poisson matrix
  3. Fill-in of a direct (Cholesky) factorisation: natural, reverse Cuthill-McKee and random orderings
  4. Steepest descent vs conjugate gradient: a 2x2 problem, then the 2D Poisson problem
  5. CG iterations grow like sqrt(kappa)
  6. Preconditioning: none vs Jacobi vs incomplete Cholesky IC(0) on a variable-coefficient diffusion problem
  7. Matrix-free CG: a toy CMB-style map-maker with per-scan offsets (and its kernel: the monopole)
  8. Beyond SPD: GMRES and BiCGSTAB (non-symmetric), LSQR (rectangular)

Run:  python sparse_iterative.py          Figure: sparse_iterative.png
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import reverse_cuthill_mckee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(4)

# ---------------------------------------------------------------- 1. formats
A = np.array([[0, 0, 3, 0, 0], [22, 0, 0, 0, 17], [0, 0, 0, 0, 0], [0, 8, 0, 0, 0], [0, 0, 0, 5, 0]], float)
val, col, ptr = [], [], [0]
for i in range(5):                                       # CSR by hand: scan rows
    for j in range(5):
        if A[i, j] != 0: val.append(float(A[i, j])); col.append(j)
    ptr.append(len(val))
print("1. the 5x5 example: nnz =", np.count_nonzero(A), "of 25")
print(f"   CSR by hand (0-based): val {val}, col_ind {col}, row_ptr {ptr}")
S = sp.csr_matrix(A)
print(f"   scipy CSR:              data {S.data.tolist()}, indices {S.indices.tolist()}, indptr {S.indptr.tolist()}")
C = sp.csc_matrix(A)
print(f"   scipy CSC:              data {C.data.tolist()}, indices {C.indices.tolist()}, indptr {C.indptr.tolist()}")
O = sp.coo_matrix(A)
print(f"   scipy COO:              row {O.row.tolist()}, col {O.col.tolist()}, data {O.data.tolist()}")
x = np.arange(1.0, 6.0)
y = np.zeros(5)
for i in range(5):                                       # y = A x with CSR: one pass over the non-zeros
    for k in range(ptr[i], ptr[i + 1]): y[i] += val[k] * x[col[k]]
print(f"   CSR mat-vec by hand: {y}, dense: {A @ x}")


# ---------------------------------------------------------------- 2. memory and SpMV
def poisson2d(k):
    """5-point Laplacian on a k x k grid, Dirichlet boundaries: n = k^2 unknowns, about 5 non-zeros per row."""
    T = sp.diags([-np.ones(k - 1), 2 * np.ones(k), -np.ones(k - 1)], [-1, 0, 1])
    I = sp.identity(k)
    return (sp.kron(I, T) + sp.kron(T, I)).tocsr()


print("\n2. 2D Poisson matrix")
for k in (64, 256, 1024):
    P = poisson2d(k); n = P.shape[0]
    xv = rng.standard_normal(n)
    t0 = time.perf_counter()
    for _ in range(10): P @ xv
    t = (time.perf_counter() - t0) / 10
    mem_csr = P.data.nbytes + P.indices.nbytes + P.indptr.nbytes
    print(f"   k = {k:5d}: n = {n:>9,}, nnz = {P.nnz:>9,}; dense {8 * n * n / 1e9:12,.1f} GB, CSR {mem_csr / 1e6:8.2f} MB; SpMV {t * 1e3:7.2f} ms")

# ---------------------------------------------------------------- 3. fill-in
k = 30
P = poisson2d(k).toarray(); n = P.shape[0]
print(f"\n3. Cholesky fill-in, 2D Poisson k = {k} (n = {n}, nnz(lower A) = {np.count_nonzero(np.tril(P))})")
perms = {"natural": np.arange(n), "reverse Cuthill-McKee": reverse_cuthill_mckee(sp.csr_matrix(P), symmetric_mode=True),
         "random": rng.permutation(n)}
for name, p in perms.items():
    L = np.linalg.cholesky(P[np.ix_(p, p)])
    print(f"   {name:22s}: nnz(L) = {np.count_nonzero(np.abs(L) > 1e-14):7,d}")


# ---------------------------------------------------------------- 4. SD vs CG
def steepest_descent(A, b, x0, tol=1e-6, maxit=100_000, path=False):
    x = x0.copy(); r = b - A @ x; nr0 = np.linalg.norm(r); pts = [x.copy()]
    for k in range(maxit):
        Ar = A @ r; alpha = (r @ r) / (r @ Ar)
        x += alpha * r; r -= alpha * Ar; pts.append(x.copy())
        if np.linalg.norm(r) / nr0 < tol: return x, k + 1, pts
    return x, maxit, pts


def cg(A, b, x0, tol=1e-6, maxit=100_000, M_solve=None):
    x = x0.copy(); r = b - A @ x; z = r if M_solve is None else M_solve(r); p = z.copy()
    rz = r @ z; nr0 = np.linalg.norm(r); pts = [x.copy()]; hist = [1.0]
    for k in range(maxit):
        Ap = A @ p; alpha = rz / (p @ Ap)
        x += alpha * p; r -= alpha * Ap; pts.append(x.copy())
        hist.append(np.linalg.norm(r) / nr0)
        if hist[-1] < tol: return x, k + 1, pts, hist
        z = r if M_solve is None else M_solve(r)
        rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
    return x, maxit, pts, hist


A2 = np.array([[1.0, 0.0], [0.0, 10.0]]); R = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]]); A2 = R @ A2 @ R.T
b2 = np.array([1.0, 1.0]); x0 = np.array([-3.0, 3.0])
_, ksd, path_sd = steepest_descent(A2, b2, x0, tol=1e-10)
_, kcg, path_cg, _ = cg(A2, b2, x0, tol=1e-10)
print(f"\n4. 2x2 SPD, kappa = 10: steepest descent {ksd} steps, CG {kcg} steps (at most n = 2 in exact arithmetic)")
k = 64; P = poisson2d(k); n = P.shape[0]; xt = rng.standard_normal(n); b = P @ xt
_, ksd, _ = steepest_descent(P, b, np.zeros(n))
_, kcg, _, hcg = cg(P, b, np.zeros(n))
lam = spla.eigsh(P, k=1, which="LA", return_eigenvectors=False)[0]; lmin = 8 * np.sin(np.pi / (2 * (k + 1))) ** 2
print(f"   2D Poisson k = {k} (n = {n}), kappa = {lam / lmin:.0f}: steepest descent {ksd} iterations, CG {kcg} iterations (to ||r||/||r0|| < 1e-6)")

# ---------------------------------------------------------------- 5. sqrt(kappa)
print("\n5. CG iterations vs kappa (2D Poisson, kappa grows like k^2; right-hand side f = 1, a smooth load)")
for k in (16, 32, 64, 128, 256):
    P = poisson2d(k); n = P.shape[0]; b = np.ones(n)
    _, kcg, _, _ = cg(P, b, np.zeros(n))
    kap = (8 * np.cos(np.pi / (2 * (k + 1))) ** 2) / (8 * np.sin(np.pi / (2 * (k + 1))) ** 2)
    bound = np.log(2 / 1e-6) / 2 * np.sqrt(kap)
    print(f"   k = {k:4d}: kappa = {kap:9.0f}, sqrt(kappa) = {np.sqrt(kap):6.1f}, CG iterations {kcg:5d} (bound 0.5 ln(2/tol) sqrt(kappa) = {bound:6.0f})")


# ---------------------------------------------------------------- 6. preconditioning
def diffusion2d(k, c):
    """-div(c grad u) on a k x k grid with cell coefficients c (k+1 x k+1 faces approximated by averages)."""
    idx = np.arange(k * k).reshape(k, k); rows, cols, vals = [], [], []
    for i in range(k):
        for j in range(k):
            d = 0.0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                cf = 0.5 * (c[i, j] + c[min(max(ii, 0), k - 1), min(max(jj, 0), k - 1)])
                d += cf
                if 0 <= ii < k and 0 <= jj < k:
                    rows.append(idx[i, j]); cols.append(idx[ii, jj]); vals.append(-cf)
            rows.append(idx[i, j]); cols.append(idx[i, j]); vals.append(d)
    return sp.csr_matrix((vals, (rows, cols)), shape=(k * k, k * k))


def ic0(A):
    """Incomplete Cholesky with zero fill: L has the sparsity pattern of tril(A)."""
    L = sp.tril(A, format="csr").astype(float).tolil(); n = A.shape[0]
    Lr = [dict(zip(L.rows[i], L.data[i])) for i in range(n)]
    for i in range(n):
        for j in sorted(Lr[i]):
            if j < i:
                s = sum(Lr[i][kk] * Lr[j][kk] for kk in Lr[i] if kk < j and kk in Lr[j])
                Lr[i][j] = (Lr[i][j] - s) / Lr[j][j]
        Lr[i][i] = np.sqrt(Lr[i][i] - sum(v * v for kk, v in Lr[i].items() if kk < i))
    rows = [i for i in range(n) for _ in Lr[i]]; cols = [j for i in range(n) for j in Lr[i]]; vals = [v for i in range(n) for v in Lr[i].values()]
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


k = 48
c = np.exp(rng.normal(0, 2.0, (k, k)))                      # coefficients spanning several decades
Ad = diffusion2d(k, c); n = Ad.shape[0]; b = Ad @ rng.standard_normal(n)
kappa_d = np.linalg.cond(Ad.toarray())
Dinv = 1 / Ad.diagonal()
L = ic0(Ad); LT = L.T.tocsr()
jac = lambda r: Dinv * r
icc = lambda r: spla.spsolve_triangular(LT, spla.spsolve_triangular(L, r, lower=True), lower=False)
print(f"\n6. variable-coefficient diffusion, k = {k} (n = {n}), coefficients from {c.min():.1e} to {c.max():.1e}, kappa = {kappa_d:.1e}")
curves = {}
for name, M in (("no preconditioner", None), ("Jacobi", jac), ("IC(0)", icc)):
    t0 = time.perf_counter(); xs, it, _, h = cg(Ad, b, np.zeros(n), M_solve=M); t = time.perf_counter() - t0
    curves[name] = h
    print(f"   {name:18s}: {it:5d} iterations, {t * 1e3:7.0f} ms")
Dh = sp.diags(np.sqrt(Dinv))
print(f"   kappa(D^-1/2 A D^-1/2) = {np.linalg.cond((Dh @ Ad @ Dh).toarray()):.1e} (Jacobi-scaled)")

# ---------------------------------------------------------------- 7. matrix-free map-making
npix, nscan, slen = 64, 1024, 16                            # 1024 short scans of 16 samples on a ring of 64 pixels
sky = np.sin(2 * np.pi * np.arange(npix) / npix * 3) + 0.5 * rng.standard_normal(npix)
pix = np.concatenate([(rng.integers(npix) + np.arange(slen) * rng.choice([1, -1])) % npix for _ in range(nscan)])
offs = 2.0 * rng.standard_normal(nscan)                     # one slow drift (offset) per scan
d = sky[pix] + np.repeat(offs, slen) + 0.1 * rng.standard_normal(pix.size)
nu = npix + nscan


def apply_A(z):                                             # time-ordered data from (map, offsets): never stored as a matrix
    return z[:npix][pix] + np.repeat(z[npix:], slen)


def apply_AT(t):
    return np.concatenate([np.bincount(pix, t, npix), t.reshape(nscan, slen).sum(1)])


Nop = spla.LinearOperator((nu, nu), matvec=lambda z: apply_AT(apply_A(z)), dtype=float)
rhs = apply_AT(d)
z, it, _, h = cg(Nop, rhs, np.zeros(nu), tol=1e-10)
m_hat = z[:npix]
print(f"\n7. matrix-free map-making: {pix.size:,} samples, {npix} pixels + {nscan} offsets, solved in {it} CG iterations")
print(f"   map error {np.std(m_hat - sky):.4f} after removing the mean; mean of map - sky = {np.mean(m_hat - sky):+.3f}")
print(f"   naive binned map (ignoring offsets): error {np.std(np.bincount(pix, d, npix) / np.bincount(pix, None, npix) - sky):.3f}")
v = np.concatenate([np.ones(npix), -np.ones(nscan)])
print(f"   kernel: (A^T A) [1,...,1 | -1,...,-1] = {np.abs(Nop @ v).max():.1e} -> a constant map is not measurable (monopole)")

# ---------------------------------------------------------------- 8. beyond SPD
k = 32; n = k * k; h_ = 1 / (k + 1)
Lp = poisson2d(k) / h_ ** 2
Dx = sp.kron(sp.identity(k), sp.diags([-np.ones(k - 1), np.ones(k - 1)], [-1, 1]) / (2 * h_))
Aad = (Lp + 40 * Dx).tocsr()                                # advection-diffusion: not symmetric
bb = Aad @ np.ones(n)
print(f"\n8. advection-diffusion (non-symmetric), n = {n}: ||A - A^T|| / ||A|| = {spla.norm(Aad - Aad.T) / spla.norm(Aad):.2f}")
for name, solver in (("GMRES(30)", lambda: spla.gmres(Aad, bb, rtol=1e-8, restart=30, maxiter=2000)),
                     ("BiCGSTAB", lambda: spla.bicgstab(Aad, bb, rtol=1e-8, maxiter=4000))):
    xs, info = solver()
    print(f"   {name:10s}: info {info}, error {np.abs(xs - 1).max():.1e}")
Ar = sp.random(2000, 300, density=0.02, random_state=5, format="csr") + sp.eye(2000, 300)
xr = rng.standard_normal(300); br = Ar @ xr
res = spla.lsqr(Ar, br, atol=1e-12, btol=1e-12)
print(f"   LSQR on a sparse 2000 x 300 least-squares problem: {res[2]} iterations, error {np.abs(res[0] - xr).max():.1e}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
xs_ = np.linalg.solve(A2, b2)
F = 0.5 * (A2[0, 0] * xx ** 2 + 2 * A2[0, 1] * xx * yy + A2[1, 1] * yy ** 2) - b2[0] * xx - b2[1] * yy
ax[0].contour(xx, yy, F, levels=25, cmap="Greys", linewidths=0.6)
ps = np.array(path_sd); pc = np.array(path_cg)
ax[0].plot(ps[:, 0], ps[:, 1], "o-", ms=3, color="C3", label=f"steepest descent ({len(ps) - 1} steps)")
ax[0].plot(pc[:, 0], pc[:, 1], "s-", ms=4, color="C0", label=f"conjugate gradient ({len(pc) - 1} steps)")
ax[0].plot(*xs_, "k*", ms=12); ax[0].set_aspect("equal"); ax[0].legend(fontsize=8); ax[0].set_title("f(x) = x^T A x / 2 - b^T x, kappa = 10")
for name, hh in curves.items():
    ax[1].semilogy(hh, label=name)
ax[1].axhline(1e-6, color="k", ls=":"); ax[1].set(xlabel="iteration", ylabel="||r|| / ||r0||", title="preconditioned CG"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("sparse_iterative.png", dpi=120)
print("\nsaved sparse_iterative.png")
