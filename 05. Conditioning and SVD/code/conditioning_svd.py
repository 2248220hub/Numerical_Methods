"""
conditioning_svd.py  --  Lecture 5: norms, condition number, iterative refinement, EVD and SVD.

  1. Vector and matrix norms (the 3x3 matrix of the notes); ||A||_2 = sigma_max, ||A||_F = sqrt(sum sigma^2)
  2. The 2x2 example: kappa ~ 4/eps and a 100% error from a tiny perturbation of b
  3. Hilbert matrices: kappa, digits lost, observed error in float32 and float64 against the bound
  4. Iterative refinement: LU in float32, residual in float64 -> double-precision accuracy
  5. EVD: symmetric (stable, orthogonal) vs non-symmetric (sensitive eigenvalues)
  6. SVD: numerical rank, kernel vectors, the four subspaces
  7. Least squares: normal equations vs QR vs SVD on an ill-conditioned polynomial fit;
     truncated SVD; minimum-norm solution of an underdetermined system

Run:  python conditioning_svd.py          Figure: conditioning_svd.png
"""
import warnings
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=la.LinAlgWarning)    # we WANT to solve ill-conditioned systems here
rng = np.random.default_rng(3)
np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------- 1. norms
A = np.array([[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]])
x = np.array([1.0, -2.0, 3.0])
s = la.svdvals(A)
print("1. norms")
print(f"   x = {x}: ||x||_1 = {la.norm(x, 1):.4f}, ||x||_2 = {la.norm(x):.4f}, ||x||_inf = {la.norm(x, np.inf):.4f}")
print(f"   ||A||_1 = {la.norm(A, 1):.4f} (max column sum), ||A||_inf = {la.norm(A, np.inf):.4f} (max row sum),"
      f" ||A||_2 = {la.norm(A, 2):.4f} = sigma_max {s[0]:.4f}, ||A||_F = {la.norm(A, 'fro'):.4f} = sqrt(sum sigma^2) {np.sqrt((s ** 2).sum()):.4f}")
B = rng.standard_normal((3, 3))
print(f"   submultiplicativity: ||AB||_2 = {la.norm(A @ B, 2):.3f} <= ||A|| ||B|| = {la.norm(A, 2) * la.norm(B, 2):.3f}")

# ---------------------------------------------------------------- 2. the 2x2 example
print("\n2. A = [[1, 1], [1, 1+eps]], b = [2, 2+eps], perturbation db = [0, eps]")
for eps in (1e-2, 1e-6, 1e-10):
    A2 = np.array([[1.0, 1.0], [1.0, 1.0 + eps]]); b2 = np.array([2.0, 2.0 + eps])
    x0 = la.solve(A2, b2); x1 = la.solve(A2, b2 + np.array([0.0, eps]))
    k = la.norm(A2, np.inf) * la.norm(la.inv(A2), np.inf)
    rel_b = eps / la.norm(b2, np.inf); rel_x = la.norm(x1 - x0, np.inf) / la.norm(x0, np.inf)
    print(f"   eps = {eps:.0e}: kappa_inf = {k:.3e} (4/eps = {4 / eps:.1e}); x = {x0}, perturbed x = {x1};"
          f" relative change b {rel_b:.1e} -> x {rel_x:.2f}")

# ---------------------------------------------------------------- 3. Hilbert matrices
print("\n3. Hilbert matrices H_ij = 1/(i+j-1), exact x = ones")
print(f"{'n':>3} {'kappa_2':>10} {'digits lost':>12} {'err f64':>10} {'err f32':>10} {'bound kappa*eps64':>18}")
H_rows = []
for n in range(2, 14):
    H = la.hilbert(n); xt = np.ones(n); b = H @ xt
    k = np.linalg.cond(H)
    e64 = la.norm(la.solve(H, b) - xt) / la.norm(xt)
    e32 = la.norm(la.solve(H.astype(np.float32), b.astype(np.float32)).astype(float) - xt) / la.norm(xt)
    H_rows.append((n, k, e64, e32))
    print(f"{n:3d} {k:10.2e} {np.log10(k):12.1f} {e64:10.1e} {e32:10.1e} {k * np.finfo(float).eps:18.1e}")


# ---------------------------------------------------------------- 4. mixed-precision iterative refinement
def refine(A, b, iters=8):
    lu32 = la.lu_factor(A.astype(np.float32))                       # O(n^3) work in single precision
    x = la.lu_solve(lu32, b.astype(np.float32)).astype(np.float64)
    hist = []
    for _ in range(iters):
        r = b - A @ x                                                # residual in DOUBLE precision
        d = la.lu_solve(lu32, r.astype(np.float32)).astype(np.float64)
        x = x + d
        hist.append(x.copy())
    return hist


n = 300
Q1, _ = la.qr(rng.standard_normal((n, n))); Q2, _ = la.qr(rng.standard_normal((n, n)))
A = Q1 @ np.diag(np.logspace(0, -4, n)) @ Q2.T                     # kappa = 1e4
xt = rng.standard_normal(n); b = A @ xt
hist = refine(A, b)
err0 = la.norm(la.solve(A.astype(np.float32), b.astype(np.float32)) - xt) / la.norm(xt)
errs_ref = [la.norm(h - xt) / la.norm(xt) for h in hist]
print(f"\n4. iterative refinement, n = {n}, kappa = {np.linalg.cond(A):.1e}")
print(f"   float32 LU alone: error {err0:.1e};  after refinement steps 1..6: " + ", ".join(f"{e:.1e}" for e in errs_ref[:6]))
print(f"   float64 LU: error {la.norm(la.solve(A, b) - xt) / la.norm(xt):.1e}")
A_bad = Q1 @ np.diag(np.logspace(0, -9, n)) @ Q2.T                 # kappa = 1e9 > 1/eps32: refinement cannot converge
h_bad = refine(A_bad, A_bad @ xt)
print(f"   kappa = 1e9 (above 1/eps32 = {1 / np.finfo(np.float32).eps:.0e}): errors " +
      ", ".join(f"{la.norm(h - xt) / la.norm(xt):.1e}" for h in h_bad[:4]) + "  -> does not converge")

# ---------------------------------------------------------------- 5. EVD
print("\n5. eigenvalues")
S = np.array([[2.0, 1.0], [1.0, 3.0]])
w, Q = la.eigh(S)
print(f"   symmetric [[2,1],[1,3]]: lambda = {w}, Q^T Q = I to {np.abs(Q.T @ Q - np.eye(2)).max():.0e},"
      f" kappa_2 = |lmax|/|lmin| = {abs(w).max() / abs(w).min():.4f} = cond {np.linalg.cond(S):.4f}")
R = np.array([[0.0, -1.0], [1.0, 0.0]])
print(f"   rotation by 90 deg: eigenvalues {la.eigvals(R)} (complex), singular values {la.svdvals(R)}")
n = 20
J = np.diag(np.ones(n - 1), 1)                                       # Jordan block with eigenvalue 0 (defective)
Jp = J.copy(); Jp[-1, 0] = 1e-16 ** 1                                # perturbation of size 1e-16
lam = la.eigvals(Jp)
print(f"   20x20 Jordan block, one entry changed by 1e-16: |eigenvalues| become {np.abs(lam).max():.3f}"
      f" (= 1e-16^(1/20) = {1e-16 ** (1 / 20):.3f}) instead of 0")
Sym = rng.standard_normal((n, n)); Sym = Sym + Sym.T; E = 1e-10 * rng.standard_normal((n, n)); E = E + E.T
print(f"   symmetric 20x20, perturbation 2e-10: eigenvalues move by at most {np.abs(la.eigvalsh(Sym + E) - la.eigvalsh(Sym)).max():.1e}"
      f" (Weyl bound ||E||_2 = {la.norm(E, 2):.1e})")

# ---------------------------------------------------------------- 6. SVD, rank and subspaces
A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
U, s, Vt = la.svd(A)
tol = max(A.shape) * np.finfo(float).eps * s[0]
print(f"\n6. SVD of the rank-deficient 4x3 matrix of the notes: sigma = {s}")
print(f"   tolerance max(m,n) eps sigma_1 = {tol:.1e} -> numerical rank {np.sum(s > tol)}; matrix_rank: {np.linalg.matrix_rank(A)}")
A = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 1.0], [2.0, 0.0, 2.0], [1.0, 1.0, 2.0]])
U, s, Vt = la.svd(A); V = Vt.T
print(f"   4x3 with col3 = col1 + col2: sigma = {s}; ||A v1|| = {la.norm(A @ V[:, 0]):.4f} = sigma_1,"
      f" ||A v3|| = {la.norm(A @ V[:, 2]):.1e}; kernel direction v3 = {V[:, 2]} (proportional to [1, 1, -1])")

# ---------------------------------------------------------------- 7. least squares
print("\n7. least squares: fit y = sum_k c_k t^k on t in [0, 10], 100 points, noise 0.5")
t = np.linspace(0, 10, 100); y_exact = 2.5 + 1.2 * t - 0.05 * t ** 2
y = y_exact + rng.normal(0, 0.5, t.size)
rows = []
for deg in (2, 5, 8, 11):
    Am = np.vander(t, deg + 1, increasing=True)
    ctrue = np.zeros(deg + 1); ctrue[:3] = [2.5, 1.2, -0.05]
    yt = Am @ ctrue                                                  # noiseless data from the true model: error is pure numerics
    c_ne = la.solve(Am.T @ Am, Am.T @ yt, assume_a="pos") if deg < 11 else np.linalg.lstsq(Am.T @ Am, Am.T @ yt, rcond=None)[0]
    Qm, Rm = la.qr(Am, mode="economic"); c_qr = la.solve_triangular(Rm, Qm.T @ yt)
    c_svd = la.lstsq(Am, yt, cond=None)[0]
    k = np.linalg.cond(Am)
    rows.append((deg, k, *[la.norm(c - ctrue) / la.norm(ctrue) for c in (c_ne, c_qr, c_svd)]))
    print(f"   degree {deg:2d}: kappa(A) = {k:.1e}, kappa(A^T A) = {k * k:.1e};  coefficient error:"
          f" normal eq {rows[-1][2]:.1e}, QR {rows[-1][3]:.1e}, SVD {rows[-1][4]:.1e}")
Am = np.vander(t, 3, increasing=True)
c, res, rank, sv = la.lstsq(Am, y)
print(f"   noisy quadratic fit (SVD lstsq): c = {c} (true [2.5, 1.2, -0.05]), kappa = {sv[0] / sv[-1]:.1f}")
# truncated SVD on a nearly singular problem
Am = np.vander(t / 10, 12, increasing=True); ct = rng.standard_normal(12)
yn = Am @ ct + 1e-6 * rng.standard_normal(t.size)
U, s, Vt = la.svd(Am, full_matrices=False)
print(f"   degree-11 basis on [0,1], data noise 1e-6: sigma from {s[0]:.1e} to {s[-1]:.1e}")
for rc in (0.0, 1e-12, 1e-8, 1e-5):
    keep = s > rc * s[0]
    cx = Vt.T[:, keep] @ ((U[:, keep].T @ yn) / s[keep])
    print(f"     rcond {rc:.0e}: keep {keep.sum():2d} singular values, residual {la.norm(Am @ cx - yn):.2e}, ||c|| = {la.norm(cx):.2e}")
# minimum-norm solution
A_u = rng.standard_normal((3, 6)); b_u = rng.standard_normal(3)
x_mn = la.pinv(A_u) @ b_u; x_other = x_mn + la.null_space(A_u)[:, 0]
print(f"   underdetermined 3x6: pinv solution ||x|| = {la.norm(x_mn):.4f}, residual {la.norm(A_u @ x_mn - b_u):.1e};"
      f" adding a kernel vector: residual {la.norm(A_u @ x_other - b_u):.1e}, ||x|| = {la.norm(x_other):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ns, ks, e64, e32 = map(np.array, zip(*H_rows))
ax[0].semilogy(ns, ks * np.finfo(float).eps, "k--", label="kappa eps_64"); ax[0].semilogy(ns, e64, "o-", label="error float64")
ax[0].semilogy(ns, np.minimum(ks * np.finfo(np.float32).eps, 10), "k:", label="kappa eps_32"); ax[0].semilogy(ns, e32, "s-", label="error float32")
ax[0].set(xlabel="Hilbert size n", ylabel="relative error", title="conditioning sets the error"); ax[0].legend(fontsize=8)
ax[1].semilogy(range(1, len(errs_ref) + 1), errs_ref, "o-", label="refined (LU float32, r float64)")
ax[1].axhline(err0, color="C3", ls="--", label="float32 LU alone"); ax[1].axhline(np.finfo(float).eps * 1e4, color="k", ls=":", label="kappa eps_64")
ax[1].set(xlabel="refinement step", ylabel="relative error", title="mixed-precision iterative refinement"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("conditioning_svd.png", dpi=120)
print("\nsaved conditioning_svd.png")
