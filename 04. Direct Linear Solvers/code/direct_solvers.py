"""
direct_solvers.py  --  Lecture 4: numerical stability and direct solvers for Ax = b.

  1. Catastrophic cancellation in the quadratic formula (float32 / float64), the stable formula
  2. Iterative root finding: bisection vs Newton, stopping criteria
  3. Cramer's rule (Leibniz determinant) vs Gaussian elimination: cost and accuracy
  4. Why pivoting: a tiny pivot epsilon, naive vs partial vs full pivoting
  5. PA = LU written from scratch, compared with scipy.linalg.lu_factor; multiple right-hand sides
  6. Determinant via LU (and log-determinant to avoid overflow); inverse via LU
  7. Cholesky from scratch; correlated Gaussian samples Y = mu + L Z
  8. Thomas algorithm for tridiagonal systems: O(n)

Run:  python direct_solvers.py          Figure: direct_solvers.png
"""
import itertools
import math
import time
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

# ---------------------------------------------------------------- 1. quadratic cancellation
print("1. roots of x^2 + b x + 1 = 0: relative error of the small root")
print(f"{'b':>8} {'naive f32':>11} {'stable f32':>11} {'naive f64':>11} {'stable f64':>11}")
bs = np.logspace(1, 8, 36)
errs = {k: [] for k in ("n32", "s32", "n64", "s64")}
for b in bs:
    b = -b
    exact = 1.0 / (-b / 2 + math.sqrt(b * b / 4 - 1))            # small root, computed stably in extended reasoning
    for dt, tag in ((np.float32, "32"), (np.float64, "64")):
        a_, b_, c_ = dt(1), dt(b), dt(1)
        disc = np.sqrt(b_ * b_ - dt(4) * a_ * c_)
        naive = (-b_ - disc) / (dt(2) * a_)                          # the root that suffers cancellation
        u = -b_ - np.sign(b_) * disc
        x1 = u / (dt(2) * a_)
        stable = c_ / (a_ * x1)
        errs["n" + tag].append(abs(float(naive) - exact) / exact)
        errs["s" + tag].append(abs(float(stable) - exact) / exact)
for i in (0, 8, 16, 24, 35):
    print(f"{bs[i]:8.1e} {errs['n32'][i]:11.2e} {errs['s32'][i]:11.2e} {errs['n64'][i]:11.2e} {errs['s64'][i]:11.2e}")
with np.errstate(over="ignore"):
    naive_c = np.sqrt(np.float64(1e200) ** 2 + np.float64(1e200) ** 2)
print("   overflow in an intermediate step, a = b = 1e200: sqrt(a^2 + b^2) =", naive_c, "; math.hypot(a, b) =", math.hypot(1e200, 1e200))

# ---------------------------------------------------------------- 2. bisection vs Newton
f = lambda x: x * x - 10 * x + 9
df = lambda x: 2 * x - 10
a, b, k = 5.0, 12.0, 0
while (b - a) / 2 > 1e-12:
    m = 0.5 * (a + b)
    a, b = (a, m) if f(a) * f(m) <= 0 else (m, b)
    k += 1
x, kn = 12.0, 0
while True:
    xn = x - f(x) / df(x); kn += 1
    if abs(xn - x) / abs(xn) < 1e-12: break
    x = xn
print(f"\n2. root x = 9 to 1e-12: bisection {k} iterations ({0.5 * (a + b):.13f}), Newton {kn} iterations ({xn:.13f})")

# ---------------------------------------------------------------- 3. Cramer vs Gaussian elimination
def det_leibniz(A):
    n = A.shape[0]; s = 0.0
    for p in itertools.permutations(range(n)):
        inv = sum(1 for i in range(n) for j in range(i + 1, n) if p[i] > p[j])
        s += (-1) ** inv * np.prod([A[i, p[i]] for i in range(n)])
    return s


def cramer(A, b):
    d = det_leibniz(A); x = np.empty(len(b))
    for i in range(len(b)):
        Ai = A.copy(); Ai[:, i] = b; x[i] = det_leibniz(Ai) / d
    return x


print("\n3. Cramer (Leibniz determinants) vs LU")
la.solve(np.eye(3), np.ones(3))                                       # warm-up: the first call loads LAPACK
for n in (3, 5, 7, 8):
    A = rng.standard_normal((n, n)); xt = rng.standard_normal(n); b = A @ xt
    t0 = time.perf_counter(); xc = cramer(A, b); tc = time.perf_counter() - t0
    t0 = time.perf_counter(); xl = la.solve(A, b); tl = time.perf_counter() - t0
    print(f"   n = {n}: Cramer {tc * 1e3:9.2f} ms (error {np.abs(xc - xt).max():.1e}),"
          f"  LU {tl * 1e3:6.3f} ms (error {np.abs(xl - xt).max():.1e});  (n+1)! = {math.factorial(n + 1):,}")
print(f"   n = 20 by Cramer would need ~ (n+1)! = {math.factorial(21):.1e} products: about {math.factorial(21) / 1e9 / 3.15e7:,.0f} years at 1 Gflop/s")


# ---------------------------------------------------------------- 4. pivoting
def gauss(A, b, pivot="partial"):
    A = A.astype(float).copy(); b = b.astype(float).copy(); n = len(b); q = np.arange(n)
    for k in range(n - 1):
        if pivot == "partial":
            p = k + np.argmax(np.abs(A[k:, k])); A[[k, p]] = A[[p, k]]; b[[k, p]] = b[[p, k]]
        elif pivot == "full":
            sub = np.abs(A[k:, k:]); p, r = np.unravel_index(np.argmax(sub), sub.shape); p += k; r += k
            A[[k, p]] = A[[p, k]]; b[[k, p]] = b[[p, k]]; A[:, [k, r]] = A[:, [r, k]]; q[[k, r]] = q[[r, k]]
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]; A[i, k:] -= m * A[k, k:]; b[i] -= m * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    out = np.empty(n); out[q] = x
    return out


print("\n4. A = [[eps, 1], [1, 1]], b = [1, 2]  (exact x ~ [1, 1])")
for eps in (1e-4, 1e-8, 1e-12, 1e-16, 1e-20):
    A = np.array([[eps, 1.0], [1.0, 1.0]]); b = np.array([1.0, 2.0])
    xe = np.array([1 / (1 - eps), (1 - 2 * eps) / (1 - eps)])
    e0 = np.abs(gauss(A, b, "none") - xe).max(); e1 = np.abs(gauss(A, b, "partial") - xe).max()
    print(f"   eps = {eps:.0e}: no pivoting error {e0:.1e}, partial pivoting {e1:.1e}")
n = 60
A = rng.standard_normal((n, n)); xt = rng.standard_normal(n); b = A @ xt
for pv in ("none", "partial", "full"):
    print(f"   random 60x60, {pv:7s} pivoting: error {np.abs(gauss(A, b, pv) - xt).max():.1e}")


# ---------------------------------------------------------------- 5. PA = LU from scratch
def lu_inplace(A):
    A = A.astype(float).copy(); n = A.shape[0]; p = np.arange(n); swaps = 0
    for k in range(n - 1):
        r = k + np.argmax(np.abs(A[k:, k]))
        if A[r, k] == 0: raise np.linalg.LinAlgError("singular")
        if r != k: A[[k, r]] = A[[r, k]]; p[[k, r]] = p[[r, k]]; swaps += 1
        A[k + 1:, k] /= A[k, k]                                      # multipliers l_ik stored in place
        A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:])    # update of the trailing submatrix
    return A, p, swaps


def lu_solve(LU, p, b):
    n = len(b); y = b[p].astype(float).copy()
    for i in range(1, n): y[i] -= LU[i, :i] @ y[:i]                 # forward, unit diagonal: no division
    x = y.copy()
    for i in range(n - 1, -1, -1): x[i] = (y[i] - LU[i, i + 1:] @ x[i + 1:]) / LU[i, i]
    return x


n = 300
A = rng.standard_normal((n, n)); LU, p, sw = lu_inplace(A)
L = np.tril(LU, -1) + np.eye(n); U = np.triu(LU)
print(f"\n5. PA = LU, n = {n}: ||PA - LU|| / ||A|| = {np.linalg.norm(A[p] - L @ U) / np.linalg.norm(A):.1e},"
      f" max |l_ik| = {np.abs(L).max():.3f} (<= 1 with partial pivoting)")
lus, piv = la.lu_factor(A)
print(f"   same U as LAPACK dgetrf: {np.abs(np.triu(lus) - U).max():.1e}")
n = 800; A = rng.standard_normal((n, n)); B = rng.standard_normal((n, 50))
t0 = time.perf_counter(); [la.solve(A, B[:, j]) for j in range(50)]; t_rep = time.perf_counter() - t0
t0 = time.perf_counter(); f_ = la.lu_factor(A); [la.lu_solve(f_, B[:, j]) for j in range(50)]; t_once = time.perf_counter() - t0
print(f"   50 right-hand sides, n = {n}: full solve each time {t_rep * 1e3:.0f} ms, factor once + 50 solves {t_once * 1e3:.0f} ms")

# ---------------------------------------------------------------- 6. determinant and inverse
A = rng.standard_normal((5, 5)); LU, p, sw = lu_inplace(A)
print(f"\n6. det via LU {(-1) ** sw * np.prod(np.diag(LU)):.10f}, numpy {np.linalg.det(A):.10f}, Leibniz {det_leibniz(A):.10f}")
A = 10 * rng.standard_normal((400, 400)); LU, p, sw = lu_inplace(A)
sign, logdet = np.linalg.slogdet(A)
with np.errstate(over="ignore"):
    prod = np.prod(np.diag(LU))
print(f"   400x400 with entries ~10: prod(diag U) = {prod} (overflow), "
      f"log|det| = sum log|u_ii| = {np.sum(np.log(np.abs(np.diag(LU)))):.4f} (numpy slogdet {logdet:.4f})")
A = rng.standard_normal((200, 200)); Ainv = la.inv(A); xt = rng.standard_normal(200); b = A @ xt
print(f"   solve vs inv(A) @ b, n = 200: error {np.abs(la.solve(A, b) - xt).max():.1e} vs {np.abs(Ainv @ b - xt).max():.1e};"
      f" ||A inv(A) - I|| = {np.abs(A @ Ainv - np.eye(200)).max():.1e}")


# ---------------------------------------------------------------- 7. Cholesky
def cholesky(A):
    A = A.astype(float).copy(); n = A.shape[0]
    for j in range(n):
        s = A[j, j] - A[j, :j] @ A[j, :j]
        if s <= 0: raise np.linalg.LinAlgError("not positive definite")
        A[j, j] = math.sqrt(s)
        A[j + 1:, j] = (A[j + 1:, j] - A[j + 1:, :j] @ A[j, :j]) / A[j, j]
    return np.tril(A)


M = rng.standard_normal((300, 300)); S = M @ M.T + 300 * np.eye(300)
L = cholesky(S)
print(f"\n7. Cholesky n = 300: ||S - L L^T|| / ||S|| = {np.linalg.norm(S - L @ L.T) / np.linalg.norm(S):.1e},"
      f" vs numpy {np.abs(L - np.linalg.cholesky(S)).max():.1e}")
n = 1500; M = rng.standard_normal((n, n)); S = M @ M.T + n * np.eye(n)
t0 = time.perf_counter(); la.lu_factor(S); t_lu = time.perf_counter() - t0
t0 = time.perf_counter(); la.cho_factor(S); t_ch = time.perf_counter() - t0
print(f"   n = {n}: LAPACK LU {t_lu * 1e3:.0f} ms, Cholesky {t_ch * 1e3:.0f} ms (theory: half the flops)")
try:
    cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
except np.linalg.LinAlgError as e:
    print("   [[1,2],[2,1]] (eigenvalues 3, -1):", e)
Sig = np.array([[4.0, 2.4], [2.4, 2.25]]); Lc = cholesky(Sig); mu = np.array([1.0, -1.0])
Y = mu[:, None] + Lc @ rng.standard_normal((2, 200_000))
print(f"   Y = mu + L Z, 2e5 samples: sample covariance [[{np.cov(Y)[0, 0]:.3f}, {np.cov(Y)[0, 1]:.3f}], [., {np.cov(Y)[1, 1]:.3f}]]"
      " (target [[4, 2.4], [., 2.25]])")


# ---------------------------------------------------------------- 8. Thomas algorithm
def thomas(a, b, c, d):
    """a: sub-diagonal (n-1), b: diagonal (n), c: super-diagonal (n-1)."""
    n = len(b); cp = np.empty(n - 1); dp = np.empty(n)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i - 1] * cp[i - 1]
        if i < n - 1: cp[i] = c[i] / den
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / den
    x = np.empty(n); x[-1] = dp[-1]
    for i in range(n - 2, -1, -1): x[i] = dp[i] - cp[i] * x[i + 1]
    return x


print("\n8. Thomas algorithm: -u'' = 1 on (0,1), u(0) = u(1) = 0, exact u = x(1-x)/2")
for n in (100, 10_000, 1_000_000):
    h = 1 / (n + 1); xs = np.arange(1, n + 1) * h
    t0 = time.perf_counter()
    u = thomas(-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1), h * h * np.ones(n))
    t = time.perf_counter() - t0
    print(f"   n = {n:>9,}: max error {np.abs(u - xs * (1 - xs) / 2).max():.1e}, {t * 1e3:8.1f} ms (dense LU would need {2 / 3 * n ** 3:.1e} flops)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, lab, st in (("n32", "naive, float32", "C3--"), ("s32", "stable, float32", "C3-"), ("n64", "naive, float64", "C0--"), ("s64", "stable, float64", "C0-")):
    ax[0].loglog(bs, np.maximum(errs[k], 1e-17), st, label=lab)
ax[0].axhline(np.finfo(np.float32).eps, color="C3", lw=0.6); ax[0].axhline(np.finfo(np.float64).eps, color="C0", lw=0.6)
ax[0].set(xlabel="|b| (a = c = 1)", ylabel="relative error of the small root", title="catastrophic cancellation"); ax[0].legend(fontsize=8)
ns = np.arange(2, 13)
ax[1].semilogy(ns, [math.factorial(n + 1) for n in ns], "o-", label="Cramer ~ (n+1)!")
ax[1].semilogy(ns, 2 / 3 * ns ** 3.0, "s-", label="Gauss / LU ~ (2/3) n^3")
ax[1].semilogy(ns, 1 / 3 * ns ** 3.0, "^-", label="Cholesky ~ (1/3) n^3")
ax[1].set(xlabel="n", ylabel="flops", title="cost of a direct solve"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("direct_solvers.png", dpi=120)
print("\nsaved direct_solvers.png")
