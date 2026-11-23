"""
knee_inference.py  --  Lecture 12 end to end, on one simulated record.

  1. simulate a 64 s record (periodogram) of knee noise
  2. maximum likelihood (Whittle) + Laplace/Hessian error bars
  3. posterior on a 3-D grid (log-sum-exp normalisation, sigma2 factor trick)
  4. marginals, MAP / median / mean, equal-tailed and HPD intervals
  5. prior flat in log10(fk) versus prior flat in fk  (the Jacobian!)
  6. coverage of  theta_hat +- 1 sigma  over many simulated records
Figures are saved as PNG next to this script.

Run:  python knee_inference.py          (about 20-40 s on a laptop)
"""
import time
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from knee import frequencies, spectrum, simulate_periodogram, neg_log_like, TRUTH

rng = np.random.default_rng(12)          # one seed, written down
f = frequencies(T=64.0)
P = simulate_periodogram(f, **TRUTH, rng=rng)
names = ["log10 sigma2", "log10 fk", "alpha"]
theta_true = np.array([-3.0, 0.0, 1.0])

# ---------------------------------------------------------------- 2. MLE
def fit_mle(f, P, start=(-3.0, 0.0, 1.0)):
    res = minimize(neg_log_like, start, args=(f, P), method="Nelder-Mead",
                   options=dict(xatol=1e-7, fatol=1e-7, maxiter=4000))
    return res.x, res.fun


def hessian(fun, x, h=1e-4):
    """Central finite-difference Hessian: O(d^2) function evaluations."""
    d = len(x)
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            e_i = np.eye(d)[i] * h
            e_j = np.eye(d)[j] * h
            H[i, j] = (fun(x + e_i + e_j) - fun(x + e_i - e_j)
                       - fun(x - e_i + e_j) + fun(x - e_i - e_j)) / (4 * h * h)
    return H


theta_hat, nll_min = fit_mle(f, P)
H = hessian(lambda t: neg_log_like(t, f, P), theta_hat)
C = np.linalg.inv(H)                       # Laplace covariance  C = H^-1
err = np.sqrt(np.diag(C))
corr = C / np.outer(err, err)
print("=== Maximum likelihood (Whittle) + Laplace errors ===")
for n, t, e, tt in zip(names, theta_hat, err, theta_true):
    print(f"  {n:13s} = {t:+.4f} +- {e:.4f}   (truth {tt:+.3f}, pull {(t-tt)/e:+.2f})")
print(f"  fk_hat = {10**theta_hat[1]:.3f} Hz")
print("  correlation matrix:\n", np.array2string(corr, precision=2))

# ---------------------------------------------------------------- 3. grid
t0 = time.time()
u = np.linspace(-3.25, -2.75, 51)          # log10 sigma2   (+-0.25 dex)
v = np.linspace(-2.0, np.log10(50.0), 161)  # log10 fk       (0.01 - 50 Hz)
a = np.linspace(0.0, 3.0, 121)             # alpha
s2 = 10.0 ** u
N = f.size
lnf = np.log(f)
lp = np.empty((u.size, v.size, a.size))
for k, alpha in enumerate(a):
    # g = 1 + (fk/f)^alpha  for all fk at once:  shape (nv, nf)
    g = 1.0 + np.exp(alpha * (v[:, None] * np.log(10.0) - lnf[None, :]))
    A = np.log(g).sum(axis=1)              # sum ln g_j      (depends on fk, alpha only)
    B = (P[None, :] / g).sum(axis=1)       # sum P_j / g_j
    # -lnL = N ln s2 + A + B / s2   -> each sigma2 costs one operation
    lp[:, :, k] = -(N * np.log(s2)[:, None] + A[None, :] + B[None, :] / s2[:, None])
# flat priors in (log10 s2, log10 fk, alpha): ln pi = const inside the box
post = np.exp(lp - lp.max())               # subtract the max BEFORE exp (underflow!)
post /= post.sum()
print(f"\n=== Grid posterior: {post.size:,} points in {time.time()-t0:.2f} s ===")

# fraction of the grid that matters
sorted_p = np.sort(post.ravel())[::-1]
n99 = np.searchsorted(np.cumsum(sorted_p), 0.99) + 1
print(f"  {n99} of {post.size} points ({100*n99/post.size:.2f}%) hold 99% of the mass")

# ---------------------------------------------------------------- 4. summaries
m_v = post.sum(axis=(0, 2))                # marginal of log10 fk
m_a = post.sum(axis=(0, 1))                # marginal of alpha
m_va = post.sum(axis=0)                    # 2-D marginal (log10 fk, alpha)


def summarise(x, p, label):
    p = p / p.sum()
    cdf = np.cumsum(p) - 0.5 * p             # cdf at the bin centres
    mean = np.sum(x * p)
    std = np.sqrt(np.sum((x - mean) ** 2 * p))
    q = lambda c: np.interp(c, cdf, x)
    print(f"  {label:9s} mean {mean:+.3f} +- {std:.3f}   median {q(0.5):+.3f}"
          f"   68% equal-tailed [{q(0.16):+.3f}, {q(0.84):+.3f}]   MAP {x[np.argmax(p)]:+.3f}")
    return mean, std


def hpd(x, p, mass=0.68):
    """Highest-posterior-density set on a grid: take the highest points first."""
    order = np.argsort(p)[::-1]
    keep = order[: np.searchsorted(np.cumsum(p[order]) / p.sum(), mass) + 1]
    return x[keep].min(), x[keep].max()


print("=== Posterior summaries (prior flat in log10 fk) ===")
summarise(v, m_v, "log10 fk")
summarise(a, m_a, "alpha")
lo, hi = hpd(v, m_v)
print(f"  68% HPD of log10 fk: [{lo:+.3f}, {hi:+.3f}]  ->  fk in [{10**lo:.2f}, {10**hi:.2f}] Hz")

# ---------------------------------------------------------------- 5. prior flat in fk
# A prior flat in fk is, in the variable v = log10 fk, proportional to dfk/dv = fk ln10:
m_v_flatfk = m_v * 10.0 ** v
m_v_flatfk /= m_v_flatfk.sum()
med = lambda p: np.interp(0.5, np.cumsum(p / p.sum()) - 0.5 * p / p.sum(), v)
print("\n=== Does the prior matter? (64 s of data) ===")
print(f"  median fk, prior flat in log fk: {10**med(m_v):.3f} Hz")
print(f"  median fk, prior flat in fk    : {10**med(m_v_flatfk):.3f} Hz")
print("  (re-run with T=8 in frequencies() to see the prior take over)")

# ---------------------------------------------------------------- 6. coverage
n_rec = 300
inside = np.zeros(3)
for r in range(n_rec):
    Pr = simulate_periodogram(f, **TRUTH, rng=rng)
    th, _ = fit_mle(f, Pr, start=theta_hat)
    Hr = hessian(lambda t: neg_log_like(t, f, Pr), th)
    er = np.sqrt(np.diag(np.linalg.inv(Hr)))
    inside += np.abs(th - theta_true) < er
print(f"\n=== Coverage of theta_hat +- 1 sigma over {n_rec} records (nominal 68.3%) ===")
for n, c in zip(names, inside / n_rec):
    print(f"  {n:13s} {100*c:.1f}%")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ax[0].loglog(f, P, ",", color="0.6", label="periodogram")
ax[0].loglog(f, spectrum(f, **TRUTH), "k", lw=2, label="truth")
ax[0].loglog(f, spectrum(f, 10**theta_hat[0], 10**theta_hat[1], theta_hat[2]), "r--", lw=2, label="MLE")
ax[0].set(xlabel="f [Hz]", ylabel="S(f)", title="one 64 s record")
ax[0].legend()
ax[1].contourf(a, v, m_va, levels=20, cmap="Greens")
ax[1].plot(1, 0, "r*", ms=14, label="truth")
ax[1].set(xlabel="alpha", ylabel="log10 fk", title="2-D marginal (fk, alpha): degeneracy")
ax[1].legend()
ax[2].plot(10**v, m_v / m_v.max(), label="prior flat in log fk")
ax[2].plot(10**v, m_v_flatfk / m_v_flatfk.max(), "--", label="prior flat in fk")
ax[2].set(xscale="log", xlabel="fk [Hz]", ylabel="marginal (peak = 1)", title="prior sensitivity")
ax[2].legend()
fig.tight_layout()
fig.savefig("knee_inference.png", dpi=120)
print("\nsaved knee_inference.png")
