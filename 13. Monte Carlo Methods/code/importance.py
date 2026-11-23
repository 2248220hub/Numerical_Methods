"""
importance.py  --  Lecture 13, parts III-IV: importance sampling
  A. three bumps: uniform vs Gaussian proposals of several widths
  B. a rare event, P(X > 4), with a shifted-exponential proposal
  C. tails: integral of a Student t3 with a Gaussian (fails) and a Cauchy proposal
  D. self-normalised importance sampling of a posterior + effective sample size
  E. why it collapses: ESS/N versus dimension
Run:  python importance.py   -> prints numbers, saves importance.png
"""
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(2027)


def importance_sampling(f, g_pdf, g_draw, n, rng):
    x = g_draw(n, rng)              # 1. samples from g
    w = f(x) / g_pdf(x)             # 2. weights f/g
    return w.mean(), w.std(ddof=1) / np.sqrt(n), w


# ---------------------------------------------------------------- A. three bumps
def bumps(x):
    return (np.exp(-0.5 * ((x + 1.5) / 0.3) ** 2) + 2 * np.exp(-0.5 * (x / 0.25) ** 2)
            + 1.5 * np.exp(-0.5 * ((x - 1.2) / 0.35) ** 2))


I_true = np.sqrt(2 * np.pi) * (0.3 + 2 * 0.25 + 1.5 * 0.35)
n, runs = 10**4, 300
print("A. three bumps, N=1e4, relative rms error over", runs, "runs")
res = [rng.uniform(-4, 4, n) for _ in range(runs)]
rel = np.std([8 * bumps(x).mean() for x in res]) / I_true
print(f"   uniform on [-4,4]        : {100*rel:.2f}%")
for s in (0.5, 0.7, 1.0, 2.0):
    est = [importance_sampling(bumps, stats.norm(0, s).pdf,
                               lambda m, r: r.normal(0, s, m), n, rng)[0] for _ in range(runs)]
    print(f"   Gaussian proposal s={s:<4}: {100*np.std(est)/I_true:.2f}%")

# ---------------------------------------------------------------- B. rare event
lam = 4.0
I, err, w = importance_sampling(
    f=stats.norm.pdf,
    g_pdf=lambda x: lam * np.exp(-lam * (x - 4)),
    g_draw=lambda m, r: 4 + r.exponential(1 / lam, m),
    n=10**4, rng=rng)
plain = np.mean(rng.standard_normal(10**6) > 4)
print(f"B. P(X>4): IS (1e4 pts) {I:.4e} +- {err:.1e}  | plain MC (1e6 pts) {plain:.2e}"
      f" | exact {stats.norm.sf(4):.4e}")

# ---------------------------------------------------------------- C. tails
t3 = stats.t(3).pdf
for name, prop in (("Gaussian (lighter tails)", stats.norm(0, 1)), ("Cauchy (heavier tails)", stats.cauchy())):
    errs, wmax = [], []
    for _ in range(200):
        I, _, w = importance_sampling(t3, prop.pdf, lambda m, r: prop.rvs(size=m, random_state=r), 10**4, rng)
        errs.append(abs(I - 1)); wmax.append(w.max() / w.sum())
    print(f"C. {name:26s}: median error {np.median(errs):.4f}, worst {np.max(errs):.3f},"
          f" largest weight up to {100*np.max(wmax):.2f}% of the total")

# ---------------------------------------------------------------- D. self-normalised
# a skewed 1-D "posterior" known up to a constant: q(t) = exp(-ln-normal shape)
def log_q(t):                        # unnormalised log-posterior (a lognormal in t > 0)
    lt = np.log(np.where(t > 0, t, 1.0))           # safe log (value unused where t <= 0)
    return np.where(t > 0, -lt - 0.5 * (lt / 0.4) ** 2, -np.inf)


# Laplace proposal: mode and curvature of log q
grid = np.linspace(1e-3, 6, 200001)
lq = log_q(grid)
mode = grid[np.argmax(lq)]
h = 1e-4
H = -(log_q(mode + h) - 2 * log_q(mode) + log_q(mode - h)) / h**2
print(f"D. Laplace proposal: mode {mode:.4f}, sigma {1/np.sqrt(H):.4f}")
true_mean = np.exp(0.4**2 / 2)
for scale in (1.0, 1.5, 2.5):
    g = stats.norm(mode, scale / np.sqrt(H))
    t = g.rvs(size=5000, random_state=rng)
    lw = log_q(t) - g.logpdf(t)                    # ln w = ln q - ln g
    w = np.exp(lw - lw.max())                      # subtract the max, then exp
    mean = np.sum(w * t) / np.sum(w)               # self-normalised: Z cancels
    neff = w.sum() ** 2 / np.sum(w**2)
    print(f"   proposal x{scale}: mean {mean:.4f} (exact {true_mean:.4f}), Neff/N {neff/5000:.2f},"
          f" largest weight {100*w.max()/w.sum():.2f}%")

# ---------------------------------------------------------------- E. ESS vs dimension
print("E. target N(0,I_d), proposal N(0, s^2 I_d):  ESS/N = [s^2/sqrt(2 s^2 - 1)]^(-d)")
ds = np.arange(1, 101)
for s in (1.2, 2.0):
    theory = (s**2 / np.sqrt(2 * s**2 - 1)) ** (-ds)
    d = 20
    x = rng.normal(0, s, (10**5, d))
    lw = stats.norm.logpdf(x).sum(1) - stats.norm(0, s).logpdf(x).sum(1)
    w = np.exp(lw - lw.max())
    print(f"   s={s}: d=20 measured {w.sum()**2/np.sum(w**2)/1e5:.2e}, theory {theory[19]:.2e};"
          f"  d=100 theory {theory[-1]:.2e}")
    plt.semilogy(ds, theory, label=f"proposal width {s}")
plt.axhline(1e-5, color="k", ls=":", label="ESS = 1 (N = 1e5)")
plt.xlabel("dimension d"); plt.ylabel("ESS / N"); plt.legend(); plt.title("importance sampling collapses with d")
plt.savefig("importance.png", dpi=120)
print("saved importance.png")
