"""
mc_integrals.py  --  Lecture 13, part I: Monte Carlo integration
  1. hit-or-miss: pi from a quarter circle, with its binomial error
  2. sample mean: integral of x^2 on [0,2], error from the SAME samples
  3. rms error versus N  (the 1/sqrt(N) law), hit-or-miss vs sample mean
  4. grid (Simpson) versus Monte Carlo in d dimensions
  5. the empty cube: volume of the unit ball by hit-or-miss, Wilson interval
Run:  python mc_integrals.py      -> prints numbers, saves mc_integrals.png
"""
import numpy as np
from scipy.special import gamma
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(13)          # one seed, written down


# ---------------------------------------------------------------- 1. hit-or-miss
def hit_or_miss_pi(n, rng):
    x, y = rng.random(n), rng.random(n)
    p_hat = np.mean(x**2 + y**2 < 1.0)       # fraction of hits in the unit square
    pi_hat = 4.0 * p_hat                     # A_box = 1, quarter circle = pi/4
    err = 4.0 * np.sqrt(p_hat * (1 - p_hat) / n)
    return pi_hat, err


pi_hat, err = hit_or_miss_pi(1500, rng)
print(f"1. hit-or-miss, N=1500: pi = {pi_hat:.3f} +- {err:.3f}")


# ---------------------------------------------------------------- 2. sample mean
def mc_integrate(func, a, b, n, rng):
    x = rng.uniform(a, b, n)                 # n points in [a, b]
    y = func(x)
    I = (b - a) * y.mean()
    err = (b - a) * y.std(ddof=1) / np.sqrt(n)
    return I, err


I, err = mc_integrate(lambda x: x**2, 0, 2, 10**4, rng)
print(f"2. sample mean, N=1e4: I = {I:.4f} +- {err:.4f}   (exact 8/3 = 2.6667)")
print(f"   predicted error 2.385/sqrt(N) = {2.385/np.sqrt(1e4):.4f}")

# ---------------------------------------------------------------- 3. rms error vs N
Ns = np.unique(np.logspace(2, 5, 10).astype(int))
rms_sm, rms_hm = [], []
for n in Ns:
    e_sm, e_hm = [], []
    for _ in range(200):
        e_sm.append(mc_integrate(lambda x: x**2, 0, 2, n, rng)[0] - 8 / 3)
        x, y = rng.uniform(0, 2, n), rng.uniform(0, 4, n)          # box [0,2]x[0,4]
        e_hm.append(8.0 * np.mean(y < x**2) - 8 / 3)
    rms_sm.append(np.sqrt(np.mean(np.square(e_sm))))
    rms_hm.append(np.sqrt(np.mean(np.square(e_hm))))
k_sm = np.mean(np.array(rms_sm) * np.sqrt(Ns))
k_hm = np.mean(np.array(rms_hm) * np.sqrt(Ns))
print(f"3. rms*sqrt(N): sample mean {k_sm:.3f} (theory 2.385), hit-or-miss {k_hm:.3f} (3.771)")
print(f"   hit-or-miss needs {(k_hm/k_sm)**2:.1f}x more points for the same error")


# ---------------------------------------------------------------- 4. Simpson vs MC
def simpson_1d(n):
    """Composite Simpson of (pi/2) sin(pi x) on [0,1] with n points (n odd)."""
    x = np.linspace(0, 1, n)
    w = np.ones(n); w[1:-1:2] = 4; w[2:-1:2] = 2
    return np.sum(w * np.pi / 2 * np.sin(np.pi * x)) / (3 * (n - 1))


print("4. product integrand prod_i (pi/2) sin(pi x_i) on [0,1]^d, exact = 1")
for d in (1, 2, 4, 8):
    n = 3 if d == 8 else 9
    N = n**d
    err_grid = abs(simpson_1d(n) ** d - 1)     # product rule = (1-D rule)^d
    sig_f = np.sqrt((np.pi**2 / 8) ** d - 1)
    print(f"   d={d}: Simpson with N={N:6d}: error {err_grid:.2e} | MC sigma_f/sqrt(N) = {sig_f/np.sqrt(N):.2e}")

# ---------------------------------------------------------------- 5. empty cube
print("5. unit ball in the cube [-1,1]^d, N = 1e6")
for d in (2, 5, 10, 15, 20):
    hits = 0
    for _ in range(10):                          # 10 chunks of 1e5 points: low memory
        x = rng.uniform(-1, 1, (100_000, d))
        hits += np.count_nonzero(np.sum(x * x, axis=1) < 1)
    N = 1_000_000
    exact = np.pi ** (d / 2) / gamma(d / 2 + 1)
    ci = binomtest(hits, N).proportion_ci(method="wilson")
    print(f"   d={d:2d}: hits {hits:7d}  estimate {2**d*hits/N:8.4f}  exact {exact:8.4f}"
          f"  Wilson 95% on p: [{ci.low:.2e}, {ci.high:.2e}]")

# ---------------------------------------------------------------- figure
plt.loglog(Ns, rms_sm, "o-", label=f"sample mean ({k_sm:.2f}/sqrt N)")
plt.loglog(Ns, rms_hm, "s-", label=f"hit-or-miss ({k_hm:.2f}/sqrt N)")
plt.loglog(Ns, 2.385 / np.sqrt(Ns), "k--", lw=1, label="2.385 / sqrt N")
plt.xlabel("N"); plt.ylabel("rms error"); plt.legend(); plt.title("Monte Carlo error ~ 1/sqrt(N)")
plt.savefig("mc_integrals.png", dpi=120)
print("saved mc_integrals.png")
