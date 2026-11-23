"""
hyperspectral_L13_mc.py  --  Lecture 13 (Monte Carlo) applied to a hyperspectral cube.

  Part 1  sample-mean Monte Carlo: the mean spectrum of a class from N random pixels,
          error bar from the same samples, rms error ~ 1/sqrt(N)
  Part 2  hit-or-miss: fraction of the scene with NDVI > 0.4, with a Wilson interval
  Part 3  Monte Carlo error propagation: band noise -> distribution of NDVI in one pixel
          (compared with linear error propagation)
  Part 4  self-normalised importance sampling: abundances with a physical prior
          (a_k >= 0, sum a_k = 1) using the Lecture-12 Gaussian posterior as proposal;
          effective sample size and largest weight.

Run:  python hyperspectral_L13_mc.py            (synthetic cube; Parts 1-3 also work on real scenes)
      python hyperspectral_L13_mc.py paviaU
Figures: figures/L13_fig*.png
"""
import sys
import os
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hsi_io import load, band, FIG

scene = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
d = load(scene)
cube, wl, labels, names = d["cube"], d["wavelengths"], d["labels"], d["names"]
H, W, B = cube.shape
X = cube.reshape(-1, B).astype(float)
y = labels.ravel()
rng = np.random.default_rng(13)

# ======================================================== Part 1: sample mean
c = np.bincount(y[y > 0] if scene != "synthetic" else y).argmax()   # the most common class
pix = X[y == c]
exact = pix.mean(0)                                  # "the truth": all pixels of the class
Ns = np.unique(np.logspace(1, 3.3, 12).astype(int))
rms = []
for n in Ns:
    errs = [np.sqrt(np.mean((pix[rng.integers(0, len(pix), n)].mean(0) - exact) ** 2)) for _ in range(200)]
    rms.append(np.mean(errs))
sample = pix[rng.integers(0, len(pix), 100)]
est, err = sample.mean(0), sample.std(0, ddof=1) / np.sqrt(100)
# coverage of the +-1 sigma error bar at ONE band, over 500 repetitions (bands are correlated:
# checking all bands of a single repetition is not 500 independent trials!)
j = band(wl, 850)
hits = 0
for _ in range(500):
    s = pix[rng.integers(0, len(pix), 100), j]
    hits += abs(s.mean() - exact[j]) < s.std(ddof=1) / 10
print(f"Part 1: class '{names[c]}', N=100 pixels: +-1 sigma covers the truth at 850 nm in {hits/5:.1f}% of 500 runs (68.3%)")
slope = np.polyfit(np.log(Ns), np.log(rms), 1)[0]
print(f"        rms error ~ N^{slope:.2f}  (Monte Carlo: -0.5)")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].plot(wl, exact, "k", label="all pixels")
ax[0].fill_between(wl, est - err, est + err, alpha=0.4, label="100 random pixels, +-1 sigma")
ax[0].set(xlabel="wavelength [nm]", ylabel="reflectance", title=f"mean spectrum of '{names[c]}'"); ax[0].legend()
ax[1].loglog(Ns, rms, "o-", label="measured"); ax[1].loglog(Ns, rms[0] * np.sqrt(Ns[0] / Ns), "k--", label="1/sqrt(N)")
ax[1].set(xlabel="N pixels", ylabel="rms error", title="sample-mean error"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "L13_fig1_sample_mean.png"), dpi=110)

# ======================================================== Part 2: hit-or-miss
R, NIR = X[:, band(wl, 660)], X[:, band(wl, 850)]
ndvi = (NIR - R) / (NIR + R + 1e-12)
p_true = np.mean(ndvi > 0.4)
print(f"\nPart 2: true fraction with NDVI > 0.4: {p_true:.4f}")
for n in (30, 300, 3000):
    hits = np.count_nonzero(ndvi[rng.integers(0, len(ndvi), n)] > 0.4)
    p = hits / n
    ci = binomtest(hits, n).proportion_ci(method="wilson")
    print(f"   N={n:5d}: p = {p:.3f} +- {np.sqrt(p*(1-p)/n):.3f}   Wilson 95% [{ci.low:.3f}, {ci.high:.3f}]")

# ======================================================== Part 3: error propagation
sig = d.get("noise_sigma", np.full(B, 0.01))
k = np.argmax(ndvi)                                   # a strongly vegetated pixel
r0, n0 = X[k, band(wl, 660)], X[k, band(wl, 850)]
sr, sn = sig[band(wl, 660)], sig[band(wl, 850)]
r_s = r0 + sr * rng.standard_normal(100_000)          # 1e5 noisy "re-observations"
n_s = n0 + sn * rng.standard_normal(100_000)
nd_s = (n_s - r_s) / (n_s + r_s)
# linear propagation: dNDVI/dR = -2 NIR/(NIR+R)^2, dNDVI/dNIR = 2 R/(NIR+R)^2
lin = np.hypot(2 * n0 / (n0 + r0) ** 2 * sr, 2 * r0 / (n0 + r0) ** 2 * sn)
print(f"\nPart 3: NDVI of pixel {k}: MC {nd_s.mean():.4f} +- {nd_s.std():.4f}   linear +- {lin:.4f}")
plt.figure(figsize=(6, 4))
plt.hist(nd_s, 120, density=True, alpha=0.6, label="Monte Carlo")
t = np.linspace(nd_s.min(), nd_s.max(), 300)
plt.plot(t, np.exp(-0.5 * ((t - (n0 - r0) / (n0 + r0)) / lin) ** 2) / (lin * np.sqrt(2 * np.pi)), "k", label="linear propagation")
plt.xlabel("NDVI"); plt.legend(); plt.title("noise -> NDVI"); plt.tight_layout(); plt.savefig(os.path.join(FIG, "L13_fig2_ndvi.png"), dpi=110)

# ======================================================== Part 4: importance sampling
if scene != "synthetic":
    print("\nPart 4 needs the endmembers: run on the synthetic cube."); sys.exit()
A, Kc = d["endmembers"].T, d["endmembers"].shape[0]
Cinv = 1 / sig**2
Sigma = np.linalg.inv(A.T @ (A * Cinv[:, None]))      # Gaussian (flat-prior) posterior, Lecture 12
# a mixed pixel near the river: the interesting case
k = np.argmin(np.abs(d["abundances"].reshape(-1, Kc)[:, 2] - 0.3))
yk, truth = X[k], d["abundances"].reshape(-1, Kc)[k]
m = Sigma @ (A.T @ (yk * Cinv))
Lc = np.linalg.cholesky(Sigma)


def log_q(th):          # unnormalised log-posterior: Gaussian likelihood x physical prior
    r = yk[None, :] - th @ A.T
    ok = np.all(th >= 0, 1) & (np.abs(th.sum(1) - 1) < 0.05)   # simplex, with a little slack
    return np.where(ok, -0.5 * np.sum(r**2 * Cinv, 1), -np.inf)


n = 20000
for scale in (1.0, 1.5, 3.0):
    th = m + scale * rng.standard_normal((n, Kc)) @ Lc.T            # proposal N(m, scale^2 Sigma)
    lg = -0.5 * np.sum(np.linalg.solve(Lc, (th - m).T) ** 2, 0) / scale**2
    lw = log_q(th) - lg                                             # ln w = ln q - ln g
    w = np.exp(lw - lw[np.isfinite(lw)].max()) if np.isfinite(lw).any() else np.zeros(n)
    neff = w.sum() ** 2 / np.sum(w**2) if w.sum() > 0 else 0
    mean = (w[:, None] * th).sum(0) / w.sum()
    print(f"\nPart 4: proposal x{scale}: Neff/N {neff/n:.3f}, largest weight {100*w.max()/w.sum():.2f}%")
    print("   truth          " + " ".join(f"{v:.3f}" for v in truth))
    print("   Gaussian mean  " + " ".join(f"{v:.3f}" for v in m))
    print("   IS mean (prior)" + " ".join(f"{v:.3f}" for v in mean))
print("\nsaved L13_fig1_sample_mean.png, L13_fig2_ndvi.png")
