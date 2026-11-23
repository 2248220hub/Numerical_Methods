"""
hyperspectral_L10_noise.py  --  Lecture 10 (spectral estimation) applied to the noise of a cube.

Every later step (Bayes classifier, unmixing, Monte Carlo) needs the noise of each band.

  Part 1  sigma per band from the data alone: the shift-difference estimator
          sigma ~ std(x[i, j+1] - x[i, j]) / sqrt(2)  (scene texture is smooth, noise is not),
          compared with the true sigma of the synthetic cube. SNR per band.
  Part 2  along-track noise spectrum: a push-broom records one row per time step, so each column
          is a TIME SERIES. Add a slow 1/f drift to every detector, take the residual
          (data - model) along the track and estimate its spectrum with Welch and a chi^2 band.
  Part 3  fit sigma^2, f_k, alpha with the Whittle likelihood on the column periodograms.

Run:  python hyperspectral_L10_noise.py          Figures: figures/L10_fig1.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import chi2
from scipy.optimize import minimize
from hsi_io import load, band, FIG

d = load("synthetic")
cube, wl, sig_true = d["cube"].astype(float), d["wavelengths"], d["noise_sigma"]
H, W, B = cube.shape
model = d["abundances"] @ d["endmembers"]

# ---------------------------------------------------------------- 1. sigma per band from the data
mad = lambda a: np.median(np.abs(a - np.median(a, axis=(0, 1))), axis=(0, 1)) * 1.4826
d1 = np.diff(cube, axis=1)                                   # x[j+1] - x[j]: variance 2 sigma^2
d2 = cube[:, 2:] - 2 * cube[:, 1:-1] + cube[:, :-2]          # second difference: variance 6 sigma^2, kills gradients
sig1 = mad(d1) / np.sqrt(2)
sig_hat = mad(d2) / np.sqrt(6)
print("1. noise per band from shift differences (robust MAD)")
print(f"   first difference : sigma_hat / sigma_true median {np.median(sig1 / sig_true):.3f}, max {np.max(sig1 / sig_true):.3f}"
      "  (scene gradients leak in)")
ratio = sig_hat / sig_true
print("   second difference:", end="")
print(f" sigma_hat / sigma_true: median {np.median(ratio):.3f}, range {ratio.min():.3f} .. {ratio.max():.3f}")
snr = cube.mean((0, 1)) / sig_hat
print(f"   SNR: best {snr.max():.0f} at {wl[snr.argmax()]:.0f} nm, worst {snr.min():.1f} at {wl[snr.argmin()]:.0f} nm"
      f"  -> {np.sum(snr < 5)} bands with SNR < 5 (the water-vapour bands: drop or down-weight them)")

# ---------------------------------------------------------------- 2. along-track spectrum with a 1/f drift
line_rate = 100.0                                  # rows per second
b = band(wl, 860)
rng = np.random.default_rng(5)
sigma2, fk, alpha = sig_true[b] ** 2, 2.0, 1.5     # truth of the drift we add
n = 1 << 12                                        # a long acquisition: 4096 rows
f = np.fft.rfftfreq(n, 1 / line_rate)
S = np.zeros_like(f); S[1:] = 2 * sigma2 / line_rate * (fk / f[1:]) ** alpha   # one-sided PSD of the drift
X = np.sqrt(n * line_rate * S / 4) * (rng.standard_normal((W, f.size)) + 1j * rng.standard_normal((W, f.size)))
drift = np.fft.irfft(X, n, axis=1)                 # one drift per detector (column)
white = sig_true[b] * rng.standard_normal((W, n))
resid = drift + white                              # data - model along the track, every column
fw, Pw = welch(resid, fs=line_rate, window="hann", nperseg=512, axis=1)
Pm = Pw.mean(0)                                    # average over the 128 detectors too
K = (n - 512) // 256 + 1
nu = 2 * 0.95 * K * W
lo, hi = nu / chi2.ppf(0.975, nu), nu / chi2.ppf(0.025, nu)
Strue = 2 * sigma2 / line_rate * (1 + (fk / np.maximum(fw, 1e-9)) ** alpha)
inside = np.mean((Strue[1:] >= Pm[1:] * lo) & (Strue[1:] <= Pm[1:] * hi))
print(f"\n2. along-track residual at {wl[b]:.0f} nm, 128 detectors x 4096 rows at {line_rate:.0f} lines/s")
print(f"   Welch L = 512, K = {K} per detector, nu ~ {nu:.0f}: band {lo:.3f} .. {hi:.3f}; truth inside in {100 * inside:.0f}% of bins")

# ---------------------------------------------------------------- 3. Whittle fit of the knee
fp = np.fft.rfftfreq(n, 1 / line_rate)[1:-1]
Pp = (2 * np.abs(np.fft.rfft(resid, axis=1)) ** 2 / (n * line_rate))[:, 1:-1]


def nll(th):
    s2, lfk, a = th
    Sm = 2 * np.exp(s2) / line_rate * (1 + (np.exp(lfk) / fp) ** a)
    return np.sum(np.log(Sm)) * W + np.sum(Pp / Sm)


res = minimize(nll, [np.log(sigma2 * 0.5), np.log(1.0), 1.0], method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
s2h, fkh, ah = np.exp(res.x[0]), np.exp(res.x[1]), res.x[2]
print(f"3. Whittle on {W} column periodograms: sigma^2 {s2h:.3e} (true {sigma2:.3e}), f_k {fkh:.2f} Hz (true {fk}),"
      f" alpha {ah:.3f} (true {alpha})")
print(f"   -> averaging more than ~1/f_k = {1 / fkh:.1f} s of rows ({line_rate / fkh:.0f} lines) no longer beats down the drift")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].semilogy(wl, sig_true, "k", label="true sigma"); ax[0].semilogy(wl, sig_hat, "C1--", label="shift-difference estimate")
ax[0].set(xlabel="wavelength [nm]", ylabel="noise sigma", title="noise per band"); ax[0].legend()
ax[1].fill_between(fw[1:], Pm[1:] * lo, Pm[1:] * hi, alpha=0.3, label="95% band")
ax[1].loglog(fw[1:], Pm[1:], label="Welch, mean of 128 detectors"); ax[1].loglog(fw[1:], Strue[1:], "k--", label="truth")
ax[1].loglog(fp, 2 * s2h / line_rate * (1 + (fkh / fp) ** ah), "C3:", label="Whittle fit")
ax[1].set(xlabel="frequency along track [Hz]", ylabel="PSD", title=f"along-track noise at {wl[b]:.0f} nm"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "L10_fig1.png"), dpi=120)
print("saved L10_fig1.png")
