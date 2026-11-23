"""
hyperspectral_L09_response.py  --  Lecture 9 (convolution, deconvolution) applied to a cube.

Two convolutions blur every hyperspectral measurement:
  * along WAVELENGTH, the spectral response function (SRF) of each band;
  * in SPACE, the point-spread function (PSF) of the optics and the detector.

  Part 1  spectral: a 1-nm spectrum convolved with a Gaussian SRF (FWHM 10 nm) by np.convolve and
          by FFT with zero padding; boundary choice (zero vs reflect) at the ends of the range.
  Part 2  spatial: the 860-nm band blurred by a Gaussian PSF (sigma 1.5 px) plus noise,
          then deconvolved: plain inverse, best Tikhonov lambda (needs the truth),
          and Wiener with the noise floor read from the data.

Run:  python hyperspectral_L09_response.py        Figures: figures/L09_fig1.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len
from hsi_io import load, band, FIG

g = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2)
rms = lambda a: np.sqrt(np.mean(a ** 2))

# ---------------------------------------------------------------- 1. spectral response
lam = np.arange(400.0, 2501.0, 1.0)
d = load("synthetic")
spec = np.interp(lam, d["wavelengths"], d["endmembers"][0]) * (1 - 0.6 * g(lam, 760, 1.5))
fwhm = 10.0; s = fwhm / 2.3548
k = g(np.arange(-30, 31), 0, s); k /= k.sum()                 # centred kernel, M = 61, m+ = m- = 30
M = k.size; m = (M - 1) // 2
direct = np.convolve(spec, k, mode="same")                    # zero boundary
nfft = next_fast_len(spec.size + M - 1)
fft = np.fft.irfft(np.fft.rfft(spec, nfft) * np.fft.rfft(k, nfft), nfft)[m:m + spec.size]
padded = np.pad(spec, m, mode="reflect")
refl = np.convolve(padded, k, mode="valid")
print("1. SRF convolution (FWHM 10 nm), 2101 samples, kernel M = 61")
print(f"   FFT vs np.convolve (both zero boundary): max diff {np.abs(fft - direct).max():.1e}")
print(f"   zero vs reflect boundary: differ by {np.abs(direct - refl)[:m].max():.3f} in the first {m} nm,"
      f" {np.abs(direct - refl)[m:-m].max():.1e} inside")
print(f"   O2-A dip: {1 - spec[360] / np.interp(760, [740, 780], spec[[340, 380]]):.2f} at 1 nm,"
      f" {1 - refl[360] / np.interp(760, [740, 780], refl[[340, 380]]):.2f} after the SRF")

# ---------------------------------------------------------------- 2. spatial PSF and deconvolution
b = band(d["wavelengths"], 860)
x = d["cube"][..., b].astype(float)
H, W = x.shape
fy, fx = np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :]
sig_psf = 1.5
Hk = np.exp(-2 * np.pi ** 2 * sig_psf ** 2 * (fx ** 2 + fy ** 2))     # transform of a Gaussian PSF
rng = np.random.default_rng(1)
noise = 0.003
y = np.real(np.fft.ifft2(np.fft.fft2(x) * Hk)) + noise * rng.standard_normal(x.shape)
Y = np.fft.fft2(y)
inv = np.real(np.fft.ifft2(Y / Hk))
lams = 10.0 ** np.arange(-8, 0.01, 0.25)
errs = [rms(np.real(np.fft.ifft2(np.conj(Hk) * Y / (np.abs(Hk) ** 2 + l))) - x) for l in lams]
# Wiener: radially averaged |Y|^2, noise floor from the outer frequencies, cutoff where the signal sinks
P = np.abs(Y) ** 2 / (H * W)
r = np.sqrt(fx ** 2 + fy ** 2)
edges = np.linspace(0, r.max(), 40)
idx = np.digitize(r, edges) - 1
Prad = np.array([P[idx == i].mean() for i in range(len(edges) - 1)])
Sn = np.median(P[r > 0.4]) / np.log(2)                 # median of an exponential = ln2 x mean
rc = edges[np.argmax(Prad < 2 * Sn)]
Phi = np.clip((np.interp(r, 0.5 * (edges[1:] + edges[:-1]), Prad) - Sn)
              / np.interp(r, 0.5 * (edges[1:] + edges[:-1]), Prad), 0, 1) * (r < rc)
wien = np.real(np.fft.ifft2(Phi * Y / Hk))
print("\n2. 860-nm band, Gaussian PSF sigma 1.5 px, noise 0.003")
print(f"   rms error: blurred {rms(y - x):.4f}, plain inverse {rms(inv - x):.1e},"
      f" best Tikhonov {min(errs):.4f} (lambda 1e{np.log10(lams[int(np.argmin(errs))]):.2f}),"
      f" Wiener from the data {rms(wien - x):.4f}")
print(f"   noise floor read from |Y|^2: {Sn:.2e} (true {noise ** 2:.2e}); cutoff at {rc:.2f} cycles/px")

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].plot(lam, spec, "k", lw=0.8, label="1 nm"); ax[0].plot(lam, refl, "C3", label="SRF, FWHM 10 nm")
ax[0].set(xlim=(680, 820), xlabel="wavelength [nm]", title="spectral response"); ax[0].legend(fontsize=8)
lo, hi = np.percentile(x, (1, 99))
for a, im, t in zip(ax[1:3], (y, wien), ("blurred + noise", "Wiener from the data")):
    a.imshow(im, cmap="gray", vmin=lo, vmax=hi); a.set_title(t)
ax[3].loglog(lams, errs, "o-", ms=3, label="Tikhonov"); ax[3].axhline(rms(wien - x), color="C2", label="Wiener")
ax[3].axhline(rms(y - x), color="k", ls=":", label="blurred"); ax[3].set(xlabel="lambda", ylabel="rms error"); ax[3].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "L09_fig1.png"), dpi=120)
print("saved L09_fig1.png")
