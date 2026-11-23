"""
hyperspectral_L07_sampling.py  --  Lecture 7 (sampling, quantisation, aliasing) applied to spectra.

A hyperspectral sensor SAMPLES a continuous spectrum: each band integrates the light over its
spectral response function (SRF, ~Gaussian of width FWHM) centred every `spacing` nm.
That is exactly sampling + an anti-alias filter, along wavelength instead of time.

  Part 1  a 1-nm "true" spectrum (vegetation + narrow O2-A line at 760 nm + a 2200-nm mineral dip)
          observed by three sensors: AVIRIS-like (10 nm / FWHM 10), undersampled (30 nm / FWHM 4)
          and multispectral (100 nm / FWHM 60).
  Part 2  "sampling phase": shift the band grid by a fraction of the spacing. If FWHM << spacing
          the measured depth of a narrow line depends on where the bands fall: spectral aliasing.
  Part 3  reconstruction error (linear interpolation back to 1 nm) for each sensor.
  Part 4  quantisation: bits per band against the sensor noise of the synthetic cube.

Run:  python hyperspectral_L07_sampling.py        Figures: figures/L07_fig1.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hsi_io import load, FIG

# ---------------------------------------------------------------- 1. the true spectrum, 1 nm
lam = np.arange(400.0, 2501.0, 1.0)
g = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2)
sig = lambda x, x0, k: 1 / (1 + np.exp(-(x - x0) / k))
truth = ((0.04 + 0.06 * g(lam, 550, 35) + 0.46 * sig(lam, 715, 12) - 0.20 * sig(lam, 1300, 80)
          - 0.08 * sig(lam, 2000, 90)) * (1 - 0.55 * g(lam, 1450, 45) - 0.75 * g(lam, 1940, 60))
         * (1 - 0.25 * g(lam, 2200, 6)))           # narrow mineral feature
no_line = truth.copy()
truth = truth * (1 - 0.6 * g(lam, 760, 1.5))       # O2-A absorption: ~3.5 nm wide


def observe(centres, fwhm, spec=None):
    """Each band = SRF-weighted mean of the spectrum (a convolution, then sampling)."""
    spec = truth if spec is None else spec
    s = fwhm / 2.3548
    W = g(lam[None, :], centres[:, None], s)
    return (W * spec).sum(1) / W.sum(1)


def line_depth(centres, fwhm):
    """Largest fractional dip that the O2 line causes in any band: 1 - band / same band without the line."""
    return np.max(1 - observe(centres, fwhm) / observe(centres, fwhm, no_line))


sensors = {"AVIRIS-like  10 nm, FWHM 10": (10.0, 10.0),
           "undersampled 30 nm, FWHM 4 ": (30.0, 4.0),
           "multispectral 100 nm, FWHM 60": (100.0, 60.0)}

print("1-3. three sensors looking at the same 1-nm spectrum")
print(f"   true O2-A depth at 760 nm: {np.max(1 - truth / no_line):.3f}")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(lam, truth, "k", lw=0.8, label="truth, 1 nm")
for (name, (sp, fw)), col in zip(sensors.items(), ["C0", "C3", "C2"]):
    depths = []
    for phase in np.linspace(0, 1, 11)[:-1]:          # shift the band grid
        c = np.arange(400 + phase * sp, 2500, sp)
        depths.append(line_depth(c, fw))
    c = np.arange(400, 2500, sp); v = observe(c, fw)
    rec = np.interp(lam, c, v)
    rms = np.sqrt(np.mean((rec - truth)[(lam > 450) & (lam < 2450)] ** 2))
    print(f"   {name}: O2 depth {min(depths):.3f} .. {max(depths):.3f} over 10 grid phases,"
          f"  reconstruction rms {rms:.4f}")
    ax[0].plot(c, v, "o-", ms=3, lw=1, color=col, label=name)
    ax[1].plot(np.linspace(0, 1, 11)[:-1], depths, "o-", color=col, label=name)
ax[0].set(xlabel="wavelength [nm]", ylabel="reflectance", xlim=(650, 900), title="red edge and O2-A line")
ax[0].legend(fontsize=7)
ax[1].axhline(np.max(1 - truth / no_line), color="k", ls="--", label="true depth")
ax[1].set(xlabel="band-grid shift [fraction of the spacing]", ylabel="measured O2-A depth",
          title="spectral aliasing: the answer depends on where the bands fall")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "L07_fig1.png"), dpi=120)

# ---------------------------------------------------------------- 4. quantisation vs sensor noise
d = load("synthetic")
sigma_min = d["noise_sigma"].min()
print("\n4. quantisation of reflectance in [0, 1] with B bits: rms error Delta/sqrt(12)")
for B in (6, 8, 10, 12, 14):
    delta = 1 / 2 ** B
    q = delta * np.round(d["cube"][..., 30] / delta)
    meas = np.std(q - d["cube"][..., 30])
    print(f"   B = {B:2d}: Delta/sqrt(12) = {delta / np.sqrt(12):.2e}  (measured {meas:.2e});"
          f"  vs smallest sensor noise {sigma_min:.1e}: {'negligible' if delta / np.sqrt(12) < 0.3 * sigma_min else 'NOT negligible'}")
print("   Real sensors digitise RADIANCE with 12-16 bits: bright snow and dark water must both fit.")
print("saved L07_fig1.png")
