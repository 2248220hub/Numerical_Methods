"""
make_synthetic_cube.py  --  a small hyperspectral scene with KNOWN truth.

Why synthetic first?  Lectures 12-13 keep repeating one rule: test your inference
on simulated data where the truth is known (coverage, bias, error bars).
Real cubes (Indian Pines, Pavia) come next, with get_hyperspectral_data.py.

Physics (the linear mixing model):
    y(pixel, band) = sum_k  a_k(pixel) * E_k(band)  +  n(band),   n ~ N(0, sigma_band^2)
    a_k >= 0, sum_k a_k = 1   (fractions of the pixel covered by each material)

Output:  data/synthetic_cube.npz  with
    cube (H, W, B) float32 reflectance      wavelengths (B,) nm
    endmembers (K, B)   names (K,)          abundances (H, W, K)   labels (H, W)
    noise_sigma (B,)
Run:  python make_synthetic_cube.py
"""
import os
import numpy as np
from scipy.ndimage import gaussian_filter

H, W, B = 128, 128, 120
rng = np.random.default_rng(42)
wl = np.linspace(400, 2500, B)                       # nm


def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-(x - x0) / k))


# ---- endmember spectra (smooth caricatures of real library spectra)
water_dips = 1 - 0.55 * gauss(wl, 1450, 45) - 0.75 * gauss(wl, 1940, 60)
vegetation = (0.04 + 0.06 * gauss(wl, 550, 35) + 0.46 * sigmoid(wl, 715, 12)
              - 0.20 * sigmoid(wl, 1300, 80) - 0.08 * sigmoid(wl, 2000, 90)) * water_dips
soil = (0.08 + 0.30 * sigmoid(wl, 900, 260) - 0.05 * gauss(wl, 2200, 40)) \
       * (1 - 0.25 * gauss(wl, 1450, 45) - 0.35 * gauss(wl, 1940, 60))
water = 0.07 * np.exp(-(wl - 400) / 180) + 0.005
roof = (0.28 + 0.04 * (wl - 400) / 2100 - 0.04 * gauss(wl, 1730, 30)) \
       * (1 - 0.15 * gauss(wl, 1450, 45) - 0.2 * gauss(wl, 1940, 60))
E = np.vstack([vegetation, soil, water, roof])
names = np.array(["vegetation", "soil", "water", "roof"])
K = len(names)

# ---- abundance maps: smooth random fields -> softmax (mostly pure, mixed at borders)
fields = np.stack([gaussian_filter(rng.standard_normal((H, W)), 9) for _ in range(K)], -1)
fields /= fields.std(axis=(0, 1))
yy, xx = np.mgrid[0:H, 0:W]
fields[..., 0] += 0.6                                           # a bit more vegetation
fields[..., 2] += 5.0 * gauss(xx - 0.25 * yy, 70, 4) - 2.0      # a river
fields[..., 3] += 4.0 * ((np.abs(xx - 28) < 12) & (np.abs(yy - 30) < 12)) \
                 + 3.0 * ((np.abs(xx - 100) < 8) & (np.abs(yy - 105) < 14)) - 1.2   # two buildings
z = 2.2 * fields
abund = np.exp(z - z.max(-1, keepdims=True))
abund /= abund.sum(-1, keepdims=True)
labels = abund.argmax(-1)

# ---- noise: larger in the atmospheric water-vapour bands and at the band edges
sigma = 0.004 + 0.03 * gauss(wl, 1400, 40) + 0.04 * gauss(wl, 1900, 50) + 0.01 * (wl > 2350)
cube = abund @ E + rng.standard_normal((H, W, B)) * sigma

os.makedirs("data", exist_ok=True)
np.savez_compressed("data/synthetic_cube.npz", cube=cube.astype(np.float32), wavelengths=wl,
                    endmembers=E, names=names, abundances=abund.astype(np.float32),
                    labels=labels, noise_sigma=sigma)
print(f"saved data/synthetic_cube.npz : cube {cube.shape}, classes {dict(zip(names, np.bincount(labels.ravel())))}")
