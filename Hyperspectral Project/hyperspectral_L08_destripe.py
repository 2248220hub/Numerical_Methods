"""
hyperspectral_L08_destripe.py  --  Lecture 8 (2D FFT, notch filters) applied to a push-broom band.

A push-broom imager has one detector per image COLUMN. Unequal gains/offsets between detectors
give vertical stripes: a pattern that depends on x only, constant along y.
In the 2D spectrum a function of x only lives on the line v = 0 (the u axis): a notch along that
line (keeping the lowest |u|, which carry real large-scale structure) removes the stripes.

  Part 1  add column offsets (random + a periodic 8-column module pattern) to the 650-nm band
  Part 2  2D FFT: where does the stripe energy go?
  Part 3  three destripers: notch on v = 0, column-mean subtraction, and their rms errors
  Part 4  the same on every band: rms error before/after, worst band

Run:  python hyperspectral_L08_destripe.py         Figures: figures/L08_fig1.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from hsi_io import load, band, FIG

d = load("synthetic")
cube, wl = d["cube"].astype(float), d["wavelengths"]
H, W, B = cube.shape
rng = np.random.default_rng(3)
cols = np.arange(W)
stripe = 0.015 * rng.standard_normal(W) + 0.02 * np.cos(2 * np.pi * cols / 8)   # per-column offset


def notch_v0(img, keep=8, width=0.3):
    """Suppress the v = 0 line of the 2D spectrum except |u| <= keep (Gaussian in v, both signs)."""
    F = np.fft.fft2(img)
    v = np.fft.fftfreq(img.shape[0]) * img.shape[0]          # cycles per image, rows
    u = np.fft.fftfreq(img.shape[1]) * img.shape[1]          # columns
    Hn = 1 - np.exp(-v[:, None] ** 2 / (2 * width ** 2)) * (np.abs(u[None, :]) > keep)
    return np.real(np.fft.ifft2(F * Hn))


def notch_periodic(img, period=8, width=0.3):
    """Remove only the periodic module pattern: spikes at u = +-k W/period on the v = 0 line."""
    F = np.fft.fft2(img)
    v = np.fft.fftfreq(img.shape[0]) * img.shape[0]
    u = np.fft.fftfreq(img.shape[1]) * img.shape[1]
    Hn = np.ones(F.shape)
    for u0 in np.arange(1, period) * img.shape[1] / period:
        for s in (u0, -u0):
            Hn *= 1 - np.exp(-(v[:, None] ** 2 + (u[None, :] - s) ** 2) / (2 * width ** 2))
    return np.real(np.fft.ifft2(F * Hn))


def column_mean(img, smooth=15):
    """Classic destriper: subtract each column's mean minus a smoothed version of the column means."""
    m = img.mean(0)
    return img - (m - uniform_filter1d(m, smooth, mode="reflect"))[None, :]


rms = lambda a: np.sqrt(np.mean(a ** 2))
b = band(wl, 650)
clean = cube[..., b]
striped = clean + stripe[None, :]
F = np.fft.fftshift(np.abs(np.fft.fft2(striped - striped.mean())))
F0 = np.fft.fftshift(np.abs(np.fft.fft2(clean - clean.mean())))
e_line = (F[H // 2, :] ** 2).sum() / (F ** 2).sum()
e_line0 = (F0[H // 2, :] ** 2).sum() / (F0 ** 2).sum()
print("1-2. 650-nm band with column stripes")
print(f"   fraction of spectral power on the v = 0 line: clean {e_line0:.3f}, striped {e_line:.3f}")
outs = {"striped": striped, "notch v=0, |u|>8": notch_v0(striped), "column mean": column_mean(striped),
        "periodic notch": notch_periodic(striped)}
print("3. rms error against the clean band (noise sigma there "
      f"{d['noise_sigma'][b]:.4f}, stripe rms {rms(stripe):.4f})")
for k, v in outs.items():
    print(f"   {k:16s}: {rms(v - clean):.4f}")

print("   the v = 0 notch and the column-mean destriper are the same operation (one in Fourier space):")
print("   both remove every column pattern above |u| ~ 8; the periodic notch removes only the module pattern.")
print("4. every band")
before = np.array([rms(stripe) for _ in range(B)])
after_n = np.array([rms(notch_v0(cube[..., j] + stripe[None, :]) - cube[..., j]) for j in range(B)])
after_c = np.array([rms(column_mean(cube[..., j] + stripe[None, :]) - cube[..., j]) for j in range(B)])
print(f"   notch: median {np.median(after_n):.4f}, worst {after_n.max():.4f} at {wl[after_n.argmax()]:.0f} nm;"
      f"  column mean: median {np.median(after_c):.4f}")

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
lo, hi = np.percentile(clean, (1, 99))
ax[0].imshow(striped, cmap="gray", vmin=lo, vmax=hi); ax[0].set_title("650 nm, striped")
ax[1].imshow(np.log10(F + 1e-6), cmap="magma", extent=(-W / 2, W / 2, -H / 2, H / 2))
ax[1].set_title("log |F|: stripes on the v = 0 line"); ax[1].set_xlabel("u"); ax[1].set_ylabel("v")
ax[2].imshow(outs["notch v=0, |u|>8"], cmap="gray", vmin=lo, vmax=hi); ax[2].set_title("after the notch")
ax[3].plot(wl, after_n, label="notch"); ax[3].plot(wl, after_c, label="column mean")
ax[3].plot(wl, d["noise_sigma"], "k:", label="sensor noise"); ax[3].axhline(rms(stripe), color="r", ls="--", label="stripes")
ax[3].set(xlabel="wavelength [nm]", ylabel="rms error", yscale="log", title="every band"); ax[3].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "L08_fig1.png"), dpi=120)
print("saved L08_fig1.png")
