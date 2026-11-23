"""
Laboratory 2 -- Planck maps: the angular power spectrum on the sphere.
Example code: the whole laboratory, written the way a student would write it, with every
step explained: the core (steps 0-4) and the three extensions
  A. the same analysis at 100 GHz        (set FREQ = 100)
  B. what is left after the correction   (fit of a white sky component A)
  C. the cross-spectrum 70 x 100         (set CROSS = True; it reads both maps)

What you need
-------------
* Python 3 with numpy, matplotlib and healpy (pip install healpy; Linux, macOS or WSL).
* A folder data/ next to this script with the files of the laboratory:
      planck_T070_n1024_uK.fits   temperature map at 70 GHz, Nside 1024, RING, micro-K
      planck_T100_n1024_uK.fits   the same at 100 GHz (for the extension)
      mask_UT78_n1024.fits        the Planck common mask: 1 = keep, 0 = cut
      COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt
                                  the best-fit model D_l of Planck 2018
      pixel_window_n1024.fits     the HEALPix pixel window of Nside 1024

Run:   python3 lab2_starter.py
It prints the numbers of each step and opens (and saves) the figures.
Each spherical transform (anafast) takes about 20 s at l_max = 3071.
"""
import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
FREQ = 70                                   # 70 (core) or 100 (extension A)
CROSS = True                                # extension C: cross-spectrum 70 x 100
FWHM_ARCMIN = {70: 13.08, 100: 9.66}        # effective beam FWHM, Planck 2018 I, Table 4
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def path(name):
    """Full path of a file in data/."""
    return os.path.join(DATA, name)


# ===========================================================================
# STEP 0 -- read the map and the mask, and look at them
# ===========================================================================
# hp.read_map reads the first column of a HEALPix FITS file and returns a numpy
# array with one value per pixel, in RING order (the default of healpy).
# The files are stored in single precision: convert to float64 before doing sums.
m = hp.read_map(path(f"planck_T{FREQ:03d}_n1024_uK.fits")).astype(np.float64)
mask = hp.read_map(path("mask_UT78_n1024.fits")).astype(np.float64)

nside = hp.npix2nside(m.size)               # Npix = 12 Nside^2  ->  Nside
print(f"Step 0: {FREQ} GHz map, Nside = {nside}, Npix = {m.size}, "
      f"pixel side = {hp.nside2resol(nside, arcmin=True):.2f} arcmin")

# Unobserved pixels are stored as hp.UNSEEN = -1.6375e30. In single precision the
# value is not exactly hp.UNSEEN any more, so test with "< -1e30", not with "==".
bad = m < -1e30
print(f"        unobserved pixels: {bad.sum()}")
if bad.any():
    # (this happens at 100 GHz: one pixel, inside the kept sky)
    # A huge negative number would ruin every spectrum: remove the pixel from the
    # mask and give it a harmless value in the full-sky map.
    mask[bad] = 0.0
    m[bad] = 0.0

good = ~bad
print(f"        mean = {m[good].mean():.1f} muK, rms = {m[good].std():.1f} muK, "
      f"max = {m[good].max():.3g} muK")

# f_sky: the fraction of the sky the mask keeps (the mean of a 0/1 map).
fsky = mask.mean()
print(f"        f_sky = {fsky:.4f}")

# Mollweide projections. The colour scale is cut at +-300 muK: with the full range
# the Galaxy (up to 1e5 muK) saturates the colours and the CMB is invisible.
plt.figure(figsize=(11, 3.6))
hp.mollview(m, sub=(1, 2, 1), min=-300, max=300, unit=r"$\mu$K", cmap="RdBu_r",
            title=f"Planck {FREQ} GHz", hold=True)
masked = m.copy()
masked[mask == 0] = hp.UNSEEN               # UNSEEN pixels are drawn in grey
hp.mollview(masked, sub=(1, 2, 2), min=-300, max=300, unit=r"$\mu$K", cmap="RdBu_r",
            title=f"with the mask (f_sky = {fsky:.2f})", hold=True)
plt.savefig(f"step0_maps_{FREQ}.png", dpi=120, bbox_inches="tight")


# ===========================================================================
# STEP 1 -- the angular power spectrum of the whole sky
# ===========================================================================
# anafast computes the spherical-harmonic coefficients a_lm of the map and returns
#     C_l = 1/(2l+1) sum_m |a_lm|^2 ,   l = 0 ... lmax.
# With Nside 1024 the largest multipole the grid can represent is about 3 Nside.
lmax = 3 * nside - 1
ell = np.arange(lmax + 1)


def Dl(cl):
    """D_l = l(l+1) C_l / 2 pi: the usual way to plot a CMB spectrum (flat for a
    scale-invariant sky). Units: muK^2."""
    return ell * (ell + 1) * cl / (2 * np.pi)


print("Step 1: anafast on the full sky ...")
cl_full = hp.anafast(m, lmax=lmax)

# The Planck best-fit model: a text file with columns  l, D_l(TT), D_l(TE), ...
# It starts at l = 2 and stops at l = 2508.
th = np.loadtxt(path("COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-"
                     "minimum-theory_R3.01.txt"))
Dth = np.zeros(lmax + 1)
Dth[2:2 + len(th)] = th[:, 1]               # D_l of the model on our l grid (0 beyond 2508)
has_model = Dth > 0

for L in (10, 220, 800, 2000):
    print(f"        l = {L:4d}: D_l(full sky) = {Dl(cl_full)[L]:9.4g} muK^2, "
          f"model = {Dth[L]:7.4g} muK^2")
# What you see: on the full sky D_l is 10 times the model at the first peak
# (l = 220): the Galaxy dominates. At high l, D_l grows like l^2: a flat C_l, i.e. noise.


# ===========================================================================
# STEP 2 -- cut the Galaxy with the mask
# ===========================================================================
# 1. Monopole and dipole: a constant (l = 0) or a dipole (l = 1) multiplied by the
#    mask is no longer a pure l = 0 or 1: it leaks into the multipoles of the mask.
#    So we fit and remove them on the KEPT sky only. fit_dipole and remove_dipole
#    ignore the pixels set to UNSEEN, so we hide the cut pixels first.
mm = m.copy()
mm[mask == 0] = hp.UNSEEN
mono, dip = hp.fit_dipole(mm)
print(f"Step 2: on the kept sky: monopole = {mono:.1f} muK, |dipole| = {np.linalg.norm(dip):.1f} muK")
mm = hp.remove_dipole(mm)                   # removes monopole and dipole

# 2. Set the cut pixels to zero (anafast must see numbers, not UNSEEN) and compute
#    the spectrum of the masked map.
mm[mask == 0] = 0.0
print("        anafast on the masked sky ...")
cl_mask = hp.anafast(mm, lmax=lmax)

# 3. A map multiplied by a 0/1 mask keeps, on average, a fraction f_sky of the power:
#    divide by f_sky. (This is the "pseudo-C_l": the mask also mixes neighbouring l,
#    which we neglect here.)
cl_mask /= fsky

for L in (2, 10, 220, 800, 2000):
    print(f"        l = {L:4d}: D_l(masked) = {Dl(cl_mask)[L]:9.4g} muK^2, model = {Dth[L]:7.4g}")
# Now the shape follows the model at low l, but it is lower and lower than the model
# as l grows (the beam, step 4), and it rises again at high l (the noise, step 3).


# ===========================================================================
# STEP 3 -- the noise bias
# ===========================================================================
# The instrument noise is (nearly) white and uncorrelated between pixels: its C_l is a
# constant N, added to the sky:  <C_l> = (sky) + N.  At high l the beam has removed
# the sky, so there C_l is just N. Take the mean over a range where the sky is gone.
hi = (ell >= 2000) & (ell <= 2500) if FREQ == 70 else (ell >= 2600) & (ell <= 3000)
N = cl_mask[hi].mean()
print(f"Step 3: white noise level N = {N:.4g} muK^2 sr (mean of C_l for l = "
      f"{ell[hi][0]}-{ell[hi][-1]})")

# Is the noise really white? If yes, C_l / N is about 1 in every band where the sky is gone.
for a, b in [(1200, 1500), (1500, 2000), (2000, 2500), (2500, 3000)]:
    print(f"        l = {a}-{b}: mean C_l / N = {cl_mask[a:b].mean() / N:.3f}")


# ===========================================================================
# STEP 4 -- the beam and the pixels
# ===========================================================================
# The map is the sky seen through the beam (B_l) and averaged over the pixels (W_l):
#     <C_l(map)> = B_l^2 W_l^2 C_l(sky) + N .
# B_l of a Gaussian beam: hp.gauss_beam wants the FWHM in RADIANS.
B = hp.gauss_beam(np.radians(FWHM_ARCMIN[FREQ] / 60.0), lmax=lmax)
# W_l: hp.pixwin(nside) would download it; here we read the official table from data/.
# read_cl returns two columns (temperature, polarisation): take the first.
W = np.atleast_2d(hp.read_cl(path("pixel_window_n1024.fits")))[0][:lmax + 1]
BW2 = (B * W) ** 2

for q in (0.5, 0.1, 0.01):
    print(f"Step 4: B_l^2 = {q} at l = {np.argmax(B ** 2 < q)}")

# Compare the data with the model seen by the instrument:  D_th B^2 W^2 + D(N).
model_seen = Dth * BW2 + Dl(np.full(lmax + 1, N))

# Correct the data: subtract the noise, divide by B^2 W^2. This is a DEconvolution:
# where B^2 W^2 is small, any error left after subtracting N is amplified enormously.
cl_cmb = (cl_mask - N) / BW2
D_cmb = Dl(cl_cmb)

# Average in bins of dl = 30 to reduce the scatter, and give each bin an error bar
# (Knox 1995): the relative error of a C_l estimated from (2l+1) f_sky dl modes is
# sqrt(2 / ((2l+1) f_sky dl)), applied to (signal + noise as seen after the correction).
dl = 30
edges = np.arange(2, lmax + 1, dl)
lc = 0.5 * (edges[:-1] + edges[1:] - 1)                 # centre of each bin
Db = np.array([D_cmb[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
Dthb = np.array([Dth[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
DNb = np.array([Dl(np.full(lmax + 1, N) / BW2)[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
err = np.sqrt(2.0 / ((2 * lc + 1) * fsky * dl)) * (Dthb + DNb)

for a, b in [(30, 300), (300, 600), (600, 800), (800, 1000), (1000, 1200)]:
    k = (lc >= a) & (lc < b)
    print(f"        l = {a}-{b}: recovered / model = {np.mean(Db[k] / Dthb[k]):.3f}")
# Up to l ~ 600 the recovered spectrum follows the model; beyond, B^2 is small and the
# correction amplifies what is left (other sky components, the beam shape, noise colour).


# ===========================================================================
# EXTENSION B -- what is left after the correction?
# ===========================================================================
# After subtracting N and dividing by B^2 W^2 the recovered spectrum is above the model,
# more and more as l grows. Look at the excess in C_l (not D_l): it is roughly constant.
# A constant C_l is white; but it cannot be the instrument noise (that was subtracted,
# and it is not seen through the beam). A white component that DID pass through the beam
# is on the sky: point-like objects at random positions (unresolved radio sources?).
# Model:   C_l(mask) - N = (C_l(model) + A) B^2 W^2 ,   fit the constant A.
Cth = np.zeros(lmax + 1)
Cth[2:] = 2 * np.pi * Dth[2:] / (ell[2:] * (ell[2:] + 1))   # model C_l from D_l
s = slice(30, 1200)                          # where the beam still leaves some signal
y = cl_mask[s] - N - Cth[s] * BW2[s]         # the excess, as the instrument sees it
x = BW2[s]                                   # the shape a white sky has after the beam
var = 2 * (Cth[s] * BW2[s] + N) ** 2 / ((2 * ell[s] + 1) * fsky)   # variance of each C_l
A = (x * y / var).sum() / (x * x / var).sum()                        # weighted least squares
chi_A = ((y - A * x) ** 2 / var).sum()
chi_0 = (y ** 2 / var).sum()
print(f"Ext. B: A = {A:.3g} muK^2 sr; chi^2 = {chi_0:.0f} without A, {chi_A:.0f} with A "
      f"({s.stop - s.start} multipoles)")
# Is a narrower beam a better explanation than A? Try Gaussian beams of other widths, no A.
best = (np.inf, None)
for fw in np.arange(0.85, 1.0001, 0.0025) * FWHM_ARCMIN[FREQ]:
    x2 = (hp.gauss_beam(np.radians(fw / 60), lmax=lmax) * W)[s] ** 2
    c2 = ((cl_mask[s] - N - Cth[s] * x2) ** 2 / var).sum()
    best = min(best, (c2, fw))
print(f"        alternative, no A and a free beam width: best chi^2 = {best[0]:.0f} "
      f"at FWHM = {best[1]:.2f} arcmin")
# At 70 GHz the white component wins clearly; at 100 GHz a narrower beam does as well:
# the data alone cannot always tell a sky component from an imperfect beam model.


# ===========================================================================
# Figures of steps 1-4
# ===========================================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].semilogy(ell[2:], Dl(cl_full)[2:], color="0.6", lw=0.6, label="full sky")
ax[0].semilogy(ell[2:], Dl(cl_mask)[2:], lw=0.6, label="masked, / f_sky")
ax[0].semilogy(ell[has_model], Dth[has_model], "k", lw=1.2, label="Planck best fit")
ax[0].semilogy(ell[has_model], model_seen[has_model], "--", lw=1.2,
               label=r"best fit $\times B_\ell^2W_\ell^2$ + noise")
ax[0].set_xlabel(r"$\ell$"); ax[0].set_ylabel(r"$D_\ell$ [$\mu$K$^2$]")
ax[0].set_ylim(10, 2e5); ax[0].legend(fontsize=8); ax[0].set_title(f"{FREQ} GHz")

ax[1].errorbar(lc, Db, yerr=err, fmt="o", ms=2, lw=0.8, label="(C_l - N) / (B^2 W^2), bins of 30")
ax[1].plot(ell[has_model], Dth[has_model], "k", lw=1.2, label="Planck best fit")
ax[1].set_xlim(0, 2000); ax[1].set_ylim(-1000, 8000)
ax[1].set_xlabel(r"$\ell$"); ax[1].set_ylabel(r"$D_\ell$ [$\mu$K$^2$]")
ax[1].legend(fontsize=8); ax[1].set_title("beam and noise corrected")
plt.tight_layout()
ax[1].plot(ell[has_model], (Dth + Dl(np.full(lmax + 1, A)))[has_model], "--", lw=1.0,
           label=f"best fit + white residual A = {A:.1e}")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"steps1-4_spectra_{FREQ}.png", dpi=120)


# ===========================================================================
# EXTENSION C -- the cross-spectrum 70 x 100
# ===========================================================================
# The noise of two different instruments is independent, the sky is the same. So the
# cross-spectrum  1/(2l+1) sum_m a_lm(70) a_lm(100)*  has NO noise bias:
#     <C_l(70 x 100)> = B_l(70) B_l(100) W_l^2 C_l(sky) .
# Both maps must be treated in the same way: same mask, monopole and dipole removed.
if CROSS:
    def masked_map(freq, mask):
        """Read a map, remove monopole and dipole on the kept sky, zero the cut pixels."""
        mp = hp.read_map(path(f"planck_T{freq:03d}_n1024_uK.fits")).astype(np.float64)
        mask = mask.copy()
        mask[mp < -1e30] = 0.0               # the unobserved pixel of the 100 GHz map
        mp[mask == 0] = hp.UNSEEN
        mp = hp.remove_dipole(mp)
        mp[mask == 0] = 0.0
        return mp, mask

    mask0 = hp.read_map(path("mask_UT78_n1024.fits")).astype(np.float64)
    m70, mask_c = masked_map(70, mask0)
    m100, mask_c = masked_map(100, mask_c)   # common mask: both cuts applied
    m70[mask_c == 0] = 0.0                   # the 100 GHz cut also on the 70 GHz map
    fsky_c = mask_c.mean()
    print("Ext. C: anafast of the cross-spectrum 70 x 100 ...")
    cl_x = hp.anafast(m70, map2=m100, lmax=lmax) / fsky_c
    B70 = hp.gauss_beam(np.radians(FWHM_ARCMIN[70] / 60), lmax=lmax)
    B100 = hp.gauss_beam(np.radians(FWHM_ARCMIN[100] / 60), lmax=lmax)
    D_x = Dl(cl_x / (B70 * B100 * W ** 2))   # no noise to subtract
    Dxb = np.array([D_x[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    for L in (226, 526, 796, 1006):
        k = np.argmin(abs(lc - L))
        print(f"        l ~ {lc[k]:.0f}: cross / model = {Dxb[k] / Dthb[k]:.3f}")
    # No noise bias, but the cross-spectrum is not the CMB alone: everything on the sky
    # (Galaxy residuals, sources) is in both maps and survives.
    plt.figure(figsize=(7, 4))
    plt.plot(lc, Db, "o", ms=2, label=f"{FREQ} GHz auto, corrected")
    plt.plot(lc, Dxb, "o", ms=2, label=r"70$\times$100 cross")
    plt.plot(ell[has_model], Dth[has_model], "k", lw=1.2, label="Planck best fit")
    plt.xlim(0, 2500); plt.ylim(-1000, 8000)
    plt.xlabel(r"$\ell$"); plt.ylabel(r"$D_\ell$ [$\mu$K$^2$]"); plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("extC_cross.png", dpi=120)

plt.show()
