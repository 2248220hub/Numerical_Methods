"""Lecture 11, exercise 3 (reference solution): a mask is a window.

Sky C_l = B_l^2 / l^2 (Gaussian beam FWHM = 1 deg: a spectrum that falls by 10^-4 over the band),
Nside = 128, l_max = 3 Nside - 1, 50 realisations.
Masks: binary |b| > 20 deg; apodised = binary shrunk to |b| > 25 deg, smoothed with a 5 deg Gaussian.
 1. power spectrum of the two masks (anafast of w): how fast it falls with l;
 2. naive estimate anafast(w m) / <w^2> divided by the true C_l, in bins of 10, for full sky,
    binary and apodised masks: leakage from large to small scales;
 3. the price: effective sky fraction <w^2>^2/<w^4> (number of modes) for the two masks.
Output: cut_sky.out, cut_sky.png
"""
import numpy as np
import healpy as hp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NSIDE, NREAL, DL = 128, 50, 10
LMAX = 3 * NSIDE - 1

if __name__ == "__main__":
    np.random.seed(12)
    ell = np.arange(LMAX + 1)
    bl = hp.gauss_beam(np.radians(1.0), lmax=LMAX)
    cl = np.zeros(LMAX + 1); cl[2:] = bl[2:]**2 / ell[2:]**2
    theta, _ = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    b = np.degrees(np.pi / 2 - theta)
    masks = {"full": np.ones_like(b),
             "binary": (np.abs(b) > 20).astype(float),
             "apodised": np.clip(hp.smoothing((np.abs(b) > 25).astype(float), fwhm=np.radians(5)), 0, 1)}
    print(f"Nside = {NSIDE}, l_max = {LMAX}, sky C_l = B_l^2/l^2 (FWHM 1 deg), {NREAL} realisations")
    print(f"C_l falls by {cl[2]/cl[300]:.1e} from l = 2 to l = 300")
    for k, w in masks.items():
        if k == "full": continue
        print(f"mask {k:9s}: f_sky = <w> = {w.mean():.3f}, <w^2> = {np.mean(w**2):.3f},"
              f" effective f_sky <w^2>^2/<w^4> = {np.mean(w**2)**2/np.mean(w**4):.3f}")
    wl = {k: hp.anafast(w, lmax=LMAX) for k, w in masks.items() if k != "full"}
    print("mask power spectrum W_l (even l; odd l vanish for this symmetric mask):")
    for L in (10, 50, 100, 200, 300):
        print(f"  l = {L:3d}: binary {wl['binary'][L]:.1e}, apodised {wl['apodised'][L]:.1e}")
    est = {k: np.zeros(LMAX + 1) for k in masks}
    for _ in range(NREAL):
        m = hp.synfast(cl, NSIDE, lmax=LMAX, new=True)
        for k, w in masks.items():
            est[k] += hp.anafast(w * m, lmax=LMAX) / np.mean(w**2) / NREAL
    print(f"<C_l-hat>/C_l in bins of {DL} (naive estimate: anafast(w m)/<w^2>)")
    print("   bin        full   binary  apodised")
    for l0 in (2, 10, 20, 50, 100, 150, 200, 250, 300, 350):
        sl = slice(l0, l0 + DL)
        r = [np.mean(est[k][sl]) / np.mean(cl[sl]) for k in masks]
        print(f"  {l0:3d}-{l0+DL-1:3d}   {r[0]:6.3f}  {r[1]:7.3f}  {r[2]:7.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k in ("binary", "apodised"):
        ax[0].semilogy(ell[2::2], wl[k][2::2], label=k)
    ax[0].set_xlabel(r"$\ell$"); ax[0].set_ylabel(r"mask power $W_\ell$"); ax[0].legend(fontsize=8)
    for k in masks:
        ax[1].semilogy(ell[2:], est[k][2:], lw=1, label=k)
    ax[1].semilogy(ell[2:], cl[2:], "k--", lw=1, label=r"true $C_\ell$")
    ax[1].set_xlabel(r"$\ell$"); ax[1].set_ylabel(r"$\langle\hat C_\ell\rangle$"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("cut_sky.png", dpi=150)
