"""Lecture 11, exercise 2 (reference solution): cosmic variance on a full and on a cut sky.

Sky C_l = 1/(l(l+1)), Nside = 64, l_max = 128, 500 Gaussian realisations (healpy.synfast).
Mask: Galactic band |b| < 20 deg removed (f_sky = 0.66). Naive cut-sky estimate: anafast of the
masked map divided by f_sky.
 1. full sky: mean and scatter of C_l-hat / C_l against 1 and sqrt(2/(2l+1));
    (2l+1) C_2-hat / C_2 against chi^2_5 (Kolmogorov-Smirnov).
 2. cut sky: the same; scatter per multipole and in bins of 10 against sqrt(2/((2l+1) f_sky dl)).
 3. correlation between C_l-hat and C_{l+1}, C_{l+2}, C_{l+3}-hat: zero on the full sky; on the cut sky
    only l+2 (the band mask is symmetric about the equator: it couples multipoles of equal parity).
Output: cosmic_variance.out, cosmic_variance.png
"""
import numpy as np
import healpy as hp
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NSIDE, LMAX, NREAL, DL = 64, 128, 500, 10
rng_seed = 11

if __name__ == "__main__":
    np.random.seed(rng_seed)
    ell = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1); cl[2:] = 1.0 / (ell[2:] * (ell[2:] + 1.0))
    theta, _ = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    mask = (np.abs(np.degrees(np.pi / 2 - theta)) > 20).astype(float)
    fsky = mask.mean()
    full = np.empty((NREAL, LMAX + 1)); cut = np.empty((NREAL, LMAX + 1))
    for i in range(NREAL):
        m = hp.synfast(cl, NSIDE, lmax=LMAX, new=True)
        full[i] = hp.anafast(m, lmax=LMAX)
        cut[i] = hp.anafast(m * mask, lmax=LMAX) / fsky
    full[:, 2:] /= cl[2:]; cut[:, 2:] /= cl[2:]
    print(f"Nside = {NSIDE}, l_max = {LMAX}, {NREAL} realisations; mask |b| < 20 deg, f_sky = {fsky:.3f}")
    print(" l   <C>/C full  cut   std full (theory)   std cut (theory/sqrt(f_sky))")
    for L in (2, 3, 5, 10, 30, 100):
        th = np.sqrt(2 / (2 * L + 1))
        print(f"{L:3d}   {full[:, L].mean():6.3f} {cut[:, L].mean():6.3f}   {full[:, L].std():6.3f} ({th:5.3f})"
              f"       {cut[:, L].std():6.3f} ({th/np.sqrt(fsky):5.3f})")
    p = stats.kstest(5 * full[:, 2], "chi2", args=(5,)).pvalue
    print(f"(2l+1) C_2-hat / C_2 against chi^2_5: KS p-value = {p:.2f}")
    print(f"bins of {DL}: std of the bin mean")
    for l0 in (20, 50, 100):
        lc = l0 + (DL - 1) / 2; th = np.sqrt(2 / ((2 * lc + 1) * DL))
        print(f"  l = {l0}-{l0+DL-1}: full {full[:, l0:l0+DL].mean(1).std():.4f} ({th:.4f}),"
              f" cut {cut[:, l0:l0+DL].mean(1).std():.4f} ({th/np.sqrt(fsky):.4f})")
    print("correlation of C_l-hat with C_{l+1}-hat and C_{l+2}-hat, averaged over l = 10-100"
          f" (noise of one coefficient ~ 1/sqrt({NREAL}) = {1/np.sqrt(NREAL):.3f})")
    for name, a in (("full", full), ("cut", cut)):
        r = [np.mean([np.corrcoef(a[:, L], a[:, L + d])[0, 1] for L in range(10, 101)]) for d in (1, 2, 3)]
        print(f"  {name:4s}: l+1 {r[0]:+.3f}, l+2 {r[1]:+.3f}, l+3 {r[2]:+.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ell[2:], full[:, 2:].std(0), ".", label="full sky")
    ax[0].plot(ell[2:], cut[:, 2:].std(0), ".", label=f"cut sky, f_sky = {fsky:.2f}")
    ax[0].plot(ell[2:], np.sqrt(2 / (2 * ell[2:] + 1)), "k-", lw=1, label=r"$\sqrt{2/(2\ell+1)}$")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel(r"$\ell$")
    ax[0].set_ylabel(r"std of $\hat C_\ell/C_\ell$"); ax[0].legend(fontsize=8)
    x = np.linspace(0, 20, 200)
    ax[1].hist(5 * full[:, 2], bins=40, density=True, alpha=0.6, label=r"$5\hat C_2/C_2$, full sky")
    ax[1].plot(x, stats.chi2.pdf(x, 5), "k-", label=r"$\chi^2_5$")
    ax[1].set_xlabel("value"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("cosmic_variance.png", dpi=150)
