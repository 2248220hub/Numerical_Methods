"""Laboratory 2 (Sampling on the sphere): reference solution. The angular power spectrum of Planck maps.

Input (../data/, written by prepare_data.py; Nside 1024, RING, muK_CMB):
  planck_T070_n1024_uK.fits, planck_T100_n1024_uK.fits (optional: extension), mask_UT78_n1024.fits,
  COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt (best-fit D_l),
  pixel_window_n1024.fits (official HEALPix table from healpy-data; otherwise
  pixel_window_n1024_claude.fits from pixel_window.py; otherwise hp.pixwin).
Steps (for each map):
  1. full-sky spectrum (anafast, l <= 3 Nside - 1);
  2. masked spectrum: monopole and dipole removed on the unmasked sky, anafast(m * mask) / f_sky;
  3. beam and pixel window: Gaussian beam of the effective FWHM (Planck 2018 I, Table 4:
     70 GHz 13.08', 100 GHz 9.66'), pixel window of Nside 1024; model = theory * B^2 W^2;
  4. noise: white-noise level N from the masked spectrum where the sky is negligible
     (l = 2000-2500 at 70 GHz); recovered C_l = (C_l^mask - N) / (B^2 W^2), binned in dl = 30,
     error bars sqrt(2 / ((2l+1) f_sky dl)) (C_th + N / (B^2 W^2));
     4b. the residual after this correction grows as 1/C_th: fit a white sky component A
     seen through the beam, C_l^mask - N = (C_th + A) B^2 W^2 (unresolved sources: an interpretation);
  5. (extension, 100 GHz) the same; cross-spectrum 70 x 100: anafast(m70, map2=m100), no noise bias,
     divided by B70 B100 W^2.
Output: lab2_reference.out and ../Figures/fig_lab2_*.png.
"""
import os, sys, pathlib, time
import numpy as np
import healpy as hp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import figstyle                                                         # noqa: E402
figstyle.theme()
FIG = HERE.parent / "Figures"
DATA = HERE.parent / "data"
NSIDE = 1024
LMAX = 3 * NSIDE - 1
FWHM = {70: 13.08, 100: 9.66}                    # arcmin, Planck 2018 I, Table 4
ell = np.arange(LMAX + 1)
D = lambda c: ell * (ell + 1) * c / (2 * np.pi)


def pixel_window():
    for name, src in (("pixel_window_n1024.fits", "official HEALPix table"),
                      ("pixel_window_n1024_claude.fits", "pixel_window.py")):
        f = DATA / name
        if f.exists():
            return np.atleast_2d(hp.read_cl(str(f)))[0][: LMAX + 1], f"{name} ({src})"
    return hp.pixwin(NSIDE, lmax=LMAX), "hp.pixwin"


def theory():
    th = np.loadtxt(DATA / "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt")
    Dth = np.zeros(LMAX + 1)
    n = min(LMAX + 1, int(th[-1, 0]) + 1)
    Dth[2:n] = th[: n - 2, 1]
    return Dth


def binned(x, dl=30, lmin=2):
    edges = np.arange(lmin, LMAX + 1, dl)
    lc = np.array([0.5 * (a + b - 1) for a, b in zip(edges[:-1], edges[1:])])
    return lc, np.array([x[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])


def spectra(freq, m, mask, W, Dth, out):
    fsky = mask.mean()
    t = time.time()
    cl_full = hp.anafast(m, lmax=LMAX)
    mono_f, dip_f = hp.fit_dipole(m)
    raw = m * mask                                       # masked, monopole and dipole kept
    cl_raw = hp.anafast(raw, lmax=LMAX) / fsky
    mm = m.copy(); mm[mask == 0] = hp.UNSEEN
    mono, dip = hp.fit_dipole(mm)
    mm = hp.remove_dipole(mm)
    mm[mask == 0] = 0.0
    cl_mask = hp.anafast(mm, lmax=LMAX) / fsky
    B = hp.gauss_beam(np.radians(FWHM[freq] / 60), lmax=LMAX)
    BW2 = (B * W) ** 2
    hi = (ell >= 2000) & (ell <= 2500) if freq == 70 else (ell >= 2600) & (ell <= 3000)
    N = cl_mask[hi].mean()
    Cth = np.zeros(LMAX + 1); Cth[2:] = 2 * np.pi * Dth[2:] / (ell[2:] * (ell[2:] + 1))
    sky_hi = (Cth * BW2)[hi].mean()
    rec = (cl_mask - N) / BW2
    print(f"\n{freq} GHz: FWHM {FWHM[freq]}', f_sky {fsky:.4f}; anafast x3 in {time.time()-t:.0f} s")
    print(f"   monopole and dipole on the full sky: {mono_f:.1f} muK, |d| = {np.linalg.norm(dip_f):.1f} muK")
    print(f"   monopole and dipole on the unmasked sky: {mono:.1f} muK, |d| = {np.linalg.norm(dip):.1f} muK (removed)")
    print(f"   white noise from l = {ell[hi][0]}-{ell[hi][-1]}: N = {N:.4g} muK^2 sr "
          f"(sky in that range, theory x B^2 W^2: {sky_hi:.2g} muK^2 sr, {100*sky_hi/N:.2f}% of N)")
    print("   masked D_l without removing monopole and dipole / with:",
          ", ".join(f"l={L}: {cl_raw[L]/cl_mask[L]:.3g}" for L in (2, 3, 5, 10, 30, 100)))
    for q in (0.5, 0.1, 0.01):
        print(f"   B^2 = {q}: l = {np.argmax(B**2 < q)}", end=";")
    print(f" B^2 W^2 = 0.1: l = {np.argmax(BW2 < 0.1)}")
    print("   is the noise white? mean C_l^mask / N in bands of l:",
          ", ".join(f"{a}-{b}: {cl_mask[a:b].mean()/N:.3f}" for a, b in
                    ((1200, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000, LMAX + 1))))
    print("   l     D_full     D_mask   D_th*B2W2+DN    D_theory   B^2     W^2    D_recovered")
    DN = D(np.full(LMAX + 1, N))
    for L in (2, 10, 30, 100, 220, 540, 800, 1000, 1200, 1500, 2000):
        print(f"   {L:4d} {D(cl_full)[L]:10.4g} {D(cl_mask)[L]:10.4g} {Dth[L]*BW2[L]+DN[L]:12.4g} {Dth[L]:12.4g} "
              f"{B[L]**2:7.3f} {W[L]**2:7.3f} {D(rec)[L]:10.4g}")
    lc, Db = binned(D(rec)); _, Dtb = binned(Dth); _, nb = binned(D(np.full(LMAX + 1, N) / BW2))
    err = np.sqrt(2 / ((2 * lc + 1) * fsky * 30)) * (Dtb + nb)
    k = (lc > 30) & (lc < 2500)
    ratio = Db[k] / Dtb[k]
    lim = lc[(lc > 30) & (err / np.maximum(Dtb, 1e-30) > 0.5)]
    # 4b. residual: a white sky component seen through the beam (unresolved point sources?):
    #     C_l^mask - N = (C_th + A) B^2 W^2, A from weighted least squares over l = 30-1200
    s = slice(30, 1200)
    y, x = cl_mask[s] - N - Cth[s] * BW2[s], BW2[s]
    var = 2 * (Cth[s] * BW2[s] + N) ** 2 / ((2 * ell[s] + 1) * fsky)
    A = (x * y / var).sum() / (x * x / var).sum()
    chi_A, chi_0 = ((y - A * x) ** 2 / var).sum(), (y ** 2 / var).sum()
    print(f"   flat sky residual through the beam, fit over l = 30-1199: A = {A:.3g} muK^2 sr "
          f"(chi2 {chi_A:.0f} vs {chi_0:.0f} with A = 0, {s.stop - s.start} multipoles)")
    # alternative: no A, a narrower Gaussian beam (scan of the FWHM)
    best = (np.inf, None)
    for fw in np.arange(0.85, 1.0001, 0.0025) * FWHM[freq]:
        x2 = (hp.gauss_beam(np.radians(fw / 60), lmax=LMAX) * W)[s] ** 2
        c2 = ((cl_mask[s] - N - Cth[s] * x2) ** 2 / var).sum()
        best = min(best, (c2, fw))
    print(f"   alternative, A = 0 and a free Gaussian FWHM: best chi2 {best[0]:.0f} at FWHM {best[1]:.2f}'")
    print("   recovered / theory, mean over bins of 30 in l ranges:",
          ", ".join(f"{a}-{b}: {np.mean(Db[(lc >= a) & (lc < b)] / Dtb[(lc >= a) & (lc < b)]):.3f}"
                    for a, b in ((30, 300), (300, 600), (600, 800), (800, 1000), (1000, 1200), (1200, 1500))))
    _, Ab = binned(D(np.full(LMAX + 1, A)))
    print("   recovered / (theory + A), same ranges:",
          ", ".join(f"{a}-{b}: {np.mean(Db[(lc >= a) & (lc < b)] / (Dtb + Ab)[(lc >= a) & (lc < b)]):.3f}"
                    for a, b in ((30, 300), (300, 600), (600, 800), (800, 1000), (1000, 1200), (1200, 1500))))
    print(f"   expected error of a bin > 50% of the theory from l = {lim[0]:.0f}")
    out[freq] = dict(cl_full=cl_full, cl_mask=cl_mask, B=B, N=N, rec=rec, lc=lc, Db=Db, err=err, Dtb=Dtb, A=A, mm=mm, fsky=fsky)


if __name__ == "__main__":
    mask = hp.read_map(DATA / "mask_UT78_n1024.fits").astype(float)
    W, wsrc = pixel_window()
    Dth = theory()
    print(f"mask: f_sky = {mask.mean():.4f}; pixel window from {wsrc}; theory D_l up to l = {LMAX}")
    maps, out = {}, {}
    for freq in (70, 100):
        f = DATA / f"planck_T{freq:03d}_n1024_uK.fits"
        if not f.exists():
            print(f"\n{f.name} not found: skipped")
            continue
        maps[freq] = hp.read_map(f).astype(float)
        bad = maps[freq] < -1e30                         # UNSEEN = -1.6375e30 (stored as float32)
        if bad.any():                                    # 100 GHz: 1 unobserved pixel, inside the kept sky
            print(f"\n{freq} GHz: {bad.sum()} unobserved pixel(s), {int(mask[bad].sum())} inside the mask: "
                  "removed from the mask (common to both maps) and set to 0 in the full-sky map")
            mask[bad] = 0.0
            maps[freq][bad] = 0.0
    for freq in maps:
        spectra(freq, maps[freq], mask, W, Dth, out)
    if 70 in maps and 100 in maps:
        m1, m2 = out[70]["mm"], out[100]["mm"]
        cl_x = hp.anafast(m1, map2=m2, lmax=LMAX) / mask.mean()
        Bx = out[70]["B"] * out[100]["B"] * W ** 2
        rec_x = cl_x / Bx
        lc, Dbx = binned(D(rec_x)); _, Dtb = binned(Dth)
        print("\ncross-spectrum 70 x 100 (no noise subtraction), recovered D_l / theory, bins of 30:")
        for L in (100, 220, 540, 800, 1000, 1200, 1500):
            k = np.argmin(abs(lc - L))
            print(f"   l ~ {lc[k]:.0f}: cross {Dbx[k]/Dtb[k]:.3f}; auto 70 (noise subtracted) {out[70]['Db'][k]/Dtb[k]:.3f}; "
                  f"auto 100 {out[100]['Db'][k]/Dtb[k]:.3f}")
        out["x"] = dict(lc=lc, Db=Dbx)

    # ------------------------------------------------------------------ figures
    if 70 in maps:
        fig = plt.figure(figsize=(11, 3.4))
        hp.mollview(maps[70], fig=fig.number, sub=(1, 2, 1), min=-300, max=300, unit=r"$\mu$K",
                    title="Planck 70 GHz, Nside 1024", cmap="RdBu_r")
        mm = maps[70].copy(); mm[mask == 0] = hp.UNSEEN
        hp.mollview(mm, fig=fig.number, sub=(1, 2, 2), min=-300, max=300, unit=r"$\mu$K",
                    title=f"with the UT78 mask (f_sky = {mask.mean():.2f})", cmap="RdBu_r")
        plt.savefig(FIG / "fig_lab2_maps.png", dpi=150, bbox_inches="tight"); plt.close()

    for freq in out:
        if freq == "x":
            continue
        o = out[freq]
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].semilogy(ell[2:], D(o["cl_full"])[2:], lw=0.6, color="0.6", label="full sky")
        ax[0].semilogy(ell[2:], D(o["cl_mask"])[2:], lw=0.6, color="C0", label="masked, / $f_{sky}$")
        t = Dth > 0                                                     # the best fit stops at l = 2508
        ax[0].semilogy(ell[t], Dth[t], "k", lw=1.2, label=r"Planck best fit $D_\ell$")
        BW2 = (o["B"] * W) ** 2
        ax[0].semilogy(ell[t], (Dth * BW2 + D(np.full(LMAX + 1, o["N"])))[t], "--", color="C1", lw=1.2,
                       label=r"best fit $\times B_\ell^2W_\ell^2$ + noise")
        ax[0].set_xlabel(r"$\ell$"); ax[0].set_ylabel(r"$D_\ell=\ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
        ax[0].set_ylim(10, 2e5); ax[0].legend(fontsize=7); ax[0].set_title(f"{freq} GHz", fontsize=10)
        ax[1].errorbar(o["lc"], o["Db"], yerr=o["err"], fmt="o", ms=2, lw=0.8, color="C0",
                       label=r"$(C_\ell-N)/(B_\ell^2W_\ell^2)$, bins of 30")
        ax[1].plot(ell[2:], Dth[2:], "k", lw=1.2, label="Planck best fit")
        ax[1].plot(ell[2:], (Dth + D(np.full(LMAX + 1, o["A"])))[2:], "--", color="C3", lw=1.0,
                   label=rf"best fit + white residual $A$ = {o['A']:.1e} $\mu$K$^2$sr")
        ax[1].set_xlim(0, 2000); ax[1].set_ylim(-1000, 8000)
        ax[1].set_xlabel(r"$\ell$"); ax[1].legend(fontsize=7); ax[1].set_title(f"{freq} GHz: beam and noise corrected", fontsize=10)
        plt.tight_layout(); plt.savefig(FIG / f"fig_lab2_spectra{freq}.png", dpi=200); plt.close()
    if "x" in out:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for k, c, lab in ((70, "C0", "70 GHz auto"), (100, "C1", "100 GHz auto"), ("x", "C3", r"70$\times$100 cross")):
            ax.plot(out[k]["lc"], out[k]["Db"], "o", ms=2, color=c, label=lab)
        ax.plot(ell[2:], Dth[2:], "k", lw=1.2, label="Planck best fit")
        ax.set_xlim(0, 2500); ax.set_ylim(-1000, 8000); ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$D_\ell$ [$\mu$K$^2$]")
        ax.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(FIG / "fig_lab2_cross.png", dpi=200); plt.close()
