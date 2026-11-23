"""Exercise of Lecture 11, reference solution (Python): from a beam to a map.

1. Rule: Nside for a beam of FWHM = 1 deg (2-3 pixels per FWHM or more).
2. 'True' sky: C_l = 1/(l(l+1)), l >= 2, smoothed by the 1 deg beam, on Nside = 512.
3. Degrade to Nside = 128 and 256 with healpy.ud_grade, without and with pre-smoothing to
   FWHM = 3 new pixels; aliased power in 2 Nside <= l < 3 Nside (same sky truncated at 3 Nside - 1).
4. Recover C_l on the chosen map: anafast / (B_l^2 W_l^2), W_l from simulations (average over the
   sub-pixels, exactly as ud_grade does from Nside 512: up = 2); mean ratio to the input in bands,
   against the cosmic variance; the same with the window of a continuous pixel (up = 8), wrong here.
5. Writes map_nside16.bin (RING order, float64) and its per-ring FFT moduli for the Fortran part.
"""
import numpy as np
import healpy as hp

FWHM = np.radians(1.0)
NHI = 512
LHI = 3 * NHI - 1


def res_arcmin(n):
    return np.degrees(hp.nside2resol(n)) * 60


def pixwin_sim(nside, nreal=4, up=8):
    """W_l of ud_grade: white sky on Nside*up, averaged into Nside pixels."""
    lmax = 3 * nside - 1
    acc = np.zeros(lmax + 1)
    for _ in range(nreal):
        alm = hp.synalm(np.ones(lmax + 1), lmax=lmax, new=True)
        lo = hp.ud_grade(hp.alm2map(alm, nside * up, lmax=lmax), nside)
        acc += hp.alm2cl(hp.map2alm(lo, lmax=lmax, iter=3)) / hp.alm2cl(alm) / nreal
    return np.sqrt(acc)


if __name__ == "__main__":
    np.random.seed(2026)
    print("1. pixels per FWHM for a 1 deg beam: " +
          ", ".join(f"Nside {n}: {60/res_arcmin(n):.1f}" for n in (64, 128, 256, 512)))
    ell = np.arange(LHI + 1)
    cl = np.zeros(LHI + 1); cl[2:] = 1 / (ell[2:] * (ell[2:] + 1.0))
    bl = hp.gauss_beam(FWHM, lmax=LHI)
    alm = hp.almxfl(hp.synalm(cl, lmax=LHI, new=True), bl)
    sky = hp.alm2map(alm, NHI, lmax=LHI)
    print(f"2. sky: Nside {NHI}, beam 1 deg; rms {sky.std():.4f}")
    for ns in (128, 256):
        lc = 3 * ns - 1
        alm_t = hp.resize_alm(hp.resize_alm(alm, LHI, LHI, lc, lc), lc, lc, LHI, LHI)
        for sm in (None, np.radians(3 * res_arcmin(ns) / 60)):
            c = []
            for a in (alm, alm_t):
                m = hp.alm2map(a, NHI, lmax=LHI)
                if sm is not None:
                    m = hp.smoothing(m, fwhm=sm, lmax=LHI)
                c.append(hp.anafast(hp.ud_grade(m, ns), lmax=lc))
            r = c[0] / c[1] - 1
            lab = "no pre-smoothing" if sm is None else f"pre-smoothed to {np.degrees(sm)*60:.0f}'"
            print(f"3. Nside {ns:3d}, {lab:24s}: aliased power {100*r[2*ns:3*ns].mean():6.2f}% (l = {2*ns}-{3*ns-1})")
    ns = 256; lc = 2 * ns
    w = pixwin_sim(ns, nreal=8, up=NHI // ns)[:lc + 1]
    w8 = pixwin_sim(ns, nreal=8, up=8)[:lc + 1]
    c_map = hp.anafast(hp.ud_grade(sky, ns), lmax=lc)
    rec = c_map / (bl[:lc + 1]**2 * w**2)
    for a, b in ((2, 50), (50, 200), (200, 400), (400, 512)):
        sl = slice(a, b)
        ratio = rec[sl] / cl[sl]
        cv = np.sqrt(2 / (2 * ell[sl] + 1)).mean() / np.sqrt(b - a)
        print(f"4. Nside {ns}, l = {a:3d}-{b-1:3d}: <C_l recovered / C_l input> = {ratio.mean():.3f} "
              f"(cosmic variance of the band mean ~ {cv:.3f}); without W_l: {(c_map[sl]/(bl[sl]**2*cl[sl])).mean():.3f}; "
              f"with the continuous-pixel W_l: {(c_map[sl]/(bl[sl]**2*w8[sl]**2*cl[sl])).mean():.3f}; B_l^2 at l = {b-1}: {bl[b-1]**2:.1e}")
    n16 = 16
    m16 = hp.ud_grade(sky, n16)
    m16.astype("<f8").tofile("map_nside16.bin")
    th, _ = hp.pix2ang(n16, np.arange(12 * n16**2))
    start, out = 0, []
    while start < m16.size:
        n = np.sum(np.isclose(th, th[start], atol=1e-14, rtol=0))
        out.append(np.abs(np.fft.rfft(m16[start:start + n]))[:5])
        start += n
    np.savetxt("ring_fft_py.txt", np.array([o[:5] for o in out if len(o) >= 5]), fmt="%.17e")
    print(f"5. Nside {n16} map written; per-ring |FFT| (m = 0..4) of the {len(out)} rings saved "
          f"(rings with at least 5 coefficients: {sum(len(o) >= 5 for o in out)})")
