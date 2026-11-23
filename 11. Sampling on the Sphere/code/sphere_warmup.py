"""Lesson 11 warm-up: healpy harmonic transforms, beams and pixel windows.
Synthesis is exact, analysis is a quadrature: check the round trip, the beam as a
multiplication in harmonic space, cosmic variance, and the HEALPix numbers.
Run: python3 sphere_warmup.py   (needs numpy and healpy)"""
import numpy as np
import healpy as hp

np.random.seed(11)
nside, lmax = 128, 2 * 128                       # lmax <= 2 Nside: accurate
ell = np.arange(lmax + 1)
cl = np.zeros(lmax + 1); cl[2:] = 1.0 / (ell[2:] * (ell[2:] + 1.0))

# 1. HEALPix numbers
print(f"Nside {nside}: Npix = {hp.nside2npix(nside)}, rings = {4*nside-1}, "
      f"pixel = {hp.nside2resol(nside, arcmin=True):.1f} arcmin")

# 2. round trip: synthesis exact, analysis approximate (Jacobi iterations help)
alm = hp.synalm(cl, lmax=lmax, new=True)         # random a_lm with spectrum C_l
m = hp.alm2map(alm, nside, lmax=lmax)            # synthesis
for it in (0, 1, 3):
    back = hp.map2alm(m, lmax=lmax, iter=it)     # analysis
    print(f"map2alm iter={it}: max|a - a_in| / max|a_in| = "
          f"{np.abs(back - alm).max() / np.abs(alm).max():.1e}")

# 3. a beam is a multiplication by B_l in harmonic space
fwhm = np.radians(1.0)
bl = hp.gauss_beam(fwhm, lmax=lmax)
m_s = hp.smoothing(m, fwhm=fwhm, lmax=lmax)      # map2alm -> * B_l -> alm2map
cl_s = hp.anafast(m_s, lmax=lmax)
k = slice(10, 200)
print(f"smoothed / input spectrum divided by B_l^2, l = 10-199: "
      f"{np.mean(cl_s[k] / (bl[k]**2 * hp.alm2cl(alm)[k])):.3f}")

# 4. cosmic variance: scatter of C_l-hat / C_l over 200 skies
r = np.array([hp.alm2cl(hp.synalm(cl, lmax=30, new=True))[2:31] / cl[2:31] for _ in range(200)])
for L in (2, 10, 30):
    print(f"l = {L:2d}: std(C_hat/C) = {r[:, L-2].std():.3f}   theory sqrt(2/(2l+1)) = {np.sqrt(2/(2*L+1)):.3f}")

# 5. the Planck 70 GHz beam: how much power survives
s = np.radians(13.08 / 60) / np.sqrt(8 * np.log(2))
for L in (515, 800, 1328):
    print(f"70 GHz beam: B_l^2 at l = {L}: {np.exp(-L*(L+1)*s*s):.3f}")
