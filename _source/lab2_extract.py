"""Export compact data for the Lab 2 page widgets (binned spectra, Mollweide image)."""
import numpy as np, healpy as hp, json, base64, os
import sys
D = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Lab 2", "data")) + "/"
lmax = 3071; ell = np.arange(lmax + 1); dl = 10
edges = np.arange(2, lmax + 1, dl)
binit = lambda c: [float(f"{c[a:b].mean():.5g}") for a, b in zip(edges[:-1], edges[1:])]
th = np.loadtxt(D + "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt")
Dth = np.zeros(lmax + 1); Dth[2:2 + len(th)] = th[:, 1]
Cth = np.zeros(lmax + 1); Cth[2:] = 2*np.pi*Dth[2:]/(ell[2:]*(ell[2:]+1))
W = np.atleast_2d(hp.read_cl(D + "pixel_window_n1024.fits"))[0][:lmax + 1]
mask0 = hp.read_map(D + "mask_UT78_n1024.fits").astype(float)
out = {"dl": dl, "l0": 2, "nb": len(edges) - 1, "Cth": binit(Cth), "W2": binit(W**2)}
maps = {}
for f in (70, 100):
    m = hp.read_map(D + f"planck_T{f:03d}_n1024_uK.fits").astype(float)
    mk = mask0.copy(); bad = m < -1e30; mk[bad] = 0; m[bad] = 0
    fs = mk.mean()
    full = hp.anafast(m, lmax=lmax)
    raw = hp.anafast(m * mk, lmax=lmax) / fs
    q = m.copy(); q[mk == 0] = hp.UNSEEN; q = hp.remove_dipole(q); q[mk == 0] = 0
    cm = hp.anafast(q, lmax=lmax) / fs
    out[str(f)] = {"fsky": fs, "full": binit(full), "raw": binit(raw), "mask": binit(cm),
                   "lowraw": [float(raw[l]) for l in range(2, 31)], "lowmask": [float(cm[l]) for l in range(2, 31)]}
    maps[f] = (m, mk)
    print(f, fs, cm[2000:2501].mean())
# Mollweide image of the 70 GHz map (Nside 1024 -> 256 by averaging) + mask
m256 = hp.ud_grade(maps[70][0], 256); k256 = hp.ud_grade(maps[70][1], 256)
NX, NY = 360, 180
x = (np.arange(NX) + 0.5) / NX * 4 * np.sqrt(2) - 2 * np.sqrt(2)
y = np.sqrt(2) - (np.arange(NY) + 0.5) / NY * 2 * np.sqrt(2)
X, Y = np.meshgrid(x, y)
inside = (X / (2*np.sqrt(2)))**2 + (Y / np.sqrt(2))**2 <= 1
th_ = np.arcsin(np.clip(Y / np.sqrt(2), -1, 1))
lat = np.arcsin(np.clip((2*th_ + np.sin(2*th_)) / np.pi, -1, 1))
lon = -np.pi * X / (2*np.sqrt(2) * np.cos(th_))           # longitude increases to the left
pix = hp.ang2pix(256, np.pi/2 - lat[inside], lon[inside] % (2*np.pi))
img = np.full(NX*NY, -32768, np.int16); msk = np.zeros(NX*NY, np.uint8)
img.reshape(NY, NX)[inside] = np.clip(np.round(m256[pix] / 5), -32767, 32767)
msk.reshape(NY, NX)[inside] = (k256[pix] > 0.5).astype(np.uint8) + 1
out["img"] = {"nx": NX, "ny": NY, "scale": 5, "map": base64.b64encode(img.astype("<i2").tobytes()).decode(),
              "mask": base64.b64encode(msk.tobytes()).decode()}
js = "window.LAB2DATA = " + json.dumps(out, separators=(",", ":")) + ";\n"
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab2_data.js"), "w").write(js)
print(len(js) // 1024, "KB")
