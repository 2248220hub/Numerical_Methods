"""
hyperspectral_L04_06_linear.py  --  Lectures 4-6 (direct solvers, conditioning/SVD, sparse CG) on the cube.

  Part 1  unmixing every pixel as a least-squares problem  y = E theta + n  (E: 120 bands x 4 endmembers):
          normal equations + Cholesky, QR, SVD; noise weighting (divide each band by its sigma);
          condition numbers kappa(E) and kappa(E^T E)
  Part 2  SVD of the whole cube (pixels x bands, noise-whitened): how many singular values stand above
          the noise? -> the number of materials; rank-4 reconstruction as a denoiser
  Part 3  spatially regularised unmixing of the whole image at once:
              min  sum_p ||W (y_p - E theta_p)||^2 + lambda ||grad theta||^2
          a sparse SPD system with 4 x 128 x 128 = 65 536 unknowns, solved matrix-free with
          (block-Jacobi) preconditioned conjugate gradients

Run:  python hyperspectral_L04_06_linear.py       Figure: figures/L04_06_fig1.png
"""
import os
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hsi_io import load, FIG

d = load("synthetic")
cube, wl, E0, ab_true, sig = d["cube"].astype(float), d["wavelengths"], d["endmembers"], d["abundances"], d["noise_sigma"]
H, W, B = cube.shape
E = E0.T                                                    # bands x materials (120 x 4)
Y = cube.reshape(-1, B).T                                   # bands x pixels
T = ab_true.reshape(-1, E.shape[1]).T                       # materials x pixels
rms = lambda a: np.sqrt(np.mean(a ** 2))

# ---------------------------------------------------------------- 1. per-pixel least squares
Ew = E / sig[:, None]; Yw = Y / sig[:, None]                # whitening: every band in units of its noise
print("1. unmixing 16 384 pixels, 120 bands, 4 endmembers")
print(f"   kappa(E) = {np.linalg.cond(E):.1f}, kappa(E^T E) = {np.linalg.cond(E.T @ E):.1f};"
      f" whitened: kappa(W E) = {np.linalg.cond(Ew):.1f}")
res = {}
t0 = time.perf_counter(); c = la.cho_factor(Ew.T @ Ew); th = la.cho_solve(c, Ew.T @ Yw); res["normal eq. + Cholesky"] = (th, time.perf_counter() - t0)
t0 = time.perf_counter(); Q, R = la.qr(Ew, mode="economic"); th = la.solve_triangular(R, Q.T @ Yw); res["QR"] = (th, time.perf_counter() - t0)
t0 = time.perf_counter(); th = la.lstsq(Ew, Yw)[0]; res["SVD (lstsq)"] = (th, time.perf_counter() - t0)
t0 = time.perf_counter(); th = la.lstsq(E, Y)[0]; res["SVD, no noise weighting"] = (th, time.perf_counter() - t0)
for k, (th, t) in res.items():
    print(f"   {k:26s}: abundance rms error {rms(th - T):.4f}   ({t * 1e3:6.1f} ms)")
Fisher = Ew.T @ Ew
print("   predicted posterior std sqrt(diag((W E)^T (W E))^-1):", np.round(np.sqrt(np.diag(la.inv(Fisher))), 4),
      " measured:", np.round((res["QR"][0] - T).std(1), 4))

# ---------------------------------------------------------------- 2. SVD of the cube
X = (Yw - Yw.mean(1, keepdims=True)).T                     # pixels x bands, centred, noise-whitened
t0 = time.perf_counter(); U, s, Vt = la.svd(X, full_matrices=False); tsvd = time.perf_counter() - t0
noise_floor = np.sqrt(X.shape[0])                          # a pure unit-variance noise matrix has sigma ~ sqrt(n_pixels)
above = int(np.sum(s > 1.5 * noise_floor))
print(f"\n2. SVD of the whitened cube (16 384 x 120) in {tsvd:.2f} s")
print(f"   first singular values: {np.round(s[:7], 1)}; noise level sqrt(N_pix) = {noise_floor:.1f}")
print(f"   {above} components stand above 1.5 x the noise level -> {above + 1} materials (the mean removed one dimension)")
for r in (1, 2, 3, 5, 10):
    Xr = (U[:, :r] * s[:r]) @ Vt[:r]
    clean = ((model := (E @ T)) / sig[:, None] - Yw.mean(1, keepdims=True)).T
    print(f"   rank-{r:2d} reconstruction: rms difference from the noiseless cube {rms(Xr - clean):.3f} (in noise sigmas; raw data: {rms(X - clean):.3f})")

# ---------------------------------------------------------------- 3. spatially regularised unmixing with CG
boost = 6.0                                                 # make the noise 6x larger: regularisation matters
rng = np.random.default_rng(2)
Yn = E @ T + boost * sig[:, None] * rng.standard_normal(Y.shape)
Ewn = E / (boost * sig[:, None]); Ywn = Yn / (boost * sig[:, None])
G = Ewn.T @ Ewn                                            # 4 x 4, the same for every pixel
rhs = Ewn.T @ Ywn                                          # 4 x Npix
K = E.shape[1]; N = H * W


def lap(th):                                               # th: K x H x W -> -Laplacian with Neumann boundaries (graph Laplacian)
    out = np.zeros_like(th)
    for ax in (1, 2):
        dlt = np.diff(th, axis=ax)
        sl_lo = [slice(None)] * 3; sl_hi = [slice(None)] * 3
        sl_lo[ax] = slice(0, -1); sl_hi[ax] = slice(1, None)
        out[tuple(sl_lo)] -= dlt; out[tuple(sl_hi)] += dlt
    return out


def make_op(lam):
    def mv(v):
        th = v.reshape(K, H, W)
        return (np.einsum("ij,jhw->ihw", G, th) + lam * lap(th)).ravel()
    return spla.LinearOperator((K * N, K * N), matvec=mv, dtype=float)


Ginv = la.inv(G)
th_ls = (Ginv @ rhs)                                        # per-pixel least squares
print(f"\n3. noise x {boost:.0f}: per-pixel least squares abundance rms error {rms(th_ls - T):.4f}")
for lam in (0.0, 1.0, 10.0, 50.0, 200.0):
    A = make_op(lam)
    # block-Jacobi preconditioner: (G + 4 lam I)^-1 per pixel (exact for lam = 0)
    Pinv = la.inv(G + 4 * lam * np.eye(K))
    M = spla.LinearOperator(A.shape, matvec=lambda v, P=Pinv: (P @ v.reshape(K, N)).ravel(), dtype=float)
    it = [0]
    cb = lambda xk: it.__setitem__(0, it[0] + 1)
    t0 = time.perf_counter(); x, info = spla.cg(A, rhs.ravel(), rtol=1e-8, maxiter=5000, M=M, callback=cb); t = time.perf_counter() - t0
    it_p = it[0]; it[0] = 0
    x2, _ = spla.cg(A, rhs.ravel(), rtol=1e-8, maxiter=5000, callback=cb); it_n = it[0]
    print(f"   lambda = {lam:6.1f}: rms error {rms(x.reshape(K, N) - T):.4f};  CG iterations {it_n:4d} plain, {it_p:4d} block-Jacobi ({t * 1e3:.0f} ms)")
    if lam == 50.0:
        th_reg = x.reshape(K, H, W)

fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
ax[0].semilogy(np.arange(1, 21), s[:20], "o-"); ax[0].axhline(noise_floor, color="k", ls=":", label="noise level")
ax[0].set(xlabel="index i", ylabel="singular value", title="SVD of the whitened cube"); ax[0].legend()
ax[1].imshow(T[0].reshape(H, W), vmin=0, vmax=1, cmap="Greens"); ax[1].set_title("vegetation, truth")
ax[2].imshow(th_ls[0].reshape(H, W), vmin=0, vmax=1, cmap="Greens"); ax[2].set_title(f"per-pixel LS (noise x{boost:.0f})")
ax[3].imshow(th_reg[0], vmin=0, vmax=1, cmap="Greens"); ax[3].set_title("regularised, CG, lambda = 50")
for a in ax[1:]: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(os.path.join(FIG, "L04_06_fig1.png"), dpi=120)
print("saved figures/L04_06_fig1.png")
