"""
hyperspectral_L12_bayes.py  --  Lecture 12 (Bayesian inference) applied to a hyperspectral cube.

  Part 0  look at the data: RGB composite, labels, mean spectrum of each class
  Part 1  Bayes classifier: Gaussian likelihood per class (in PCA space) x prior
          -> posterior probability of every class in every pixel, MAP class map.
          Flat prior versus "class frequency" prior: watch the rare class!
  Part 2  linear mixing model y = A theta + n  (Lecture 12, appendix C: everything exact)
          -> posterior mean and std of the abundances in every pixel,
             flat prior vs Gaussian prior, and a COVERAGE check against the truth.

Run:  python hyperspectral_L12_bayes.py                    (synthetic cube)
      python hyperspectral_L12_bayes.py indian_pines       (Part 0-1 only: no true abundances)
Figures: figures/L12_fig*.png
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hsi_io import load, rgb, FIG

scene = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
d = load(scene)
cube, wl, labels, names = d["cube"], d["wavelengths"], d["labels"], d["names"]
H, W, B = cube.shape
X = cube.reshape(-1, B)
y = labels.ravel()
classes = np.unique(y) if scene == "synthetic" else np.unique(y[y > 0])   # 0 = unlabelled
rng = np.random.default_rng(12)
print(f"{scene}: {H}x{W} pixels, {B} bands, classes {list(classes)}")

# ======================================================== Part 0: look first
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].imshow(rgb(cube, wl)); ax[0].set_title("RGB composite")
ax[1].imshow(labels, cmap="tab20", interpolation="nearest"); ax[1].set_title("ground truth labels")
for c in classes:
    s = X[y == c]
    ax[2].plot(wl, s.mean(0), label=str(names[c]))
    ax[2].fill_between(wl, s.mean(0) - s.std(0), s.mean(0) + s.std(0), alpha=0.15)
ax[2].set(xlabel="wavelength [nm]", ylabel="reflectance", title="class mean +- 1 std")
ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "L12_fig0_data.png"), dpi=110)

# ======================================================== Part 1: Bayes classifier
# 1a. reduce 100-200 correlated bands to a few components (PCA by SVD: Lecture on linear algebra)
mu = X.mean(0)
U, S, Vt = np.linalg.svd(X[rng.choice(len(X), 5000, replace=False)] - mu, full_matrices=False)
n_pc = 8
Z = (X - mu) @ Vt[:n_pc].T                          # every pixel -> 8 numbers
print(f"PCA: first {n_pc} components keep {100*np.sum(S[:n_pc]**2)/np.sum(S**2):.2f}% of the variance")

# 1b. training set: 2% of the labelled pixels, picked at random
lab = np.flatnonzero(np.isin(y, classes))
train = rng.choice(lab, size=max(len(lab) // 50, 20 * len(classes)), replace=False)
test = np.setdiff1d(lab, train)

# 1c. likelihood of each class: multivariate Gaussian N(m_c, C_c) in PCA space
loglike = np.empty((len(Z), len(classes)))
for i, c in enumerate(classes):
    Zc = Z[train][y[train] == c]
    m = Zc.mean(0)
    C = np.cov(Zc.T) + 1e-6 * np.eye(n_pc) * np.trace(np.cov(Zc.T))   # small ridge: few samples
    L = np.linalg.cholesky(C)                                        # C = L L^T
    r = np.linalg.solve(L, (Z - m).T)                                # whitened residuals
    loglike[:, i] = -0.5 * np.sum(r**2, 0) - np.sum(np.log(np.diag(L))) - 0.5 * n_pc * np.log(2 * np.pi)


def posterior(loglike, log_prior):
    lp = loglike + log_prior                         # ln L + ln pi
    lp -= lp.max(1, keepdims=True)                   # log-sum-exp: subtract the max first
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


freq = np.array([np.mean(y[train] == c) for c in classes])
priors = {"flat prior": np.log(np.ones(len(classes)) / len(classes)),
          "frequency prior": np.log(freq)}
results = {}
for pname, lpri in priors.items():
    post = posterior(loglike, lpri)
    pred = classes[post.argmax(1)]                   # MAP class
    acc = np.mean(pred[test] == y[test])
    recall = {names[c]: np.mean(pred[test][y[test] == c] == c) for c in classes}
    results[pname] = (post, pred)
    print(f"\n{pname}: overall accuracy {100*acc:.1f}%")
    print("   recall per class: " + ", ".join(f"{k} {100*v:.0f}%" for k, v in recall.items()))

post, pred = results["frequency prior"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].imshow(pred.reshape(H, W), cmap="tab20", interpolation="nearest"); ax[0].set_title("MAP class (frequency prior)")
im = ax[1].imshow(post.max(1).reshape(H, W), cmap="magma", vmin=0, vmax=1)
ax[1].set_title("max posterior probability (confidence)"); plt.colorbar(im, ax=ax[1])
ent = -np.sum(post * np.log(post + 1e-300), 1)
im = ax[2].imshow(ent.reshape(H, W), cmap="viridis"); ax[2].set_title("posterior entropy (uncertainty)")
plt.colorbar(im, ax=ax[2])
fig.tight_layout(); fig.savefig(os.path.join(FIG, "L12_fig1_classifier.png"), dpi=110)

# confusion matrix (rows: truth, columns: prediction)
cm = np.array([[np.sum((y[test] == a) & (pred[test] == b)) for b in classes] for a in classes])
print("\nconfusion matrix (frequency prior):\n", cm)

# ======================================================== Part 2: linear mixing model
if scene != "synthetic":
    print("\nPart 2 needs true abundances: run on the synthetic cube.")
    sys.exit()

A = d["endmembers"].T                                # (B, K): columns = endmember spectra
sig = d["noise_sigma"]                               # known noise per band
K = A.shape[1]
Cinv = 1 / sig**2                                    # diagonal C^-1
AtCA = A.T @ (A * Cinv[:, None])                     # A^T C^-1 A   (K x K)
AtCy = (X * Cinv) @ A                                # rows: (A^T C^-1 y)^T for every pixel

out = {}
for pname, (P0inv, th0) in {"flat prior": (np.zeros((K, K)), np.zeros(K)),
                            "Gaussian prior N(1/K, 0.3^2)": (np.eye(K) / 0.3**2, np.full(K, 1 / K))}.items():
    Sigma = np.linalg.inv(AtCA + P0inv)              # posterior covariance (same for all pixels)
    mean = (AtCy + P0inv @ th0) @ Sigma              # posterior mean, every pixel at once
    std = np.sqrt(np.diag(Sigma))
    truth = d["abundances"].reshape(-1, K)
    cover = np.mean(np.abs(mean - truth) < std, 0)   # frequentist check of a Bayesian interval
    rmse = np.sqrt(np.mean((mean - truth) ** 2, 0))
    out[pname] = mean
    print(f"\n{pname}:")
    for k in range(K):
        print(f"   {names[k]:10s} posterior std {std[k]:.4f}   rmse {rmse[k]:.4f}   coverage of +-1 std {100*cover[k]:.1f}%")

fig, ax = plt.subplots(2, K, figsize=(4 * K, 7))
mean = out["flat prior"]
for k in range(K):
    ax[0, k].imshow(d["abundances"][..., k], vmin=0, vmax=1); ax[0, k].set_title(f"true {names[k]}")
    ax[1, k].imshow(mean[:, k].reshape(H, W), vmin=0, vmax=1); ax[1, k].set_title(f"posterior mean {names[k]}")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "L12_fig2_unmixing.png"), dpi=110)
print("\nsaved L12_fig0_data.png, L12_fig1_classifier.png, L12_fig2_unmixing.png")
