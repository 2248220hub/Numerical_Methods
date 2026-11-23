"""
spectral_estimation.py  --  Lecture 10: a power spectrum from one finite record, with error bars.
  Model signal: 100 Hz sampling, harmonics at 7, 14, 21, 28 Hz in white + 1/f noise (knee 1 Hz).
  1. the periodogram is inconsistent: scatter of P/S does not shrink with T
  2. Welch: scatter ~ 1/sqrt(K), chi^2_nu band and its coverage
  3. a weak line next to a strong one: rectangular vs Hann
  4. coloured noise (Timmer & Koenig): generate alpha = 0, 1, 2 and measure the slope back
  5. the knee: least squares on log Welch vs Whittle on the periodogram (bias and scatter)
  6. a steep spectrum: leakage of the rectangular window, Hann, and prewhitening
Run:  python spectral_estimation.py      (about 20 s)  -> prints numbers, saves spectral_estimation.png
"""
import numpy as np
from scipy import signal as sp
from scipy.stats import chi2
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(10)
fs = 100.0


def coloured_noise(n, S, rng):
    """Timmer & Koenig (1995): complex Gaussian Fourier coefficients with E|X|^2 ~ S(f), inverse FFT.
    S is a one-sided PSD in units^2/Hz."""
    f = np.fft.rfftfreq(n, 1 / fs)
    s = np.zeros_like(f); s[1:] = S(f[1:])
    X = np.sqrt(n * fs * s / 4) * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
    X[-1] = X[-1].real * np.sqrt(2)                   # Nyquist bin real (n even)
    return np.fft.irfft(X, n)


Sknee = lambda f, s2=1e-3, fk=1.0, a=1.0: 2 * s2 * (1 + (fk / f) ** a)   # one-sided (factor 2)


def model_signal(T, rng):
    n = int(T * fs); t = np.arange(n) / fs
    x = coloured_noise(n, Sknee, rng)
    for k, A in zip((1, 2, 3, 4), (0.05, 0.03, 0.02, 0.01)):
        x += A * np.sin(2 * np.pi * 7 * k * t)
    return x


# ---------------------------------------------------------------- 1. periodogram
print("1. periodogram: scatter of P/S_n in 32-45 Hz (lines absent there)")
for T in (64, 256, 1024):
    x = model_signal(T, rng)
    f, P = sp.periodogram(x, fs)
    band = (f > 32) & (f < 45)
    r = P[band] / Sknee(f[band])
    print(f"   T = {T:5d} s: mean {r.mean():.2f}, std {r.std():.2f}  ({band.sum()} bins)")

# ---------------------------------------------------------------- 2. Welch + chi2 band
x = model_signal(64, rng)
L = 512
f, W = sp.welch(x, fs, window="hann", nperseg=L, noverlap=L // 2)
K = (len(x) - L) // (L // 2) + 1
nu = 2 * K * 0.95                                     # Hann 50%: K_eff ~ 0.95 K
lo, hi = nu * W / chi2.ppf(0.975, nu), nu * W / chi2.ppf(0.025, nu)
band = (f > 32) & (f < 45)
lines = np.any(np.abs(f[:, None] - np.array([7, 14, 21, 28])) < 1, axis=1)
free = (f > 0.5) & ~lines
cover = np.mean((Sknee(f[free]) >= lo[free]) & (Sknee(f[free]) <= hi[free]))
print(f"\n2. Welch, Hann, L = {L} (bins {fs/L:.2f} Hz), K = {K}, nu = {nu:.1f}:"
      f" scatter of W/S_n in 32-45 Hz {np.std(W[band]/Sknee(f[band])):.2f} (1/sqrt(K) = {1/np.sqrt(K):.2f})")
print(f"   95% band: from {nu/chi2.ppf(0.975, nu):.2f} to {nu/chi2.ppf(0.025, nu):.2f} times the estimate;"
      f" true S_n inside in {100*cover:.0f}% of {free.sum()} line-free bins")

# ---------------------------------------------------------------- 3. weak line
n = 4096; t = np.arange(n)
x = 10 * np.sin(2 * np.pi * 0.20 * t) + 1 * np.sin(2 * np.pi * 0.22 * t) + rng.standard_normal(n)
print("\n3. lines at 0.20 (A = 10) and 0.22 (A = 1, -20 dB), L = 256")
for win in ("boxcar", "hann"):
    fq, Wq = sp.welch(x, 1.0, window=win, nperseg=256)
    i22 = np.argmin(np.abs(fq - 0.22)); i21 = np.argmin(np.abs(fq - 0.212))
    print(f"   {win:<7}: weak line / valley at 0.212 = {Wq[i22]/Wq[i21]:6.1f}")

# ---------------------------------------------------------------- 4. colours of noise
print("\n4. generate pure power laws and measure the slope back (Welch, L = 4096, 0.1-10 Hz, 50 records of 655 s)")
for a in (0, 1, 2):
    slopes = []
    for _ in range(50):
        y = coloured_noise(65536, lambda f: f ** (-a), rng)
        fq, Wq = sp.welch(y, fs, nperseg=4096)
        m = (fq >= 0.1) & (fq <= 10)
        slopes.append(-np.polyfit(np.log10(fq[m]), np.log10(Wq[m]), 1)[0])
    print(f"   alpha = {a}: measured {np.mean(slopes):.3f} +- {np.std(slopes):.3f}")

# ---------------------------------------------------------------- 5. knee: LS vs Whittle
print("\n5. knee fit, 60 records of 64 s (truth sigma2 = 1e-3, fk = 1 Hz, alpha = 1)")
logS = lambda f, q: np.log(2 * 10 ** q[0]) + np.log1p((10 ** q[1] / f) ** q[2])
res_ls, res_wh = [], []
for _ in range(60):
    y = coloured_noise(6400, Sknee, rng)
    fq, Wq = sp.welch(y, fs, nperseg=1000)
    m = fq > 0
    ls = minimize(lambda q: np.sum((np.log(Wq[m]) - logS(fq[m], q)) ** 2), [-3, 0, 1], method="Nelder-Mead").x
    fp, P = sp.periodogram(y, fs)
    mp = fp > 0
    wh = minimize(lambda q: np.sum(logS(fp[mp], q) + P[mp] / np.exp(logS(fp[mp], q))), ls, method="Nelder-Mead").x
    res_ls.append(ls); res_wh.append(wh)
for name, r in (("least squares on log Welch", np.array(res_ls)), ("Whittle on the periodogram", np.array(res_wh))):
    print(f"   {name:<28}: sigma2 = {1e3*np.mean(10**r[:,0]):.3f}e-3 +- {1e3*np.std(10**r[:,0]):.3f}e-3,"
          f" fk = {np.mean(10**r[:,1]):.2f} +- {np.std(10**r[:,1]):.2f}, alpha = {np.mean(r[:,2]):.2f} +- {np.std(r[:,2]):.2f}")

# ---------------------------------------------------------------- 6. steep spectrum
Ssteep = lambda f: 1e-6 + (f / 10) ** (-4)
y = coloured_noise(2 ** 16, Ssteep, rng)
print("\n6. spectrum ~ f^-4 over a white floor, Welch L = 512: estimate / truth at 40 Hz")
for win in ("boxcar", "hann"):
    fq, Wq = sp.welch(y, fs, window=win, nperseg=512)
    i = np.argmin(np.abs(fq - 40))
    print(f"   {win:<7}: {Wq[i]/Ssteep(fq[i]):.3g}")
yd = np.diff(y)                                       # prewhiten: first difference, |A(f)|^2 = 4 sin^2(pi f/fs)
fq, Wd = sp.welch(yd, fs, window="hann", nperseg=512)
A2 = 4 * np.sin(np.pi * fq / fs) ** 2
i = np.argmin(np.abs(fq - 0.4))
fq2, Wh = sp.welch(y, fs, window="hann", nperseg=512)
print(f"   at the lowest bins (0.4 Hz): Hann alone {Wh[i]/Ssteep(fq[i]):.2f}, prewhitened {Wd[i]/A2[i]/Ssteep(fq[i]):.2f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
x = model_signal(64, rng)
fp_, P_ = sp.periodogram(x, fs)
ax[0].loglog(fp_[1:], P_[1:], ",", color="0.6"); ax[0].loglog(f[1:], W[1:], "b"); ax[0].fill_between(f[1:], lo[1:], hi[1:], alpha=.3)
ax[0].loglog(f[1:], Sknee(f[1:]), "r"); ax[0].set(xlabel="f [Hz]", title="periodogram, Welch and its 95% band")
for a in (0, 1, 2):
    y = coloured_noise(65536, lambda f: f ** (-a), rng); fq, Wq = sp.welch(y, fs, nperseg=4096)
    ax[1].loglog(fq[1:], Wq[1:], label=f"alpha = {a}")
ax[1].legend(); ax[1].set(xlabel="f [Hz]", title="colours of noise")
fig.tight_layout(); fig.savefig("spectral_estimation.png", dpi=110)
print("saved spectral_estimation.png")
