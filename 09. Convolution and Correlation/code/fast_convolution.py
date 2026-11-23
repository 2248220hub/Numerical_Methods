"""
fast_convolution.py  --  Lecture 9: convolution with the FFT, in blocks, and its inverse.
  1. circular vs linear convolution: wrap-around pollution and zero padding
  2. overlap-add and overlap-save (any kernel: causal, centred, asymmetric), checked vs np.convolve
  3. cost per output sample against the FFT length (optimum ~ 4-8 M)
  4. correlation = convolution with the time-reversed kernel: a matched filter finds a pulse
  5. deconvolution: plain inverse, Tikhonov (best constant lambda), Wiener from the spectra
  6. a causal filter delays: centred Hann vs causal Hann vs RC (IIR)
Run:  python fast_convolution.py   -> prints numbers, saves fast_convolution.png
"""
import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import lfilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(9)


# ---------------------------------------------------------------- 1. wrap-around
N, M = 64, 15
x = np.zeros(N); x[40:] = 1.0                       # a step near the end of the record
h = np.exp(-np.arange(M) / 4.0); h /= h.sum()        # causal kernel
lin = np.convolve(x, h)                              # N + M - 1 samples
circ = irfft(rfft(x, N) * rfft(h, N), N)             # Nfft = N: circular
pad = irfft(rfft(x, N + M - 1) * rfft(h, N + M - 1), N + M - 1)
print("1. step convolved with a causal kernel (N = 64, M = 15)")
print(f"   Nfft = N        : max error on the first M-1 samples {np.max(np.abs(circ[:M-1] - lin[:M-1])):.3f} (wrap-around)")
print(f"   Nfft = N + M - 1: max error everywhere {np.max(np.abs(pad - lin)):.1e}")


# ---------------------------------------------------------------- 2. overlap-add / save
def kernel_spectrum(h, m_minus, nfft):
    """h[0] is lag -m_minus. Wrap-around storage: lags 0..m_plus first, negative lags at the end."""
    hb = np.zeros(nfft)
    hb[: len(h) - m_minus] = h[m_minus:]
    if m_minus:
        hb[nfft - m_minus:] = h[:m_minus]
    return rfft(hb)


def overlap_add(x, h, nfft, m_minus=0):
    M = len(h); B = nfft - M + 1
    H = kernel_spectrum(h, m_minus, nfft)
    y = np.zeros(len(x) + M - 1 + nfft)
    for s in range(0, len(x), B):                    # disjoint blocks of B samples
        yb = irfft(rfft(x[s:s + B], nfft) * H, nfft)
        y[s:s + nfft] += np.roll(yb, m_minus)        # spreads m_minus left, m_plus right
    return y[: len(x) + M - 1]


def overlap_save(x, h, nfft, m_minus=0):
    M = len(h); B = nfft - M + 1; m_plus = M - 1 - m_minus
    H = kernel_spectrum(h, m_minus, nfft)
    xp = np.concatenate((np.zeros(M - 1), x, np.zeros(nfft)))
    Ny = len(x) + M - 1
    y = np.empty(Ny)
    for s in range(0, Ny, B):                        # overlapping blocks of nfft samples
        yb = irfft(rfft(xp[s:s + nfft]) * H, nfft)
        n = min(B, Ny - s)
        y[s:s + n] = yb[m_plus:m_plus + n]           # discard m_plus at start, m_minus at end
    return y


x = rng.standard_normal(10_000)
h = rng.standard_normal(31)
ref = np.convolve(x, h)
worst = max(max(np.max(np.abs(f(x, h, 256, mm) - ref)) for f in (overlap_add, overlap_save)) for mm in range(31))
print(f"\n2. overlap-add and overlap-save vs np.convolve, every m_minus = 0..30: max error {worst:.1e}")

# ---------------------------------------------------------------- 3. FFT length
print("\n3. cost per output sample ~ Nfft log2 Nfft / (Nfft - M + 1), M = 101")
M = 101
for nfft in (128, 256, 512, 1024, 4096, 16384):
    print(f"   Nfft = {nfft:6d} ({nfft/M:5.1f} M): {nfft*np.log2(nfft)/(nfft-M+1):6.1f}")

# ---------------------------------------------------------------- 4. matched filter
L = 4096
template = np.sin(2 * np.pi * np.linspace(0, 8, 200) * np.linspace(1, 3, 200)) * np.hanning(200)   # a short chirp
data = rng.standard_normal(L)
t0 = 2600
data[t0:t0 + 200] += 0.6 * template                  # buried: amplitude below the noise rms
corr = irfft(rfft(data, 2 * L) * np.conj(rfft(template, 2 * L)), 2 * L)[:L]   # correlation: H -> H*
print(f"\n4. matched filter: peak of the correlation at lag {np.argmax(corr)} (true {t0}),"
      f" peak / rms = {corr.max()/corr.std():.1f}")

# ---------------------------------------------------------------- 5. deconvolution
N = 4096
f = np.fft.rfftfreq(N)
Xtrue = (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)) / np.sqrt(1 + (f / 0.01) ** 2)
x_true = irfft(Xtrue, N); x_true /= x_true.std()                     # red-noise "sky"
Hf = np.exp(-0.5 * (f / 0.02) ** 2)                                   # Gaussian blur
sig_n = 0.02
y = irfft(rfft(x_true) * Hf, N) + sig_n * rng.standard_normal(N)
Y = rfft(y)
rms = lambda xh: np.sqrt(np.mean((xh - x_true) ** 2))
inv = irfft(np.where(Hf > 1e-12, Y / np.maximum(Hf, 1e-12), 0), N)
lams = np.logspace(-6, 0, 61)
errs = [rms(irfft(np.conj(Hf) * Y / (np.abs(Hf) ** 2 + l), N)) for l in lams]
Sx = np.abs(rfft(x_true)) ** 2                                         # (the truth: for reference only)
Sn = N * sig_n ** 2
from scipy.ndimage import uniform_filter1d
Ps = uniform_filter1d(np.abs(Y) ** 2, 33)                              # smoothed |Y|^2 (single bins scatter 100%)
cut = np.argmax(Ps < 1.5 * Sn)                                         # where the data sink into the noise floor
signal = np.where(np.arange(f.size) < cut, np.clip(Ps - Sn, 0, None), 0.0)   # |H|^2 S_x from the data
Phi = signal / (signal + Sn)                                           # in [0, 1]; 0 above the cut
wien = irfft(np.where(Phi > 0, Phi * Y / np.maximum(Hf, 1e-12), 0), N)
print(f"   noise floor reached at f = {f[cut]:.3f} cycles/sample")
print(f"\n5. deconvolution rms error: blurred data {rms(y):.3f}, plain inverse {rms(inv):.2e},"
      f" best Tikhonov {min(errs):.3f} (lambda = {lams[np.argmin(errs)]:.1e}), Wiener from the data {rms(wien):.3f}")

# ---------------------------------------------------------------- 6. delay of causal filters
pulse = np.zeros(400); pulse[100:110] = 1.0
hk = np.hanning(21); hk /= hk.sum()
centred = np.convolve(pulse, hk, mode="same")
causal = np.convolve(pulse, hk)[:400]
a = np.exp(-1 / 6.0)
rc = lfilter([1 - a], [1, -a], pulse)                                   # y[n] = a y[n-1] + (1-a) x[n]
centroid = lambda v: np.sum(np.arange(v.size) * v) / np.sum(v)
print(f"\n6. centroid of a pulse at 104.5: centred {centroid(centred):.1f}, causal Hann {centroid(causal):.1f}"
      f" (delay (M-1)/2 = 10), RC {centroid(rc):.1f} (delay tau = 6)")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(lin, "k", lw=2, label="linear"); ax[0].plot(circ, "r--", label="circular, Nfft = N")
ax[0].legend(); ax[0].set_title("wrap-around pollution")
ax[1].semilogx(lams, errs); ax[1].axhline(rms(wien), color="g", label="Wiener from data")
ax[1].set(xlabel="lambda", ylabel="rms error", title="Tikhonov vs Wiener"); ax[1].legend()
ax[2].plot(pulse, "k"); ax[2].plot(centred, label="centred"); ax[2].plot(causal, label="causal Hann"); ax[2].plot(rc, label="RC")
ax[2].set(xlim=(80, 170), title="causal filters delay"); ax[2].legend()
fig.tight_layout(); fig.savefig("fast_convolution.png", dpi=110)
print("saved fast_convolution.png")
