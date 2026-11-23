"""
fft_algorithms.py  --  Lecture 8: the FFT, from scratch to the library.
  1. recursive, iterative (in place, bit reversal) and vectorised radix-2 FFTs, checked
  2. timing against the direct DFT and numpy.fft; operation counts
  3. real-input packing trick (rfft from one complex FFT of half the length)
  4. 2D DFT = 1D FFTs along rows then columns (separability); the shift theorem
  5. notch filtering of a periodic artefact in an image
  6. a spectrogram (STFT) of a chirp with scipy.signal.ShortTimeFFT
Run:  python fft_algorithms.py     -> prints numbers, saves fft_algorithms.png
"""
import time
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(8)


# ---------------------------------------------------------------- 1. three FFTs
def fft_recursive(x):
    """Recursive radix-2 decimation in time (N a power of two)."""
    N = len(x)
    if N <= 1:
        return x.astype(complex)
    even = fft_recursive(x[0::2])
    odd = fft_recursive(x[1::2])
    tw = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd      # W_N^k O[k]
    return np.concatenate([even + tw, even - tw])                # butterflies


def bit_reverse_table(N):
    q = N.bit_length() - 1
    rev = np.zeros(N, dtype=int)
    for i in range(1, N):
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (q - 1))
    return rev


def fft_iterative(x):
    """In-place iterative radix-2 FFT: bit reversal, then log2 N stages of butterflies."""
    x = np.asarray(x, dtype=complex).copy()
    N = len(x)
    assert N & (N - 1) == 0, "N must be a power of two"
    x = x[bit_reverse_table(N)]
    m = 2
    while m <= N:
        w_m = np.exp(-2j * np.pi * np.arange(m // 2) / m)        # precomputed twiddles
        for k in range(0, N, m):
            t = w_m * x[k + m // 2: k + m]
            u = x[k: k + m // 2].copy()
            x[k: k + m // 2] = u + t
            x[k + m // 2: k + m] = u - t
        m *= 2
    return x


def fft_vectorized(x):
    """Vectorised Cooley-Tukey (after J. VanderPlas, 2013)."""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    Nmin = min(N, 2)
    n = np.arange(Nmin)
    X = np.exp(-2j * np.pi * n * n[:, None] / Nmin) @ x.reshape((Nmin, -1))
    while X.shape[0] < N:
        even, odd = X[:, : X.shape[1] // 2], X[:, X.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([even + factor * odd, even - factor * odd])
    return X.ravel()


x = rng.standard_normal(1024)
ref = np.fft.fft(x)
for f in (fft_recursive, fft_iterative, fft_vectorized):
    print(f"1. {f.__name__:<16} max |error| vs numpy: {np.max(np.abs(f(x) - ref)):.1e}")
print("   bit reversal N = 8:", bit_reverse_table(8))

# ---------------------------------------------------------------- 2. timing
print(f"\n2. {'N':>6} {'direct':>10} {'recursive':>10} {'iterative':>10} {'vectorised':>11} {'numpy':>9}   [ms]   N^2/((N/2)log2N)")
for q in (8, 10, 12):
    N = 2 ** q
    x = rng.standard_normal(N)
    W = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N)
    row = []
    for fun in (lambda v: W @ v, fft_recursive, fft_iterative, fft_vectorized, np.fft.fft):
        t0 = time.perf_counter(); reps = 3
        for _ in range(reps): fun(x)
        row.append(1e3 * (time.perf_counter() - t0) / reps)
    print(f"   {N:6d} " + " ".join(f"{v:10.3f}" for v in row) + f"   {N**2/(N/2*q):10.0f}")

# ---------------------------------------------------------------- 3. real-input packing
N = 1024
x = rng.standard_normal(N)
z = x[0::2] + 1j * x[1::2]                      # N/2 complex samples
Z = np.fft.fft(z)
k = np.arange(N // 2 + 1)
Zk, Zc = Z[k % (N // 2)], np.conj(Z[(-k) % (N // 2)])
E, O = 0.5 * (Zk + Zc), -0.5j * (Zk - Zc)
Xr = E + np.exp(-2j * np.pi * k / N) * O
print(f"\n3. rfft via one N/2 complex FFT: max error {np.max(np.abs(Xr - np.fft.rfft(x))):.1e}")

# ---------------------------------------------------------------- 4. 2D DFT
img = rng.standard_normal((64, 128))
F_rows_cols = np.fft.fft(np.fft.fft(img, axis=1), axis=0)     # rows, then columns
print(f"\n4. separability: max |fft(fft(rows),cols) - fft2| = {np.max(np.abs(F_rows_cols - np.fft.fft2(img))):.1e}")
shifted = np.roll(img, (5, 9), axis=(0, 1))
F1, F2 = np.fft.fft2(img), np.fft.fft2(shifted)
print(f"   shift theorem: |F| unchanged: {np.allclose(np.abs(F1), np.abs(F2))}")
R = F2 * np.conj(F1); r = np.fft.ifft2(R / np.abs(R)).real
print(f"   phase correlation finds the shift: {tuple(int(v) for v in np.unravel_index(np.argmax(r), r.shape))}  (true (5, 9))")

# ---------------------------------------------------------------- 5. notch filter
n1, n2 = np.mgrid[0:256, 0:256]
scene = np.exp(-((n1 - 128) ** 2 + (n2 - 100) ** 2) / 400) + 0.6 * np.exp(-((n1 - 80) ** 2 + (n2 - 180) ** 2) / 150)
striped = scene + 0.3 * np.cos(2 * np.pi * 20 * n1 / 256)    # horizontal stripes: v = 20 cycles/frame


def notch_filter(img, spikes, sigma=2.0):
    """Gaussian notch at each spike and at its conjugate partner (fftshift-ed indices)."""
    F = np.fft.fftshift(np.fft.fft2(img))
    rows, cols = F.shape
    rr, cc = np.ogrid[:rows, :cols]
    H = np.ones(F.shape)
    for r0, c0 in spikes:
        for r_, c_ in ((r0, c0), (rows - r0, cols - c0)):
            H *= 1.0 - np.exp(-((rr - r_) ** 2 + (cc - c_) ** 2) / (2 * sigma ** 2))
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))


clean = notch_filter(striped, [(128 + 20, 128)])
print(f"\n5. notch filter: rms error before {np.std(striped - scene):.2e}, after {np.std(clean - scene):.2e}")

# ---------------------------------------------------------------- 6. STFT of a chirp
fs = 2048
t = np.arange(4 * fs) / fs
chirp = np.sin(2 * np.pi * (50 * t + 0.5 * 87.5 * t ** 2)) + 0.5 * rng.standard_normal(t.size)
L = 256
stft = ShortTimeFFT(hann(L, sym=False), hop=L // 2, fs=fs, scale_to="psd")
Sx = stft.spectrogram(chirp)
ridge = stft.f[np.argmax(Sx, axis=0)]
tm = stft.t(len(chirp))
ok = (tm > 0.2) & (tm < 3.8)
print(f"\n6. STFT ridge follows f = 50 + 87.5 t to {np.sqrt(np.mean((ridge[ok] - (50 + 87.5 * tm[ok])) ** 2)):.1f} Hz rms"
      f" (bins of {fs / L:.0f} Hz)")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(striped, cmap="gray"); ax[0].set_title("image with stripes")
ax[1].imshow(clean, cmap="gray"); ax[1].set_title("after the notch")
ax[2].imshow(10 * np.log10(Sx + 1e-12), origin="lower", aspect="auto", extent=stft.extent(len(chirp)), cmap="magma")
ax[2].set(title="spectrogram of a chirp", xlabel="t [s]", ylabel="f [Hz]")
fig.tight_layout(); fig.savefig("fft_algorithms.png", dpi=110)
print("saved fft_algorithms.png")
