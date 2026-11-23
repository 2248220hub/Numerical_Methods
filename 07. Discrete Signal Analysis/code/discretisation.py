"""
discretisation.py  --  Lecture 7: from a continuous signal to numbers, and the DFT.
  1. B-bit quantisation: measured SQNR against 6.02 B + 1.76 dB, under-utilisation, clipping
  2. aliasing: a 9 Hz tone sampled at 10 Hz looks like 1 Hz
  3. sinc (Whittaker-Shannon) reconstruction from the samples
  4. the direct O(N^2) DFT as a matrix, checked against numpy.fft
  5. the DFT grid: on-bin and off-bin tones (leakage), zero padding
  6. windows: a weak tone next to a strong one, rectangular vs Hann
Run:  python discretisation.py      -> prints numbers, saves discretisation.png
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)


# ---------------------------------------------------------------- 1. quantisation
def simulate_adc(x, B_bits, V_min, V_max):
    """B-bit mid-tread uniform quantiser with saturation (symmetric range)."""
    N_levels = 2 ** B_bits
    Delta = (V_max - V_min) / N_levels               # quantisation step (LSB)
    code = np.floor(x / Delta + 0.5)                  # nearest level
    code = np.clip(code, -N_levels // 2, N_levels // 2 - 1)   # two's-complement range
    return Delta * code, Delta


t = np.arange(200_000) / 200_000.0
print("1. quantisation of a full-scale sinusoid (V_FS = 5 V)")
print(f"   {'B':>3} {'Delta [V]':>11} {'var(e)':>11} {'Delta^2/12':>11} {'SQNR meas.':>11} {'6.02B+1.76':>11}")
for B in (4, 8, 12, 16):
    x = 2.5 * 0.9999 * np.sin(2 * np.pi * 7.123 * t)  # just below full scale: no clipping
    xq, D = simulate_adc(x, B, -2.5, 2.5)
    e = x - xq
    sqnr = 10 * np.log10(x.var() / e.var())
    print(f"   {B:3d} {D:11.3e} {e.var():11.3e} {D**2/12:11.3e} {sqnr:11.2f} {6.02*B+1.76:11.2f}")
for rho in (1.0, 0.25):
    x = 2.5 * 0.9999 * rho * np.sin(2 * np.pi * 7.123 * t)
    xq, D = simulate_adc(x, 12, -2.5, 2.5)
    print(f"   12 bits, signal using {rho:.2f} of full scale: SQNR = {10*np.log10(x.var()/(x-xq).var()):.1f} dB"
          f"  (B_eff = B + log2(rho) = {12 + np.log2(rho):.0f})")
xq, D = simulate_adc(2.8 * np.sin(2 * np.pi * 3 * t), 8, -2.5, 2.5)
print(f"   over-range 2.8 V into +-2.5 V: {np.mean(np.abs(xq) >= 2.5 - D):.1%} of the samples clipped")

# ---------------------------------------------------------------- 2. aliasing
f_sig, fs = 9.0, 10.0
k = round(f_sig / fs)
print(f"\n2. a {f_sig} Hz tone sampled at {fs} Hz appears at |f - k fs| = {abs(f_sig - k*fs):.1f} Hz")
n = np.arange(20)
same = np.allclose(np.cos(2*np.pi*f_sig*n/fs), np.cos(2*np.pi*abs(f_sig-k*fs)*n/fs))
print(f"   samples of the 9 Hz and of the 1 Hz cosine identical: {same}")

# ---------------------------------------------------------------- 3. sinc reconstruction
fs = 10.0; T = 1 / fs
n = np.arange(-200, 201)                             # many samples: the sinc tails are long
x_n = np.sin(2 * np.pi * 1.3 * n * T) + 0.5 * np.cos(2 * np.pi * 3.7 * n * T)
tt = np.linspace(-1, 2, 601)
x_rec = np.array([np.sum(x_n * np.sinc((ti - n * T) / T)) for ti in tt])   # np.sinc = sin(pi u)/(pi u)
x_true = np.sin(2 * np.pi * 1.3 * tt) + 0.5 * np.cos(2 * np.pi * 3.7 * tt)
print(f"\n3. sinc interpolation of a band-limited signal (fmax = 3.7 Hz < fs/2 = 5 Hz): "
      f"max error {np.max(np.abs(x_rec - x_true)):.1e} (truncated sum)")


# ---------------------------------------------------------------- 4. direct DFT
def dft_matrix(x):
    """Direct O(N^2) DFT: X = W x with W[k, n] = exp(-2 pi i k n / N)."""
    N = len(x)
    n = np.arange(N)
    W = np.exp(-2j * np.pi * np.outer(n, n) / N)
    return W @ x


x = rng.standard_normal(1024)
assert np.allclose(dft_matrix(x), np.fft.fft(x))
print("\n4. direct DFT == numpy.fft.fft:", True)
print(f"   {'N':>6} {'direct [ms]':>12} {'np.fft [ms]':>12} {'N^2/(N/2 log2 N)':>17}")
for N in (256, 512, 1024, 2048):
    x = rng.standard_normal(N)
    t0 = time.perf_counter(); dft_matrix(x); t1 = time.perf_counter()
    for _ in range(100): np.fft.fft(x)
    t2 = time.perf_counter()
    print(f"   {N:6d} {1e3*(t1-t0):12.2f} {1e3*(t2-t1)/100:12.4f} {N**2/(N/2*np.log2(N)):17.0f}")

# ---------------------------------------------------------------- 5. grid, leakage, padding
N, fs = 64, 64.0                                     # bins of 1 Hz
n = np.arange(N)
for f0 in (10.0, 10.5):
    X = np.abs(np.fft.rfft(np.cos(2 * np.pi * f0 * n / fs)))
    print(f"\n5. tone at {f0} Hz on a 1 Hz grid: bins with > 1% of the peak = {np.sum(X > 0.01*X.max())}")
Xp = np.abs(np.fft.rfft(np.cos(2*np.pi*10.5*n/fs), 1024))
print(f"   zero-padded to 1024: peak at {np.argmax(Xp)*fs/1024:.3f} Hz (finer grid, same resolution)")
x = rng.standard_normal(N)
X = np.fft.fft(x)
print(f"   Hermitian symmetry X[k] = conj X[N-k]: {np.allclose(X[1:], np.conj(X[1:][::-1]))};"
      f"  Parseval: {np.sum(x**2):.6f} = {np.sum(np.abs(X)**2)/N:.6f}")

# ---------------------------------------------------------------- 6. windows
fs, N = 1000.0, 1000
n = np.arange(N)
x = np.sin(2*np.pi*150.3*n/fs) + 1e-3*np.sin(2*np.pi*175.0*n/fs)   # weak line 60 dB below
f = np.fft.rfftfreq(4*N, 1/fs)
spec = {}
for name, w in (("rectangular", np.ones(N)), ("Hann", np.hanning(N)), ("Blackman", np.blackman(N))):
    S = np.abs(np.fft.rfft(x * w, 4*N))**2
    S /= S.max()
    spec[name] = 10*np.log10(S + 1e-30)
    i175 = np.argmin(np.abs(f - 175))
    print(f"6. {name:<12}: level at 175 Hz {spec[name][i175]:6.1f} dB, "
          f"floor at 190 Hz {spec[name][np.argmin(np.abs(f-190))]:6.1f} dB")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(t[:4000], (2.5*np.sin(2*np.pi*7.123*t) - simulate_adc(2.5*0.9999*np.sin(2*np.pi*7.123*t), 4, -2.5, 2.5)[0])[:4000])
ax[0].set(title="quantisation error, 4 bits", xlabel="t [s]")
ax[1].plot(tt, x_true, "k", lw=2, label="x(t)"); ax[1].plot(tt, x_rec, "r--", label="sinc sum")
ax[1].plot(n[:0], []); ax[1].set(title="sinc reconstruction", xlabel="t [s]"); ax[1].legend()
for k_, v in spec.items(): ax[2].plot(f, v, label=k_)
ax[2].set(xlim=(100, 250), ylim=(-140, 5), title="weak tone at 175 Hz, -60 dB", xlabel="f [Hz]"); ax[2].legend()
fig.tight_layout(); fig.savefig("discretisation.png", dpi=110)
print("saved discretisation.png")
