"""
Laboratory 1 -- The discrete signal: spectral analysis and filtering of a noisy recording.
Example code: the whole laboratory, written the way a student would write it, with every
step explained: the core (steps 0-4, and recording 2) and the optional extensions (5-7).

  0. read, listen, look                 4. a comb by eye around the lines
  1. periodogram and Welch              5. fit of the noise spectrum (extension)
  2. is the signal stationary? (STFT)   6. Wiener filter: naive, then on the detected lines
  3. a band-pass by eye                 7. Whittle fit of the noise (extension)

What you need
-------------
* Python 3 with numpy, scipy and matplotlib.
* A folder data/ next to this script (or next to its folder) with the four files:
      dati_rumorosi.wav, segnale.wav        recording 1 and its clean signal
      dati_rumorosi_2.wav, segnale_2.wav    recording 2 and its clean signal
  Each file: 120 s, 44.1 kHz, 16-bit integers, one channel.

Run:   python3 lab1_starter.py        (RECORDING = 1 or 2 below; about 1 minute)
It prints the numbers of each step, saves the figures as PNG and the filtered recordings
as WAV in out_lab1/ (listen to them!). Set SHOW = True to open the figures on screen.
"""
import os
import numpy as np
import scipy.signal as sp
from scipy.io import wavfile
from scipy.fft import rfft, irfft, rfftfreq
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit, minimize
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
RECORDING = 1                     # 1: dati_rumorosi.wav, 2: dati_rumorosi_2.wav
SHOW = False                      # True: show the figures on screen as well
HUM = 50.0                        # mains hum [Hz]: an instrumental line, NOT signal

# The lines of the signal, READ OFF THE ZOOMED SPECTRUM of step 1 (the code prints the
# strongest peaks: run it once, look, then write the values here).
LINES = {1: [523.3, 659.3, 784.0],      # recording 1
         2: [440.0, 442.0, 880.0]}[RECORDING]

if not SHOW:
    matplotlib.use("Agg")         # no windows: only the PNG files

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
if not os.path.isdir(DATA):                       # course folder: data/ next to Codes/
    DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "out_lab1")
os.makedirs(OUT, exist_ok=True)
suffix = "" if RECORDING == 1 else "_2"


def read_wav(name):
    """Sampling rate and samples of a WAV file, as float64.
    The file stores 16-bit integers (-32768...32767): sums and squares of int16 would
    overflow, so convert to float before doing anything else."""
    fs, x = wavfile.read(os.path.join(DATA, name))
    return fs, x.astype(np.float64)


def write_wav(name, y, fs):
    """Save y as a 16-bit WAV, scaled to full scale (amplitude in arbitrary units)."""
    y16 = np.int16(y / np.abs(y).max() * 32767)
    wavfile.write(os.path.join(OUT, name), fs, y16)


def save(name):
    plt.tight_layout()
    plt.savefig(f"{name}{suffix}.png", dpi=120)
    if not SHOW:
        plt.close()


# ===========================================================================
# STEP 0 -- read, listen, look
# ===========================================================================
fs, x = read_wav(f"dati_rumorosi{suffix}.wav")
N = x.size                        # number of samples
T = N / fs                        # duration [s]
t = np.arange(N) / fs             # times of the samples: n / fs (not linspace(0, T, N)!)
print(f"Step 0: recording {RECORDING}: fs = {fs} Hz, N = {N}, T = {T:.1f} s, "
      f"rms = {x.std():.0f}, peak = {np.abs(x).max():.0f} (int16 full scale: 32767)")

# The whole record (one sample in 200: 5 million points are too many for a plot) and
# a zoom of 20 ms. The mean is removed in the zoom: the 1/f noise wanders slowly.
z = (t > 60) & (t < 60.02)                       # 20 ms, used again for every filter
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(t[::200], x[::200], lw=0.4)
ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("sample value"); ax[0].set_title("120 s")
ax[1].plot(1e3 * (t[z] - 60), x[z] - x[z].mean(), lw=1)
ax[1].set_xlabel("time [ms] after t = 60 s"); ax[1].set_title("20 ms")
save("step0_time")
# -> the samples are dominated by a slow, large noise: no tone is visible by eye.

# ===========================================================================
# STEP 1 -- periodogram and Welch
# ===========================================================================
# Periodogram: the squared modulus of the FFT of the whole record, in density units
# (one-sided, power per Hz): P_k = 2 |X_k|^2 / (fs N). Resolution 1/T = 0.008 Hz,
# but each value scatters by 100% around the true spectrum (chi^2 with 2 dof),
# however long the record is.
fp, P_scipy = sp.periodogram(x, fs)

# Welch: cut the record into K segments of L samples (here half-overlapping), multiply
# each by a window, average their periodograms. The scatter drops as 1/sqrt(K); the
# resolution becomes a few fs/L. L = 2^16 samples = 1.49 s, bins of fs/L = 0.67 Hz.
L = 2 ** 16
fw, W = sp.welch(x, fs, window="nuttall", nperseg=L)       # noverlap = L/2 by default
fb, Wbart = sp.welch(x, fs, window="boxcar", nperseg=L, noverlap=0)    # Bartlett
print(f"Step 1: Welch, L = {L} ({L/fs:.2f} s, bins of {fs/L:.3f} Hz), "
      f"K = {2 * N // L - 1} half-overlapping segments")

# The noise level of a window that leaks: compare the median of Welch at 50-100 Hz,
# away from the hum, for the rectangular window and for Nuttall. The rectangular window
# lets the huge power below 50 Hz leak into the higher bins.
band = (fw > 50) & (fw < 100) & (np.abs(fw - HUM) > 5)
print(f"        noise level at 50-100 Hz, Bartlett / Nuttall = "
      f"{np.median(Wbart[band]) / np.median(W[band]):.2f}")

# Which peaks stand out? Divide Welch by its running median (the local noise floor,
# over 201 bins = 135 Hz) and keep the local maxima more than 20 times above it.
floor = median_filter(W, size=201, mode="nearest")
peaks, _ = sp.find_peaks(W / floor, height=20)
peaks = peaks[fw[peaks] > 20]                    # ignore the first bins
order = peaks[np.argsort(W[peaks] / floor[peaks])[::-1]]
print("        strongest peaks [Hz] (height over the local floor):  " +
      ",  ".join(f"{fw[k]:.1f} ({W[k] / floor[k]:.0f})" for k in order[:8]))

# Two wrong windows and two right ones, zoomed on 30-850 Hz.
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
ax[0].loglog(fp[1::20], P_scipy[1::20], lw=0.3, color="0.6", label="periodogram (every 20th bin)")
ax[0].loglog(fw[1:], W[1:], lw=1, label=r"Welch, Nuttall, $2^{16}$")
ax[0].set_xlabel("frequency [Hz]"); ax[0].set_ylabel("PSD"); ax[0].legend(fontsize=7)
for w in ("boxcar", "flattop", "hann", "nuttall"):
    fq, Wq = sp.welch(x, fs, window=w, nperseg=L)
    s_ = (fq > 30) & (fq < 850)
    ax[1].loglog(fq[s_], Wq[s_], lw=1, label=w)
ax[1].set_xlabel("frequency [Hz]"); ax[1].legend(fontsize=7); ax[1].set_title("four windows")
save("step1_spectra")

# Recording 2: are the two lines at 440 and 442 Hz separated? They are 2 Hz apart: the
# segment must be long enough for its resolution (a few fs/L) to be below 2 Hz.
if RECORDING == 2:
    print("        440/442 Hz: dip between the peaks / lower peak (< 0.5: resolved)")
    for e in (16, 17, 18):
        fq, Wq = sp.welch(x, fs, window="nuttall", nperseg=2 ** e)
        p1, p2 = Wq[np.argmin(abs(fq - 440))], Wq[np.argmin(abs(fq - 442))]
        mid = (fq > 440.4) & (fq < 441.6)
        print(f"          L = 2^{e} ({2**e/fs:.2f} s): {Wq[mid].min() / min(p1, p2):.2f}")

# ===========================================================================
# STEP 2 -- is the signal stationary? Spectrogram (STFT)
# ===========================================================================
# Short FFTs of 1 s (Hann window), every 0.5 s: power against time and frequency.
# Lines that start, stop or move would show up here; a stationary line is a straight
# horizontal stripe.
fq, tt, Z = sp.stft(x, fs, window="hann", nperseg=fs, noverlap=fs // 2)
band = (fq > 30) & (fq < 850)
plt.figure(figsize=(10, 3.0))
plt.pcolormesh(tt, fq[band], 10 * np.log10(np.abs(Z[band]) ** 2 + 1e-12), shading="auto")
plt.xlabel("time [s]"); plt.ylabel("frequency [Hz]"); plt.title("spectrogram, 1 s window")
save("step2_stft")

if RECORDING == 2:
    # A short window (0.1 s: resolution 10 Hz) cannot separate 440 and 442 Hz: it sees
    # their sum, which beats at 442 - 440 = 2 Hz. A long window (2 s: 0.5 Hz) sees two
    # steady lines. Both are right: they answer different questions.
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.0))
    for a, Lsec in zip(ax, (0.1, 2.0)):
        Ls = int(Lsec * fs)
        fq, tt, Z = sp.stft(x[:20 * fs], fs, window="hann", nperseg=Ls, noverlap=Ls - Ls // 8)
        b = (fq > 436) & (fq < 446)
        power = np.sum(np.abs(Z[b]) ** 2, axis=0)
        mid = (tt > 2) & (tt < 18)
        depth = (power[mid].max() - power[mid].min()) / (power[mid].max() + power[mid].min())
        print(f"Step 2: STFT, window {Lsec} s: modulation depth of the power in 436-446 Hz = {depth:.2f}")
        a.plot(tt[mid], power[mid], lw=1)
        a.set_xlabel("time [s]"); a.set_title(f"power in 436-446 Hz, window {Lsec} s")
    save("step2_beats")

# ===========================================================================
# Filtering in Fourier space (steps 3, 4, 6)
# ===========================================================================
# X_k = rfft(x): one complex coefficient per frequency f_k = k fs / N, 0 <= f_k <= fs/2.
# A filter is a real weight H_k in [0, 1] (1 = keep, 0 = remove): Y_k = H_k X_k and
# y = irfft(Y, n=N). A real H does not shift the phases: the signal stays where it was.
# This filters the whole stored record at once.
f = rfftfreq(N, 1 / fs)
X = rfft(x)

fs_s, s = read_wav(f"segnale{suffix}.wav")       # the clean signal, for the comparison


def apply_filter(H, name, title):
    """Filter, save as WAV, and plot 20 ms against the clean signal."""
    y = irfft(H * X, n=N)
    write_wav(f"{name}{suffix}.wav", y, fs)
    plt.figure(figsize=(6, 3))
    plt.plot(1e3 * (t[z] - 60), s[z], lw=2, label="clean signal")
    plt.plot(1e3 * (t[z] - 60), y[z], lw=1, label="filtered data")
    plt.xlabel("time [ms] after t = 60 s"); plt.title(title); plt.legend(fontsize=7)
    save(name)
    return y


# ===========================================================================
# STEP 3 -- a band-pass by eye
# ===========================================================================
# Keep one band around the notes, 30 Hz beyond the lowest and the highest: everything
# else (the huge noise below a few hundred Hz, the hum, the white noise above) goes.
flo, fhi = min(LINES) - 30, max(LINES) + 30
H_band = ((f > flo) & (f < fhi)).astype(float)
apply_filter(H_band, "step3_bandpass", f"band-pass {flo:.0f}-{fhi:.0f} Hz")
print(f"Step 3: band-pass {flo:.0f}-{fhi:.0f} Hz: {int(H_band.sum())} bins kept")
# Try a band from 20 Hz: the 1/f noise and the hum come back.

# ===========================================================================
# STEP 4 -- a comb by eye
# ===========================================================================
# Keep only |f - f_line| < df around each line. The narrower the teeth, the less noise
# passes; but a tooth narrower than the resolution 1/T, or than the line, cuts the signal.
for df in (50.0, 2.5, 0.1):
    H_comb = np.zeros(f.size)
    for fl in LINES:
        H_comb[np.abs(f - fl) < df] = 1.0
    apply_filter(H_comb, f"step4_comb_{df:g}Hz", f"comb, half-width {df:g} Hz")
    print(f"Step 4: comb, half-width {df:g} Hz: {int(H_comb.sum())} bins kept")
# Never put a tooth at 50 Hz: the hum is not signal.

# ===========================================================================
# STEP 5 (extension) -- fit of the noise spectrum
# ===========================================================================
# Model: S_n(f) = A [1 + (fk / f)^alpha]. Fit its logarithm to log(Welch): each Welch value
# scatters by a fixed FRACTION of S, so in logarithms the error is the same at every
# frequency and an unweighted least squares is right; and the six decades of W do not
# let the few low-frequency points decide everything. Fitting ln A and ln fk keeps A, fk > 0.
def logS(f, lA, lfk, alpha):
    """ln S_n(f); np.log1p(u) = ln(1 + u), accurate also for u << 1 (above fk)."""
    return lA + np.log1p((np.exp(lfk) / f) ** alpha)


lines_all = [HUM] + LINES
sel = (fw >= 1) & (fw <= 20000)
far = np.all(np.abs(fw[:, None] - np.array(lines_all)) > 10, axis=1)    # lines out
p0 = [np.log(10), np.log(300), 1.5]                                      # initial guess
p_all, _ = curve_fit(logS, fw[sel], np.log(W[sel]), p0=p0)
p, cov = curve_fit(logS, fw[sel & far], np.log(W[sel & far]), p0=p0)
for lab, q in (("all points    ", p_all), ("lines excluded", p)):
    print(f"Step 5: fit, {lab}: A = {np.exp(q[0]):.2f}, fk = {np.exp(q[1]):.1f} Hz, alpha = {q[2]:.3f}")

plt.figure(figsize=(6, 3.2))
s_ = sel & far
plt.semilogx(fw[s_], W[s_] / np.exp(logS(fw[s_], *p)), ",", color="0.4")
plt.axhline(1, color="C1")
plt.xlabel("frequency [Hz]"); plt.ylabel("Welch / model"); plt.ylim(0.5, 1.5)
save("step5_fit")
# A good fit: data/model scatters around 1 at every frequency, with no trend.

# ===========================================================================
# STEP 6 (extension) -- the Wiener filter, H = 1 (no blur)
# ===========================================================================
# Phi = S_s / (S_s + S_n): ~1 where the signal dominates, ~0 where the noise does.
# With S_s ~ P - S_n (what the data have in excess of the noise): Phi = 1 - 1/z, z = P/S_n.
P = np.abs(X) ** 2 * 2 / (fs * N)              # periodogram, density: the same as scipy
Sn = np.exp(logS(np.maximum(f, 1e-3), *p))      # noise model at every frequency (no f = 0)
zk = P / Sn                                     # how many times the data exceed the noise
phi = np.clip(1 - 1 / zk, 0, 1)

# 6a. Naive: Phi in every bin. In a noise-only bin z is exponential around 1, so z > 1 in
# 37% of the bins: millions of noise bins pass, weighted by up to 1.
print(f"Step 6a: fraction of bins above 20 Hz with z > 1: {np.mean(zk[f > 20] > 1):.3f} (e^-1 = 0.368)")
apply_filter(phi, "step6a_wiener_naive", "naive Wiener, every bin")

# 6b. Only the bins where a line is DETECTED: z above a threshold z* that noise alone
# reaches with probability p = 0.01 in the whole record. M bins tested, Pr(z > z*) = e^-z*:
# M e^-z* = p  ->  z* = ln(M / p).
M = np.sum(f > 20)
zstar = np.log(M / 0.01)
det = (zk > zstar) & (f > 20)
print(f"Step 6b: M = {M}, z* = {zstar:.1f}; detected lines [Hz]: "
      + ", ".join(f"{v:.3f}" for v in f[det]))
det &= np.abs(f - HUM) > 0.5                    # detected, but instrumental: out
H_wien = np.where(det, phi, 0.0)
apply_filter(H_wien, "step6b_wiener_lines", "Wiener on the detected lines, 50 Hz out")
print(f"         kept after removing the hum: " + ", ".join(f"{v:.3f}" for v in f[det]))

# ===========================================================================
# STEP 7 (extension) -- Whittle fit of the noise
# ===========================================================================
# The periodogram values are independent and exponential around S(f): their likelihood
# gives -ln L = sum_k [ln S(f_k) + P_k / S(f_k)]. Minimise it over (ln A, ln fk, alpha),
# starting from the fit of step 5. The ln S term is what an ordinary chi^2 misses.
far = np.all(np.abs(f[:, None] - np.array(lines_all)) > 2, axis=1)
sel = (f >= 1) & (f <= 20000) & far
f_, P_ = f[sel], P[sel]


def nll(q):
    S = np.exp(logS(f_, *q))
    return np.sum(np.log(S) + P_ / S)


r = minimize(nll, p, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-8, fatol=1e-4))
print(f"Step 7: Whittle, lines excluded: A = {np.exp(r.x[0]):.3f}, fk = {np.exp(r.x[1]):.1f} Hz, "
      f"alpha = {r.x[2]:.3f}")
# Question: what happens if the lines are not excluded? (remove "& far" and run again)

print(f"Figures: *{suffix}.png here; filtered recordings: {OUT}")
if SHOW:
    plt.show()
