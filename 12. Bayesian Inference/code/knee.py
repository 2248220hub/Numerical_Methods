"""
knee.py  --  the model signal of Lectures 12-14 (Computing Methods for Astrophysics)

Noise with a "knee":   S(f) = sigma2 * [ 1 + (fk / f)**alpha ]

  sigma2 : white-noise level           (units^2 / Hz)
  fk     : knee frequency              (Hz)   below fk the 1/f part dominates
  alpha  : slope of the 1/f part       (alpha = 1 -> "pink" 1/f noise)

We never need the time series to do inference: for a stationary Gaussian
process each periodogram bin P_j is (asymptotically) EXPONENTIAL with mean S(f_j)
  ->  P_j = -S_j * ln(u),  u ~ U(0,1]          (inversion sampling, Lecture 13)
and the bins are independent, which gives the Whittle likelihood (Lecture 12):
  -ln L(theta) = sum_j [ ln S_j + P_j / S_j ]
"""
import numpy as np

TRUTH = dict(sigma2=1e-3, fk=1.0, alpha=1.0)      # the simulation's truth


def frequencies(T=64.0, fs=100.0):
    """Fourier frequencies of a record of length T [s] sampled at fs [Hz].
    DC (j=0) and Nyquist (j=N/2) are excluded: 64 s at 100 Hz -> 3199 bins."""
    N = int(round(T * fs))
    j = np.arange(1, N // 2)
    return j / T


def spectrum(f, sigma2, fk, alpha):
    """Model power spectral density S(f)."""
    return sigma2 * (1.0 + (fk / f) ** alpha)


def simulate_periodogram(f, sigma2, fk, alpha, rng):
    """One noisy periodogram: exponential bins with mean S(f) (inversion)."""
    S = spectrum(f, sigma2, fk, alpha)
    u = 1.0 - rng.random(f.size)          # in (0,1]: never log(0)
    return -S * np.log(u)


def neg_log_like(theta, f, P):
    """Whittle -ln L for theta = (log10 sigma2, log10 fk, alpha).
    We fit log10 of the positive parameters: the posterior is more Gaussian there."""
    log10_s2, log10_fk, alpha = theta
    S = spectrum(f, 10.0 ** log10_s2, 10.0 ** log10_fk, alpha)
    return np.sum(np.log(S) + P / S)


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    f = frequencies()
    P = simulate_periodogram(f, **TRUTH, rng=rng)
    print(f"{f.size} frequency bins from {f[0]:.4f} to {f[-1]:.2f} Hz")
    print("mean of P/S (should be ~1):", np.mean(P / spectrum(f, **TRUTH)))
    print("-lnL at the truth:", neg_log_like((-3.0, 0.0, 1.0), f, P))
