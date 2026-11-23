"""
sampling.py  --  Lecture 13, part II: from uniform numbers to any distribution
  1. inversion: exponential   x = -ln(1-u)/lambda
  2. a periodogram bin of noise with spectrum S:  P = -S ln u
  3. Box-Muller: two uniforms -> two N(0,1)
  4. rejection: half-normal from an exponential envelope, M = sqrt(2e/pi)
  Every sampler is tested on its moments AND with a Kolmogorov-Smirnov test.
Run:  python sampling.py   -> prints numbers, saves sampling.png
"""
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(2027)
n = 10**6

# ---------------------------------------------------------------- 1. inversion
lam = 2.0
u = rng.random(n)
x_exp = -np.log(1.0 - u) / lam
ks = stats.kstest(x_exp[:10**5], stats.expon(scale=1 / lam).cdf)
print(f"1. Exp(2) by inversion: mean {x_exp.mean():.4f} (0.5), var {x_exp.var():.4f} (0.25), KS p = {ks.pvalue:.2f}")

# ---------------------------------------------------------------- 2. periodogram bin
S = 3.7
P = -S * np.log(1.0 - rng.random(n))
print(f"2. periodogram bin, S={S}: mean {P.mean():.4f}, std {P.std():.4f}  (both = S)")

# ---------------------------------------------------------------- 3. Box-Muller
u1 = 1.0 - rng.random(n // 2)          # (0,1]: ln(0) = -inf must never happen
u2 = rng.random(n // 2)
r = np.sqrt(-2.0 * np.log(u1))
z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])
ks = stats.kstest(z[:10**5], "norm")
print(f"3. Box-Muller: mean {z.mean():+.4f}, var {z.var():.4f}, excess kurtosis {stats.kurtosis(z):+.4f},"
      f" KS p = {ks.pvalue:.2f}")

# ---------------------------------------------------------------- 4. rejection
# target f(x) = sqrt(2/pi) exp(-x^2/2), x>0 ; envelope g(x) = exp(-x) ; f <= M g
M = np.sqrt(2 * np.e / np.pi)
xg = rng.exponential(1.0, n)                       # 1. draw from g
v = rng.random(n)                                  #    and v ~ U(0,1)
f = np.sqrt(2 / np.pi) * np.exp(-xg**2 / 2)
accept = v < f / (M * np.exp(-xg))                 # 2. accept if v < f/(M g)
x_half = xg[accept]
ks = stats.kstest(x_half[:10**5], stats.halfnorm.cdf)
print(f"4. rejection: accepted {accept.mean():.4f} (1/M = {1/M:.4f}), mean {x_half.mean():.4f}"
      f" (exact {np.sqrt(2/np.pi):.4f}), KS p = {ks.pvalue:.2f}")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
t = np.linspace(0, 3, 200)
ax[0].hist(x_exp, 100, range=(0, 3), density=True, alpha=.6); ax[0].plot(t, lam * np.exp(-lam * t), "k")
ax[0].set_title("inversion: Exp(2)")
t = np.linspace(-4, 4, 200)
ax[1].hist(z, 100, range=(-4, 4), density=True, alpha=.6); ax[1].plot(t, stats.norm.pdf(t), "k")
ax[1].set_title("Box-Muller: N(0,1)")
t = np.linspace(0, 4, 200)
ax[2].hist(x_half, 100, range=(0, 4), density=True, alpha=.6)
ax[2].plot(t, stats.halfnorm.pdf(t), "k", label="f"); ax[2].plot(t, M * np.exp(-t), "r--", label="M g")
ax[2].legend(); ax[2].set_title(f"rejection: accepted {accept.mean():.2f}")
fig.tight_layout(); fig.savefig("sampling.png", dpi=120)
print("saved sampling.png")
