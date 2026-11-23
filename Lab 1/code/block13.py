# Block 13 — Whittle fit of the noise on the periodogram
def whittle(mask, start):
    fs_, Ps = f[mask], Pk[mask]
    def nll(q):
        S = np.exp(logS(fs_, *q))
        return np.sum(np.log(S) + Ps / S)
    return minimize(nll, start, method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-3, maxiter=2000)).x

base = (f >= 1) & (f <= 20000)
far2 = np.all(np.abs(f[:, None] - np.array(lines_all)) > 2, axis=1)
q_all = whittle(base, p)
q_far = whittle(base & far2, p)
print(f"{'estimator':<40}{'A':>8}{'fk [Hz]':>10}{'alpha':>8}")
for name, q in [('least squares on log Welch (lines out)', p), ('Whittle, all bins', q_all),
                ('Whittle, lines excluded', q_far)]:
    print(f"{name:<40}{np.exp(q[0]):8.3f}{np.exp(q[1]):10.1f}{q[2]:8.3f}")
print(f"{'generator (truth)':<40}{'':>8}{440.0:10.1f}{2.0:8.3f}")
