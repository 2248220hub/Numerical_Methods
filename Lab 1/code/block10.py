# Block 10 — Fit the noise spectrum (three parameters, in logarithms)
def logS(f, lA, lfk, alpha):                       # ln S_n, with A = e^lA, fk = e^lfk
    return lA + np.log1p((np.exp(lfk) / f)**alpha)

sel = (fw >= 1) & (fw <= 20000)
far = np.all(np.abs(fw[:, None] - np.array(lines_all)) > 10, axis=1)
p_all, _ = curve_fit(logS, fw[sel], np.log(W[sel]), p0=[np.log(10), np.log(300), 1.5])
p, cov = curve_fit(logS, fw[sel & far], np.log(W[sel & far]), p0=[np.log(10), np.log(300), 1.5])
err = np.sqrt(np.diag(cov))
# log-binned version: equal weight per decade
edges = np.logspace(0, np.log10(20000), 61)
idx = np.digitize(fw, edges)
fb_, Wb_ = [], []
for i in range(1, 61):
    m = (idx == i) & sel & far
    if m.sum():
        fb_.append(np.exp(np.mean(np.log(fw[m])))); Wb_.append(np.exp(np.mean(np.log(W[m]))))
p_bin, _ = curve_fit(logS, np.array(fb_), np.log(Wb_), p0=p)

print(f"{'fit':<32}{'A':>8}{'fk [Hz]':>10}{'alpha':>8}")
for name, q in [('all points', p_all), ('lines excluded', p), ('log bins, lines excluded', p_bin)]:
    print(f"{name:<32}{np.exp(q[0]):8.3f}{np.exp(q[1]):10.1f}{q[2]:8.3f}")
print(f"{'generator (truth)':<32}{'':>8}{440.0:10.1f}{2.0:8.3f}")
print(f"1-sigma errors (lines excluded): ln A ±{err[0]:.4f}, ln fk ±{err[1]:.4f}, alpha ±{err[2]:.4f}  (statistical only)")
resid = np.log(W[sel & far]) - logS(fw[sel & far], *p)
print(f"rms of the log residuals: {resid.std():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].loglog(fw[sel], W[sel], lw=0.6, label='Welch')
ax[0].loglog(fw[sel], np.exp(logS(fw[sel], *p)), 'r', lw=1.5, label='model (lines excluded)')
ax[0].set(xlabel='frequency [Hz]', ylabel='PSD', title='Noise model $S_n=A[1+(f_k/f)^α]$'); ax[0].legend(fontsize=8)
ax[1].semilogx(fw[sel & far], W[sel & far] / np.exp(logS(fw[sel & far], *p)), lw=0.4)
ax[1].axhline(1, color='k', lw=0.8)
ax[1].set(xlabel='frequency [Hz]', ylabel='data / model', title='Residuals: flat at 1 within the scatter')
savefig('b10_noise_fit')
