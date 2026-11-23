# Block 11 — Wiener filter, naive version (every bin)
Pk = np.abs(X)**2 * 2 / (fs * N)                    # periodogram in density units (= sp.periodogram)
Sn = np.exp(logS(np.maximum(f, 1e-3), *p))          # noise model at every frequency
z_snr = Pk / Sn                                      # how many times the data exceed the noise
phi = np.clip(1 - 1 / z_snr, 0, 1)                   # Wiener weight Phi = 1 - 1/z
y_naive = apply(phi, 'Wiener, naive (all bins)')

noise_bins = (f > 20) & np.all(np.abs(f[:, None] - np.array(lines_all)) > 2, axis=1)
print(f"fraction of noise-only bins with z > 1: {np.mean(z_snr[noise_bins] > 1):.3f}  (theory e^-1 = {np.exp(-1):.3f})")
print(f"SNR of the naive Wiener output: {snr_db(y_naive, s):.1f} dB")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
zz = z_snr[noise_bins][::7]
ax[0].hist(zz, bins=np.linspace(0, 8, 81), density=True, alpha=0.6, label='noise bins')
ax[0].plot(np.linspace(0, 8, 200), np.exp(-np.linspace(0, 8, 200)), 'k', label='$e^{-z}$')
ax[0].set(yscale='log', xlabel='z = P / S_n', title='Noise bins are exponential: 37% exceed 1'); ax[0].legend(fontsize=8)
ax[1].plot(t_ms, s[z], color='crimson', lw=2, label='clean')
ax[1].plot(t_ms, y_naive[z], lw=1, label='naive Wiener')
ax[1].set(xlabel='time [ms] after t = 60 s', title='The noise comes back (note the scale)'); ax[1].legend(fontsize=8)
savefig('b11_wiener_naive')
