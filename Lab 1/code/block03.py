# Block 3 — Periodogram and Welch
fp, P = sp.periodogram(x, fs)                                  # one-sided PSD, whole record
L = 2**16                                                      # 1.49 s segments
fw, W = sp.welch(x, fs, window='nuttall', nperseg=L)           # 50% overlap (default)
fb, B = sp.welch(x, fs, window='boxcar', nperseg=L, noverlap=0)  # Bartlett
K = (N - L) // (L // 2) + 1
print(f"Welch: L = {L} samples ({L/fs:.2f} s), bins of {fs/L:.3f} Hz, K = {K} segments; Bartlett K = {N // L}")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].loglog(fp[1::20], P[1::20], lw=0.3, color='0.6', label='periodogram (every 20th bin)')
ax[0].loglog(fb[1:], B[1:], lw=1, label='Bartlett (boxcar, no overlap)')
ax[0].loglog(fw[1:], W[1:], lw=1, label='Welch, Nuttall, $2^{16}$')
ax[0].set(xlabel='frequency [Hz]', ylabel='PSD [units$^2$/Hz]', title='Whole band')
ax[0].legend(fontsize=8)
band = (fp > 30) & (fp < 850)
ax[1].semilogy(fp[band], P[band], lw=0.3, color='0.6', label='periodogram')
ax[1].semilogy(fw[(fw > 30) & (fw < 850)], W[(fw > 30) & (fw < 850)], lw=1.2, label='Welch')
ax[1].set(xlabel='frequency [Hz]', title='Zoom 30–850 Hz: four lines')
ax[1].legend(fontsize=8)
savefig('b03_periodogram_welch')
