# Block 12 — Wiener filter on the detected lines (threshold z*, hum removed)
M = np.sum(f > 20)                                   # bins tested
zstar = np.log(M / 0.01)                             # one false line in 100 recordings
det_all = (z_snr > zstar) & (f > 20)
det = det_all & (np.abs(f - HUM) > 0.5)              # veto the mains hum
print(f"M = {M} bins tested, z* = ln(M/0.01) = {zstar:.1f}; largest noise-only z = {z_snr[noise_bins].max():.1f}")
print(f"{'f [Hz]':>9} {'z':>9} {'amplitude 2|X|/N':>17}   nature")
for k in np.flatnonzero(det_all):
    kind = 'mains hum: vetoed' if abs(f[k] - HUM) < 1 else note_name(f[k])
    print(f"{f[k]:9.3f} {z_snr[k]:9.0f} {2 * np.abs(X[k]) / N:17.1f}   {kind}")
y_wien = apply(np.where(det, phi, 0.0), 'Wiener, detected lines (hum vetoed)')
y_wienhum = apply(np.where(det_all, phi, 0.0), 'Wiener, detected lines (hum kept)')

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].plot(t_ms, s[z], color='crimson', lw=2, label='clean')
ax[0].plot(t_ms, y_wienhum[z], lw=1, label=f'hum kept ({snr_db(y_wienhum, s):.1f} dB)')
ax[0].set(xlabel='time [ms] after t = 60 s', title='Without the veto: the 50 Hz hum'); ax[0].legend(fontsize=8)
ax[1].plot(t_ms, s[z], color='crimson', lw=3, label='clean')
ax[1].plot(t_ms, y_wien[z], 'k--', lw=1.2, label=f'Wiener, hum vetoed ({snr_db(y_wien, s):.1f} dB)')
ax[1].set(xlabel='time [ms] after t = 60 s', title='Three bins kept: the clean waveform'); ax[1].legend(fontsize=8)
savefig('b12_wiener_lines')
