# Block 9 — A comb around the lines: width of the teeth (table + plot)
def comb(centres, df):
    H = np.zeros(f.size)
    for fl in centres:
        H[np.abs(f - fl) < df] = 1.0
    return H

print(f"{'half-width df [Hz]':>18} {'bins kept':>10} {'SNR [dB]':>9}")
dfs = [50, 10, 2.5, 0.5, 0.1, 0.01]
snrs = []
for df in dfs:
    H = comb(lines, df)
    y = irfft(H * X, n=N)
    snrs.append(snr_db(y, s))
    print(f"{df:18g} {int(H.sum()):10d} {snrs[-1]:9.1f}")
y_comb = apply(comb(lines, 2.5), 'comb, 2.5 Hz teeth')
y_combhum = apply(comb(lines_all, 2.5), 'comb 2.5 Hz + tooth at 50 Hz')
print(f"comb with a tooth at 50 Hz: SNR = {snr_db(y_combhum, s):.1f} dB (the hum comes back)")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].semilogx(dfs, snrs, 'o-'); ax[0].invert_xaxis()
ax[0].set(xlabel='tooth half-width df [Hz] (narrower →)', ylabel='SNR [dB]', title='Narrower teeth, less noise')
ax[1].plot(t_ms, s[z], color='crimson', lw=2, label='clean')
ax[1].plot(t_ms, y_comb[z], lw=1.2, label='comb 2.5 Hz')
ax[1].plot(t_ms, y_combhum[z], lw=0.8, alpha=0.7, label='comb + 50 Hz')
ax[1].set(xlabel='time [ms] after t = 60 s', title='20 ms after the comb'); ax[1].legend(fontsize=8)
savefig('b09_comb')
