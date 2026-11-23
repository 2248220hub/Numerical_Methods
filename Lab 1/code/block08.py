# Block 8 — Filtering in Fourier space; band-pass by eye (before / after)
f = rfftfreq(N, 1 / fs)
X = rfft(x)
results = {}                                    # name -> (filter H, output y)

def apply(H, name):
    y = irfft(H * X, n=N)                       # n=N: needed for odd N
    results[name] = (H, y)
    return y

y_bp = apply(((f > 493) & (f < 814)).astype(float), 'band-pass 493–814 Hz')
y_bp20 = apply(((f > 20) & (f < 814)).astype(float), 'band-pass 20–814 Hz (hum kept)')

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
Pb = np.abs(rfft(y_bp))**2 * 2 / (fs * N)
ax[0].semilogy(f[::50], (np.abs(X)**2 * 2 / (fs * N))[::50], lw=0.3, color='0.6', label='before')
ax[0].semilogy(f[::50], Pb[::50] + 1e-8, lw=0.5, label='after band-pass')
ax[0].set(xlim=(0, 2000), xlabel='frequency [Hz]', ylabel='periodogram', title='Spectrum before / after')
ax[0].legend(fontsize=8)
ax[1].plot(t_ms, s[z], color='crimson', lw=2, label='clean')
ax[1].plot(t_ms, y_bp[z], lw=1.2, label=f'band-pass ({snr_db(y_bp, s):.1f} dB)')
ax[1].set(xlabel='time [ms] after t = 60 s', title='20 ms after filtering'); ax[1].legend(fontsize=8)
savefig('b08_bandpass')
for k in results:
    print(f"{k:<34} SNR = {snr_db(results[k][1], s):6.1f} dB")
