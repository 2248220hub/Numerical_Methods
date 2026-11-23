# Block 15 — Recording 2: two lines 2 Hz apart (resolution, beat, filters)
fs2, x2 = read_wav('dati_rumorosi_2.wav')
_, s2 = read_wav('segnale_2.wav')
N2 = x2.size; t2 = np.arange(N2) / fs2
f2 = rfftfreq(N2, 1 / fs2); X2 = rfft(x2)
lines2 = [440.0, 442.0, 880.0]
lines2_all = [HUM] + lines2

# resolving 440 and 442 Hz: dip between the peaks / smaller peak
print(f"{'window, L':<22}{'bin [Hz]':>9}{'dip/peak 440-442':>18}")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
fp2, P2 = sp.periodogram(x2, fs2)
m = (fp2 > 434) & (fp2 < 448)
ax[0].semilogy(fp2[m], P2[m], lw=0.3, color='0.6', label='periodogram')
for win, q in [('nuttall', 16), ('flattop', 16), ('nuttall', 17), ('nuttall', 18)]:
    fq, Wq = sp.welch(x2, fs2, window=win, nperseg=2**q)
    mm = (fq > 434) & (fq < 448)
    i1, i2 = np.argmin(np.abs(fq - 440)), np.argmin(np.abs(fq - 442))
    dip = Wq[i1:i2 + 1].min() / min(Wq[i1], Wq[i2])
    print(f"{win + ', 2^' + str(q):<22}{fs2 / 2**q:9.3f}{dip:18.2f}")
    if win == 'nuttall' and q in (16, 18):
        ax[0].semilogy(fq[mm], Wq[mm], 'o-', ms=3, label=f'Welch nuttall $2^{{{q}}}$')
ax[0].set(xlabel='frequency [Hz]', title='440 + 442 Hz: resolved only with long segments'); ax[0].legend(fontsize=7)

for a, Lsec in zip(ax[1:], [0.1, 2.0]):
    nps = int(Lsec * fs2)
    fq, tt, Zq = sp.stft(x2, fs2, window='hann', nperseg=nps, noverlap=nps // 2)
    bnd = (fq > 425) & (fq < 455)
    a.pcolormesh(tt, fq[bnd], 10 * np.log10(np.abs(Zq[bnd])**2 + 1e-12), shading='auto', cmap='magma')
    a.set(xlim=(60, 64) if Lsec < 1 else (0, 120), xlabel='time [s]', ylabel='frequency [Hz]',
          title=f'STFT, {Lsec} s window: ' + ('one line that beats' if Lsec < 1 else 'two steady lines'))
    pw = (np.abs(Zq[(fq > 436) & (fq < 446)])**2).sum(0)[2:-2]
    print(f"STFT {Lsec} s: modulation depth of the 436-446 Hz power = {(pw.max() - pw.min()) / (pw.max() + pw.min()):.2f}")
savefig('b15_rec2_resolution')

# noise fit, detection and filters on recording 2
fw2, W2 = sp.welch(x2, fs2, window='nuttall', nperseg=2**16)
sel2 = (fw2 >= 1) & (fw2 <= 20000) & np.all(np.abs(fw2[:, None] - np.array(lines2_all)) > 10, axis=1)
p2, _ = curve_fit(logS, fw2[sel2], np.log(W2[sel2]), p0=[np.log(10), np.log(300), 1.5])
print(f"noise fit, recording 2: A = {np.exp(p2[0]):.2f}, fk = {np.exp(p2[1]):.1f} Hz, alpha = {p2[2]:.3f}")
Pk2 = np.abs(X2)**2 * 2 / (fs2 * N2)
z2 = Pk2 / np.exp(logS(np.maximum(f2, 1e-3), *p2))
phi2 = np.clip(1 - 1 / z2, 0, 1)
det2 = (z2 > np.log(np.sum(f2 > 20) / 0.01)) & (f2 > 20) & (np.abs(f2 - HUM) > 0.5)
print('detected lines [Hz]:', ', '.join(f'{v:.3f} ({note_name(v)})' for v in f2[det2]))

comb2 = np.zeros(f2.size)
for fl in lines2:
    comb2[np.abs(f2 - fl) < 2.5] = 1.0
res2 = {'band-pass 400–920 Hz': ((f2 > 400) & (f2 < 920)).astype(float),
        'comb 2.5 Hz (one band for 440+442)': comb2,
        'Wiener, detected lines (hum vetoed)': np.where(det2, phi2, 0.0)}
zz2 = (t2 > 60) & (t2 < 60.02)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4), sharey=True)
print(f"{'filter (recording 2)':<40}{'SNR [dB]':>10}")
for a, (name, H) in zip(ax, res2.items()):
    y2 = irfft(H * X2, n=N2)
    res2[name] = (H, y2)
    print(f"{name:<40}{snr_db(y2, s2):10.1f}")
    a.plot((t2[zz2] - 60) * 1e3, s2[zz2], color='crimson', lw=2, label='clean')
    a.plot((t2[zz2] - 60) * 1e3, y2[zz2], lw=1, label='filtered')
    a.set(title=f'{name}', xlabel='time [ms] after 60 s'); a.legend(fontsize=7)
savefig('b15_rec2_filters')
write_wav('rec2_noisy.wav', x2 - x2.mean(), fs2)
write_wav('rec2_comb.wav', res2['comb 2.5 Hz (one band for 440+442)'][1], fs2)
