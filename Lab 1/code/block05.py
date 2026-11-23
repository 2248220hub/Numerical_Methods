# Block 5 — Which window? Leakage and line contrast (table)
fr, Wref = sp.welch(x, fs, window='nuttall', nperseg=2**18)   # reference: long Nuttall
def level(fq, Wq, lo, hi):
    m = (fq > lo) & (fq < hi) & np.all(np.abs(fq[:, None] - np.array(lines_all)) > 5, axis=1)
    return np.median(Wq[m])

print(f"{'window':<16}{'noise/ref 50-100 Hz':>20}{'500-1000 Hz':>13}{'contrast C5':>13}{'E5':>8}{'G5':>8}")
rows = {}
fig, ax = plt.subplots(figsize=(11, 4))
for win in ['boxcar', 'triang', 'hamming', 'hann', 'blackmanharris', 'nuttall', 'flattop']:
    fq, Wq = sp.welch(x, fs, window=win, nperseg=L)
    fl = median_filter(Wq, size=201, mode='nearest')
    con = [Wq[np.argmin(np.abs(fq - l))] / fl[np.argmin(np.abs(fq - l))] for l in lines]
    rows[win] = (level(fq, Wq, 50, 100) / level(fr, Wref, 50, 100), level(fq, Wq, 500, 1000) / level(fr, Wref, 500, 1000), *con)
    print(f"{win:<16}{rows[win][0]:20.2f}{rows[win][1]:13.2f}{con[0]:13.0f}{con[1]:8.0f}{con[2]:8.0f}")
    if win in ('boxcar', 'flattop', 'hann', 'nuttall'):
        m = (fq > 30) & (fq < 1000)
        ax.loglog(fq[m], Wq[m], lw=1, label=win)
ax.set(xlabel='frequency [Hz]', ylabel='PSD', title='Two wrong windows (boxcar, flattop) and two right ones (hann, nuttall)')
ax.legend()
savefig('b05_windows')
