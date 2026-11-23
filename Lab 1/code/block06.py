# Block 6 — How long should the segments be? (table)
print(f"{'L':>9} {'duration [s]':>13} {'bin [Hz]':>9} {'K':>5} {'contrast C5':>12} {'E5':>8} {'G5':>8}")
Ls, cons = [], []
for q in [12, 14, 16, 18, 20]:
    Lq = 2**q
    fq, Wq = sp.welch(x, fs, window='nuttall', nperseg=Lq)
    fl = median_filter(Wq, size=max(21, int(135 / (fs / Lq)) | 1), mode='nearest')
    con = [Wq[np.argmin(np.abs(fq - l))] / fl[np.argmin(np.abs(fq - l))] for l in lines]
    Ls.append(Lq); cons.append(con)
    print(f"{Lq:9d} {Lq/fs:13.2f} {fs/Lq:9.3f} {(N - Lq)//(Lq//2) + 1:5d} {con[0]:12.0f} {con[1]:8.0f} {con[2]:8.0f}")
plt.figure(figsize=(6, 3.6))
plt.loglog(Ls, np.array(cons), 'o-')
plt.loglog(Ls, cons[0][0] * np.array(Ls) / Ls[0], 'k--', lw=1, label='∝ L')
plt.xlabel('segment length L [samples]'); plt.ylabel('line contrast'); plt.legend(['C5', 'E5', 'G5', '∝ L'])
plt.title('A line gains contrast in proportion to L')
savefig('b06_contrast_vs_L')
