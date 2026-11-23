# Block 4 — Find the lines automatically and name them
floor = median_filter(W, size=201, mode='nearest')             # local noise level (135 Hz)
peaks, _ = sp.find_peaks(W / floor, height=20)
peaks = peaks[fw[peaks] > 20]
peaks = np.union1d(peaks, [np.argmin(np.abs(fw - HUM))])     # the hum sits on a steep slope: add it
peaks = peaks[np.argsort(fw[peaks])]
print(f"{'f [Hz]':>9} {'Welch/floor':>12} {'periodogram contrast':>21}   nature")
for k in peaks:
    m = (fp > fw[k] - 1) & (fp < fw[k] + 1)
    jj = np.flatnonzero(m)[np.argmax(P[m])]                    # exact periodogram bin (0.008 Hz)
    ring = (np.abs(fp - fp[jj]) > 2) & (np.abs(fp - fp[jj]) < 20)
    contrast_p = P[jj] / (np.median(P[ring]) / np.log(2))      # median of an exponential = ln2 x mean
    kind = 'mains hum (instrument)' if abs(fw[k] - HUM) < 1 else note_name(fp[jj])
    print(f"{fp[jj]:9.2f} {W[k] / floor[k]:12.0f} {contrast_p:21.0f}   {kind}")
lines = [523.3, 659.3, 784.0]                                  # C5, E5, G5: a C major chord
lines_all = [HUM] + lines
