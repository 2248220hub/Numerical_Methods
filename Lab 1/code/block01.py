# Block 1 — Read the recordings and the clean references
fs, x = read_wav('dati_rumorosi.wav')     # recording 1 (noisy)
fs_s, s = read_wav('segnale.wav')          # its clean signal (only for checking)
assert fs_s == fs                          # do not overwrite fs with the second read
N = x.size
T = N / fs
t = np.arange(N) / fs                      # sample times n/fs
z = (t > 60) & (t < 60.02)                 # a 20 ms window used for every comparison
t_ms = (t[z] - 60) * 1e3

print(f"{'quantity':<28}{'value':>16}")
print('-' * 44)
for k, v in [('sampling rate fs [Hz]', fs), ('samples N', N), ('duration T [s]', T),
             ('resolution 1/T [Hz]', f'{1 / T:.4f}'), ('rms of the data', f'{x.std():.0f}'),
             ('peak |x| (int16 max 32767)', f'{np.abs(x).max():.0f}'),          # max() needs ()
             ('rms of the clean signal', f'{s.std():.1f}'),
             ('signal / noise power [dB]', f'{10 * np.log10(s.var() / (x - s).var()):.1f}')]:
    print(f"{k:<28}{v:>16}")
