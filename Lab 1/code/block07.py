# Block 7 — Is the signal stationary? Spectrogram and amplitude in 10 s blocks
fq, tt, Z = sp.stft(x, fs, window='hann', nperseg=fs, noverlap=fs // 2)   # 1 s window
band = (fq > 30) & (fq < 850)
plt.figure(figsize=(11, 3.8))
plt.pcolormesh(tt, fq[band], 10 * np.log10(np.abs(Z[band])**2 + 1e-12), shading='auto', cmap='magma')
plt.colorbar(label='power [dB]')
plt.xlabel('time [s]'); plt.ylabel('frequency [Hz]'); plt.title('Spectrogram, 1 s Hann window: four steady ridges')
savefig('b07_spectrogram')

print(f"{'line [Hz]':>10} {'mean amplitude':>15} {'std':>7} {'std/mean':>9}")
blk = int(10 * fs)
for fl in lines_all:
    amps = [2 * np.abs(np.sum(x[i:i+blk] * np.exp(-2j * np.pi * fl * t[i:i+blk]))) / blk for i in range(0, N - blk + 1, blk)]
    print(f"{fl:10.1f} {np.mean(amps):15.1f} {np.std(amps):7.1f} {np.std(amps)/np.mean(amps):9.1%}")
