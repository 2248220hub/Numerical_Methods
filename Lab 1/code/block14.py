# Block 14 — Summary of all filters on recording 1 (table, plots, listening)
print(f"{'filter':<40}{'bins kept':>10}{'SNR [dB]':>10}{'residual rms':>14}")
print('-' * 74)
print(f"{'no filter (the data)':<40}{N//2+1:>10}{snr_db(x - x.mean(), s):10.1f}{np.std(x - s):14.1f}")
for name, (H, y) in results.items():
    print(f"{name:<40}{int(np.count_nonzero(H)):>10}{snr_db(y, s):10.1f}{np.std(y - s):14.1f}")

show = ['band-pass 493–814 Hz', 'comb, 2.5 Hz teeth', 'Wiener, naive (all bins)', 'Wiener, detected lines (hum vetoed)']
fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
for ax, name in zip(axes.flat, show):
    y = results[name][1]
    ax.plot(t_ms, s[z], color='crimson', lw=2, label='clean signal')
    ax.plot(t_ms, y[z], lw=1.1, label='filtered data')
    ax.set_title(f'{name}: SNR {snr_db(y, s):.1f} dB'); ax.legend(fontsize=7, loc='upper right')
for ax in axes[1]: ax.set_xlabel('time [ms] after t = 60 s')
savefig('b14_filters_2x2')

plt.figure(figsize=(9, 3.6))
names = ['data'] + list(results)
vals = [snr_db(x - x.mean(), s)] + [snr_db(results[k][1], s) for k in results]
plt.barh(names, vals, color=['0.6'] + ['tab:blue'] * len(results)); plt.axvline(0, color='k', lw=0.8)
plt.xlabel('SNR against the clean signal [dB]'); plt.title('Recording 1: which filter brings the signal back?')
savefig('b14_snr_bar')

# save full recordings and 5 s clips (60–65 s) to listen to
clip = slice(int(60 * fs), int(65 * fs))
write_wav('rec1_noisy.wav', x - x.mean(), fs); write_wav('rec1_clean.wav', s, fs)
for k, name in enumerate(show, 1):
    write_wav(f'rec1_filter{k}.wav', results[name][1], fs)
if IN_COLAB or True:
    from IPython.display import Audio, display
    for title, y in [('noisy data', x - x.mean()), ('clean signal', s)] + [(n, results[n][1]) for n in show]:
        print(title); display(Audio(y[clip], rate=fs, normalize=True))   # only 5 s: a light notebook
