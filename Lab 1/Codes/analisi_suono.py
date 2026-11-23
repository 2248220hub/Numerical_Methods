import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import rfftfreq, rfft, irfft
import scipy.signal as sp

#imbroglio: per farvi vedere i grafici uso le frequenze
f1 = 523.3  # Do, Hz, may be float
f2 = 659.3  # sine frequency, Hz, may be float
f3 = 784.0  # sine frequency, Hz, may be float

#leggo il file wav - lo leggo in formato int16, devo convertirlo in floiar
sample_rate, sound = wavfile.read('dati_rumorosi.wav')
myduration = (sound.size-1)/sample_rate  # in seconds, may be float
sound=sound.astype(float)

ndata=sound.size


#ne stimo il pewriodogramma
fp, Pxx_den = sp.periodogram(sound, fs=sample_rate)

plt.loglog(fp[1:], Pxx_den[1:])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.ylim(1.0e-5,1.0e11)
plt.title('log Periodogram of noisy data')
plt.savefig("LogPeriodogram.pdf")
plt.savefig("LogPeriodogram.png")
plt.show()
plt.close()

# provo a stimarne lo spettro con un metodo di Welch e una finestra
# di lunggezza 1/100 dei dati
windowsize=ndata//100

(f, S1)= sp.welch(sound, window='nuttall', fs=sample_rate, nperseg=windowsize)

#loglog plot
plt.loglog(f[1:],S1[1:])
plt.axvline(x=f1,color="r",linestyle=":")
plt.axvline(x=f2,color="r",linestyle=":")
plt.axvline(x=f3,color="r",linestyle=":")
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('PSD estimate of noisy data')
plt.savefig("LogLogPowerSpec.pdf")
plt.savefig("LogLogPowerSpec.png")
plt.show()
plt.close()

plt.loglog(fp[1:], Pxx_den[1:],alpha=0.5,color="green")
plt.loglog(f[1:],S1[1:],color="red")
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.ylim(1.0e-5,1.0e11)
plt.title('log Periodogram and PSD estimate')
plt.savefig("LogPeriodogram_PSD.pdf")
plt.savefig("LogPeriodogram_PSD.png")
plt.show()
plt.close()


fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 10), gridspec_kw={'hspace': 0})  # rapporto 3:4

# Primo grafico
axes[0].loglog(fp[1:], Pxx_den[1:])
axes[0].set_ylabel('PSD [V^2/Hz]')
#axes[0].set_xlim(x_min, x_max)
axes[0].text(0.05, 0.1, 'Periodogramma', transform=axes[0].transAxes, fontsize=12, fontweight='bold')
axes[0].set_ylim(1.0e-5,1.0e11)
axes[0].grid(True, which='both')

# Secondo grafico
axes[1].loglog(fp[1:], Pxx_den[1:],alpha=0.5)
axes[1].loglog(f[1:],S1[1:],color="red")
axes[1].set_ylabel('PSD [V^2/Hz]')
#axes[1].set_xlim(x_min, x_max)
axes[1].text(0.05, 0.1, 'Periodogramma & Welch PSD', transform=axes[1].transAxes, fontsize=12, fontweight='bold')
axes[1].set_ylim(1.0e-5,1.0e11)
axes[1].grid(True, which='both')

# Terzo grafico
axes[2].loglog(f[1:],S1[1:],color="red")
axes[2].set_ylabel('PSD [V^2/Hz]')
axes[2].set_xlabel('frequency [Hz]')
#axes[2].set_xlim(x_min, x_max)
#axes[2].set_ylim(1.0e-5,1.0e11)
axes[2].text(0.05, 0.1, 'Stima spettrale di Welch (Nuttall, Nd=1/100)', transform=axes[2].transAxes, fontsize=12, fontweight='bold')
axes[2].grid(True, which='both')

plt.tight_layout()
plt.savefig('tutti_loglog_4-5.png', dpi=300)  # alta risoluzione
plt.savefig('tutti_loglog_4-5.pdf', dpi=300)  # alta risoluzione
plt.show()



#ho individuato il segnale alle tre frequenze (è un accordo di Do maggiore)
freqs = rfftfreq(ndata, 1.0/sample_rate)

# preparo un passa-banda barbaro: zero tutte le frequenze lontane dai tre picchi,
# ritaglio una banda di 5Hz intorno ai picchi
df=2.5
aa=np.logical_or(freqs<f1-df,freqs>f3+df)
bb=np.logical_and(freqs>f1+df,freqs<f2-df)
cc=np.logical_and(freqs>f2+df,freqs<f3-df)

f_sound=rfft(sound)
f_sound[aa]=0
f_sound[bb]=0
f_sound[cc]=0

clean=irfft(f_sound)

bits=np.int16(clean/ max(abs(clean)) * (2**15 - 1)) 
wavfile.write('ripuliti.wav', sample_rate, bits)
