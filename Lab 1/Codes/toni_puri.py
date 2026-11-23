"""Generatore dei dati dell'Esercitazione 1 (versione corretta, 28/10/2026).

Scrive, per ciascuna registrazione, il segnale pulito e il segnale + rumore:
  registrazione 1: accordo di Do maggiore (523.3, 659.3, 784.0 Hz) -> segnale.wav, dati_rumorosi.wav
  registrazione 2: 440 + 442 Hz (battimento a 2 Hz) + 880 Hz      -> segnale_2.wav, dati_rumorosi_2.wav
Rumore con densita' spettrale di potenza S(f) = 1 + (fk/f)^alpha, fk = 440 Hz, alpha = 2,
piu' il ronzio di rete a 50 Hz (sistematica strumentale, NON segnale: non compare in segnale*.wav).

Correzioni rispetto alla versione precedente (marcate CORRETTO nel codice):
 1. one_over_f() e' lo spettro di POTENZA: la deviazione standard di Re e Im di ogni
    coefficiente di Fourier e' la sua radice quadrata (prima si usava lo spettro stesso,
    e la potenza risultava [1 + (fk/f)^alpha]^2);
 2. alpha = 2;
 3. asse dei tempi t = arange(N)/fs (linspace(0, T, N) ha passo T/(N-1), non 1/fs);
 4. seme fisso: i dati sono riproducibili;
 5. niente divisione per zero a f = 0; il bin di Nyquist (N pari) e' reale;
 6. segnale pulito e dati rumorosi scritti con lo STESSO fattore di scala: il file pulito e'
    esattamente il segnale contenuto nei dati (prima ciascuno era normalizzato al proprio massimo).
"""
import numpy as np
from scipy.io import wavfile
from scipy.fft import rfftfreq, irfft


def one_over_f(f, fknee, alpha=2.0, scale=1.0):
    """Spettro di POTENZA del rumore: scale * (1 + (fknee/f)^alpha)."""
    return scale * (1.0 + (fknee / f) ** alpha)


def rumore(ndata, sample_rate, fk, alpha, rng):
    """Rumore gaussiano con densita' spettrale proporzionale a one_over_f (metodo di Timmer & Koenig)."""
    freqs = rfftfreq(ndata, 1.0 / sample_rate)
    S = np.zeros(freqs.size)
    S[1:] = one_over_f(freqs[1:], fk, alpha)              # CORRETTO (5): niente f = 0
    sigma = np.sqrt(S / 2)                                 # CORRETTO (1): ampiezza = sqrt(potenza)
    spectrum = sigma * (rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size))
    spectrum[0] = 0                                        # media nulla
    if ndata % 2 == 0:                                     # CORRETTO (5): Nyquist reale
        spectrum[-1] = np.sqrt(S[-1]) * rng.standard_normal()
    return irfft(spectrum, ndata)


def scrivi(nome, x, scala, sample_rate):
    y = x * scala
    if np.max(np.abs(y)) > 2**15 - 1:
        raise ValueError(f"{nome}: clipping")
    wavfile.write(nome, sample_rate, np.int16(np.round(y)))


sample_rate = 44100   # Hz
duration = 120        # s
ndata = sample_rate * duration
t = np.arange(ndata) / sample_rate                         # CORRETTO (3)
fk = 440.0
alpha = 2.0                                                # CORRETTO (2)
ampiezza_rumore = 1.0e4                                    # come nella versione precedente
ronzio = {50.0: 2.0}                                       # rete elettrica: frequenza -> ampiezza

registrazioni = {
    "": ([523.3, 659.3, 784.0], 2028),                     # Do5, Mi5, Sol5; seme (con 2026 il rumore
                                                           # supera per caso la soglia di rivelazione a 7963 Hz)
    "_2": ([440.0, 442.0, 880.0], 2027),                   # La4, La4 + 8 cent (battimento), La5
}

for suffisso, (toni, seme) in registrazioni.items():
    rng = np.random.default_rng(seme)                      # CORRETTO (4)
    somma = sum(np.sin(2 * np.pi * f0 * t) for f0 in toni)
    rete = sum(a * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi)) for f0, a in ronzio.items())
    dati = somma + rete + ampiezza_rumore * rumore(ndata, sample_rate, fk, alpha, rng)
    scala = 0.9 * (2**15 - 1) / np.max(np.abs(dati))       # CORRETTO (6): un'unica scala, 10% di margine
    scrivi(f"segnale{suffisso}.wav", somma, scala, sample_rate)
    scrivi(f"dati_rumorosi{suffisso}.wav", dati, scala, sample_rate)
    print(f"dati_rumorosi{suffisso}.wav: toni {toni} Hz, seme {seme}, scala {scala:.4g}, "
          f"rms {np.std(dati*scala):.0f}, picco {np.max(np.abs(dati*scala)):.0f}")
