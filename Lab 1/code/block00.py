# Block 0 — Setup: libraries, data folder, helper functions
import os
import numpy as np
import scipy.signal as sp
from scipy.io import wavfile
from scipy.fft import rfft, irfft, rfftfreq
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

# Data: in Google Colab upload the four WAV files once; locally, keep them in ./data
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
os.makedirs('data', exist_ok=True)
NAMES = ['dati_rumorosi.wav', 'segnale.wav', 'dati_rumorosi_2.wav', 'segnale_2.wav']
if IN_COLAB and not all(os.path.exists(os.path.join('data', n)) for n in NAMES):
    from google.colab import files
    for fn in files.upload():                      # select the four .wav files
        os.replace(fn, os.path.join('data', fn))

os.makedirs('figures', exist_ok=True)              # every figure is also saved as PNG
os.makedirs('out_wav', exist_ok=True)              # filtered recordings
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3})

HUM = 50.0                                         # mains hum [Hz]: instrumental, NOT signal

def read_wav(name):
    # int16 -> float64: sums and squares of int16 overflow at 32767
    fs, x = wavfile.read(os.path.join('data', name))
    return fs, x.astype(np.float64)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join('figures', name + '.png'), dpi=110)
    plt.show()

def snr_db(y, s):
    # signal-to-noise ratio of a filtered signal y against the clean signal s
    return 10 * np.log10(np.var(s) / np.var(y - s))

def note_name(f):
    # nearest equal-tempered note and its offset in cents (A4 = 440 Hz)
    n = 12 * np.log2(f / 440.0)
    k = int(np.round(n))
    names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    octave = 4 + (k + 9) // 12
    return f"{names[k % 12]}{octave} ({100 * (n - k):+.0f} cents)"

def write_wav(name, y, fs):
    # scale to full scale (arbitrary units) and save as 16-bit
    wavfile.write(os.path.join('out_wav', name), fs, np.int16(y / np.abs(y).max() * 32767))

print('Colab' if IN_COLAB else 'local', '| files:', sorted(os.listdir('data')))
