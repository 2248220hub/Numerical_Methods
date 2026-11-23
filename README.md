# Comp Astro — an open, interactive study guide to numerical methods

Open `index.html` in a browser (double-click). Everything works offline: maths (KaTeX), fonts and
plots are in `assets/`. Hover any symbol of a formula to see its name, its meaning and the same
quantity in Python and Fortran.

## Folders
| folder | content |
|---|---|
| `04. Direct Linear Solvers/` | stability and cancellation, Cramer, Gaussian elimination and pivoting, PA = LU, Cholesky, Thomas |
| `05. Conditioning and SVD/` | norms, condition number, digits lost, mixed-precision refinement, EVD, SVD, pseudoinverse, least squares |
| `06. Sparse and Iterative Solvers/` | CSR/CSC/COO, fill-in, steepest descent, conjugate gradient, matrix-free map-making, preconditioning |
| `07. Discrete Signal Analysis/` | quantisation, sampling and aliasing, sinc interpolation, DTFT, DFT, windows |
| `08. The FFT/` | radix-2 FFT, bit reversal, FFTW from Fortran, 2D transforms, Fourier optics, notch filters, STFT |
| `09. Convolution and Correlation/` | convolution theorem, matched filter, wrap-around, overlap-save/add, Wiener and Tikhonov, filters |
| `10. Spectral Estimation/` | periodogram, Bartlett and Welch with χ² bands, windows, 1/f noise and the knee, prewhitening |
| `Lab 1/` | laboratory: a chord buried in noise — corrected notebook, figures, audio, block-by-block page |
| `11. Sampling on the Sphere/` | spherical harmonics, C_ℓ and cosmic variance, HEALPix and the ring FFT, beams, pixel windows, masks, the Lab 2 mind map |
| `Lab 2/` | laboratory: the CMB power spectrum of Planck 70/100 GHz maps — Colab notebook, figures, block-by-block page (the professor's `Codes/`, `data/` and `Solved Tasks/` stay as they are) |
| `12. Bayesian Inference/` | probability, Bayes, likelihood and Fisher matrix, priors, posterior summaries, grids |
| `13. Monte Carlo Methods/` | hit-or-miss, sample mean, curse of dimension, sampling, importance sampling |
| `Hyperspectral Project/` | one data cube analysed with every method: scripts, synthetic data, downloader, figures |
| `assets/` | shared style, site logic, plotting helpers, the course map (`assets/js/course.js`) |

Each lesson folder holds `index.html`, its widgets (`lessonNN.js`) and tested programs in `code/`.
Every lesson section ends with a link to the matching part of the hyperspectral project, and every
project part starts with a link back to the theory.

## Run the code
    pip install numpy scipy matplotlib
    cd "04. Direct Linear Solvers/code" && python direct_solvers.py
    gfortran -O2 lu_cholesky.f90 -llapack -lblas -o lu_cholesky && ./lu_cholesky
    cd "10. Spectral Estimation/code" && python spectral_estimation.py
    gfortran -O2 -fopenmp welch_omp.f90 -I/usr/include -lfftw3 -o welch_omp && ./welch_omp

    cd "Hyperspectral Project"
    python make_synthetic_cube.py          # scene with known truth
    python hyperspectral_L04_06_linear.py  # also L07, L08, L09, L10, L12, L13 — figures go to figures/
    python get_hyperspectral_data.py       # real scenes: Indian Pines (~6 MB), paviaU, salinas

    cd "Lab 1"                             # Jupyter or Google Colab
    jupyter notebook AstroSignal_corrected.ipynb

    pip install healpy pandas              # Linux, macOS, WSL or Colab (not native Windows)
    cd "11. Sampling on the Sphere/code" && python sphere_warmup.py
    gfortran -O2 ring_fft.f90 -I/usr/include -lfftw3 -o ring_fft && ./ring_fft   # after beam_to_map.py
    cd "Lab 2" && jupyter notebook Lab2_Planck_power_spectrum.ipynb             # needs Lab 2/data (113 MB)

FFTW: `sudo apt install libfftw3-dev` (Linux), `brew install fftw` (macOS).
LAPACK/BLAS (lessons 4–5): `sudo apt install liblapack-dev` (Linux), `brew install lapack` (macOS).

## Editing and adding lessons
The pages are generated from sources kept in `_source/` (`build.py`, `contentNN.html`, `home.html`;
for Lab 2 also `make_lab2_nb.py` (writes the notebook), `lab2_extract.py` (binned spectra and sky image
for the widgets, written to `lab2_data.js`) and `lab2_widgets.js` — `Lab 2/lab2.js` is the two joined):

    cd _source && python build.py            # all pages
    python build.py 07 HSI                   # only some

Source macros: `<x-f name tag>tex<note><py><f90></x-f>` (formula with code), `<x-code py="file.py" f90="file.f90"/>`
(code from the lesson's `code/` folder), `<x-ex id ans tol lvl>question<hint><sol></x-ex>` (checked exercise),
`<x-hsi to="sN">text</x-hsi>` (link to project part N), `<x-back href>text</x-back>` (link back to a lesson),
`<x-lab href>text</x-lab>` (link forward to a laboratory page).
In formulas `\h{key}{tex}` makes a hoverable symbol; `key` is looked up in the lesson's `SYMBOLS`.

A new lesson gets its own folder (`14. Markov chain Monte Carlo/`, …), a content file, a widgets file and an
entry in `assets/js/course.js` (`status: "ready"`, `folder`, `subs`). Progress, ticked checklists and solved
exercises are stored in your browser (localStorage).

## Publishing on GitHub Pages
All links are relative, so the site works from any sub-path: push the repository and enable Pages
(Settings → Pages → deploy from the main branch, root folder). The layout adapts to tablets and phones
(sidebar becomes a drawer behind the ☰ button). The Planck maps in `Lab 2/data/` are 48 MB each:
under GitHub's 100 MB limit, but you may prefer Git LFS or keeping `data/` out of the repository
(the Lab 2 page does not need it; only the notebook does).
