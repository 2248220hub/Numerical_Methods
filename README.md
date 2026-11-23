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

FFTW: `sudo apt install libfftw3-dev` (Linux), `brew install fftw` (macOS).
LAPACK/BLAS (lessons 4–5): `sudo apt install liblapack-dev` (Linux), `brew install lapack` (macOS).

## Editing and adding lessons
The pages are generated from sources kept in `_source/` (`build.py`, `contentNN.html`, `home.html`):

    cd _source && python build.py            # all pages
    python build.py 07 HSI                   # only some

Source macros: `<x-f name tag>tex<note><py><f90></x-f>` (formula with code), `<x-code py="file.py" f90="file.f90"/>`
(code from the lesson's `code/` folder), `<x-ex id ans tol lvl>question<hint><sol></x-ex>` (checked exercise),
`<x-hsi to="sN">text</x-hsi>` (link to project part N), `<x-back href>text</x-back>` (link back to a lesson).
In formulas `\h{key}{tex}` makes a hoverable symbol; `key` is looked up in the lesson's `SYMBOLS`.

A new lesson gets its own folder (`11. Sampling on the Sphere/`, …), a content file, a widgets file and an
entry in `assets/js/course.js` (`status: "ready"`, `folder`, `subs`). Progress, ticked checklists and solved
exercises are stored in your browser (localStorage).
