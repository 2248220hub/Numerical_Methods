/* ==========================================================================
   course.js — the course map. ONE place to edit when a new lesson arrives.
   status: "ready" (folder exists)  |  "soon" (placeholder, greyed out)
   folder: the folder name inside "Main Website" (must match exactly)
   label : optional prefix shown instead of "<id>." (labs, project)
   ========================================================================== */
window.COURSE = {
  title: "Computational Methods in Astrophysics",
  short: ["Comp", "Astro"],
  credit: "Based on the lecture notes of G. de Gasperis and L. Graziani (Sapienza University of Rome)",
  units: [
    { id: "01", name: "Fortran I: from C to Fortran", part: "Programming", color: "#3ea255", status: "soon" },
    { id: "02", name: "Fortran II", part: "Programming", color: "#3ea255", status: "soon" },
    { id: "03", name: "Fortran III", part: "Programming", color: "#3ea255", status: "soon" },
    {
      id: "04", name: "Direct Linear Solvers", part: "Linear algebra", color: "#7a4fc9", status: "ready",
      folder: "04. Direct Linear Solvers",
      blurb: "Stability and catastrophic cancellation, Cramer's rule, Gaussian elimination and pivoting, PA = LU, Cholesky, banded solvers.",
      subs: ["Accuracy & stability", "Cramer's rule", "Back substitution", "Elimination & pivoting",
             "LU factorisation", "Cholesky", "Matching structure", "Code lab"]
    },
    {
      id: "05", name: "Conditioning, EVD and SVD", part: "Linear algebra", color: "#7a4fc9", status: "ready",
      folder: "05. Conditioning and SVD",
      blurb: "Vector and matrix norms, the condition number, iterative refinement in mixed precision, eigen- and singular value decompositions, the pseudoinverse and least squares.",
      subs: ["Norms", "Condition number", "Digits lost", "Iterative refinement",
             "Eigendecomposition", "SVD", "Pseudoinverse & least squares", "Code lab"]
    },
    {
      id: "06", name: "Sparse and Iterative Solvers", part: "Linear algebra", color: "#7a4fc9", status: "ready",
      folder: "06. Sparse and Iterative Solvers",
      blurb: "CSR, CSC and COO storage, fill-in, steepest descent and conjugate gradient, matrix-free map-making, preconditioning, GMRES and LSQR.",
      subs: ["Sparse storage", "Fill-in & SpMV", "Steepest descent", "Conjugate gradient",
             "Matrix-free CG", "Preconditioning", "Code lab"]
    },
    {
      id: "07", name: "Discrete Signal Analysis", part: "Signals", color: "#2f84cb", status: "ready",
      folder: "07. Discrete Signal Analysis",
      blurb: "Quantisation and SQNR, sampling and aliasing, sinc reconstruction, DTFT and DFT, windows and leakage, the O(N²) cost.",
      subs: ["From continuous to discrete", "Quantisation & SQNR", "Sampling & aliasing", "Sinc reconstruction",
             "The Fourier family & DTFT", "The DFT", "Windows & leakage", "Code lab"]
    },
    {
      id: "08", name: "The FFT", part: "Signals", color: "#2f84cb", status: "ready",
      folder: "08. The FFT",
      blurb: "Radix-2 Cooley–Tukey, butterflies and bit reversal, FFTW, 2D transforms, magnitude vs phase, notch filters, the STFT.",
      subs: ["Divide and conquer", "Butterflies & bit reversal", "Implementations & cost", "The 2D DFT",
             "Fourier optics & 2D theorems", "Phase & notch filtering", "Time–frequency: the STFT", "Code lab"]
    },
    {
      id: "09", name: "Convolution and Correlation", part: "Signals", color: "#2a9d8f", status: "ready",
      folder: "09. Convolution and Correlation",
      blurb: "The convolution theorem, wrap-around and zero padding, overlap-save and overlap-add, deconvolution and the Wiener filter.",
      subs: ["Convolution & the theorem", "Correlation & matched filter", "Wrap-around & zero padding", "Overlap-save & overlap-add",
             "Deconvolution & Wiener", "Digital filters & delay", "Convolution as linear algebra", "Code lab"]
    },
    {
      id: "10", name: "Spectral Estimation", part: "Signals", color: "#2a9d8f", status: "ready",
      folder: "10. Spectral Estimation",
      blurb: "The periodogram and why it fails, Bartlett and Welch with χ² bands, windows, the Wiener filter, coloured noise and the knee, spectrograms.",
      subs: ["The PSD problem", "The periodogram", "Bartlett & Welch", "Windows: resolution vs leakage",
             "The Wiener filter", "Coloured noise & the knee", "Steep spectra & prewhitening", "Time–frequency trade-off", "Code lab"]
    },
    {
      id: "L1", label: "Lab 1 ·", name: "Discrete signal laboratory", part: "Laboratory", color: "#e6824c", status: "ready",
      folder: "Lab 1",
      blurb: "Two noisy recordings: periodogram and Welch, spectrograms, band-pass, comb and Wiener filters, noise fit and Whittle — in Python.",
      subs: ["Objective & data", "Read, listen, look", "Periodogram & Welch", "Stationarity: STFT", "Band-pass & comb",
             "Fitting the noise", "Wiener filters", "Whittle fit", "Recording 2: the beat", "Results & conclusions"]
    },
    {
      id: "11", name: "Sampling on the Sphere", part: "Signals", color: "#5b6a82", status: "ready",
      folder: "11. Sampling on the Sphere",
      blurb: "Signals on the sky: spherical harmonics, C_ℓ and cosmic variance, HEALPix and its ring FFT, beams and pixel windows, masks — and a mind map of Lab 2.",
      subs: ["No perfect grid", "Circle → sphere", "Spherical harmonics", "C_ℓ & cosmic variance", "HEALPix",
             "Fast transform on rings", "Beams, pixels, N_side", "Recovering C_ℓ, cut sky", "Lab 2 mind map", "Exercises & code lab"]
    },
    {
      id: "L2", label: "Lab 2 ·", name: "Planck power spectrum", part: "Laboratory", color: "#e6824c", status: "ready",
      folder: "Lab 2",
      blurb: "Real Planck 70 and 100 GHz maps: mask the Galaxy, remove the noise bias, correct beam and pixels, recover the CMB acoustic peaks.",
      subs: ["Objective & data", "Step 0: look at the sky", "Step 1: full-sky spectrum", "Step 2: cut the Galaxy",
             "Step 3: noise bias", "Step 4: beam & pixels", "Extensions A–C", "Results & significance"]
    },
    {
      id: "12", name: "Bayesian Inference", part: "Inference", color: "#7a61cc", status: "ready",
      folder: "12. Bayesian Inference",
      blurb: "Probability, Bayes' theorem, likelihood & MLE, Fisher errors, priors, credible intervals, the posterior on a grid.",
      subs: ["What is probability?", "Bayes' theorem", "Likelihood & MLE", "Fisher matrix & confidence",
             "Posterior & priors", "Summarising the posterior", "The grid & its cost", "Code lab"]
    },
    {
      id: "13", name: "Monte Carlo Methods", part: "Inference", color: "#d1495b", status: "ready",
      folder: "13. Monte Carlo Methods",
      blurb: "Hit-or-miss and sample-mean integration, the 1/√N law, sampling (inversion, Box–Muller, rejection), importance sampling, ESS.",
      subs: ["The Monte Carlo idea", "Sample mean & its error", "Grid or Monte Carlo?", "Sampling a distribution",
             "Importance sampling", "Posteriors & the ESS", "Code lab"]
    },
    { id: "14", name: "Markov chain Monte Carlo", part: "Inference", color: "#d1495b", status: "soon" },
    {
      id: "HSI", label: "★", name: "Hyperspectral Project", part: "Project", color: "#e6824c", status: "ready",
      folder: "Hyperspectral Project",
      blurb: "Every method of the course applied to hyperspectral images: band sampling, 2D FFT destriping, sensor response, noise spectra, Bayesian classification and unmixing, Monte Carlo.",
      subs: ["Overview & data", "Linear algebra on the cube", "Sampling the spectrum", "2D FFT: destriping", "Convolution: sensor response",
             "Noise spectra of a cube", "Bayes classifier", "Unmixing posterior", "Monte Carlo on the cube"]
    }
  ]
};
