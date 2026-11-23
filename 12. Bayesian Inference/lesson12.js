/* ==========================================================================
   Lesson 12 — Bayesian inference: hover-symbols and interactive widgets
   ========================================================================== */
window.SYMBOLS = {
  Sf: { n: "S(f) — power spectral density", t: "model", s: "How much noise power sits at frequency f (units²/Hz). For the knee model: white level σ² plus a 1/f<sup>α</sup> rise below f<sub>k</sub>.", py: "S = s2 * (1 + (fk / f) ** alpha)", f: "S = s2*(1.0_dp + (fk/f)**alpha)" },
  s2: { n: "σ² — white-noise level", t: "parameter", s: "The flat part of the spectrum, measured above the knee. Truth 10⁻³. We fit log₁₀ σ² because it is positive and spans decades.", py: "s2 = 10 ** log10_s2", f: "s2 = 10.0_dp**log10_s2" },
  fk: { n: "f<sub>k</sub> — knee frequency", t: "parameter", s: "Where 1/f noise equals white noise. Truth 1 Hz. Below it, low-frequency drifts dominate. Its posterior is skewed in f<sub>k</sub>, nearly Gaussian in log₁₀ f<sub>k</sub>.", py: "fk = 10 ** log10_fk", f: "fk = 10.0_dp**log10_fk" },
  f: { n: "f — frequency", t: "variable", s: "Fourier frequency of the periodogram bin: f<sub>j</sub> = j/T for j = 1 … N/2−1 (DC and Nyquist excluded).", py: "f = np.arange(1, N // 2) / T", f: "f(j) = real(j,dp)/T" },
  alpha: { n: "α — 1/f slope", t: "parameter", s: "How steeply power rises below the knee: S ∝ f<sup>−α</sup>. α = 1 is “pink” noise. Anti-correlated with f<sub>k</sub> (degeneracy).", py: "alpha = theta[2]", f: "alpha = theta(3)" },
  P: { n: "P(E) — probability of event E", t: "number in [0,1]", s: "Classical: favourable/total. Frequentist: long-run frequency. Bayesian: degree of belief given information. Same axioms for all three.", py: "P_E = n_fav / n_tot", f: "P_E = real(nfav,dp)/ntot" },
  Nfav: { n: "N<sub>fav</sub> — favourable cases", t: "count", s: "Number of equally likely outcomes in which the event happens.", py: "n_fav = sum(1 for o in outcomes if event(o))", f: "nfav = count(event_mask)" },
  Ntot: { n: "N<sub>tot</sub> — all possible cases", t: "count", s: "Size of the (finite) sample space, all assumed equally likely — the circular part of Laplace's definition.", py: "n_tot = len(outcomes)", f: "ntot = size(outcomes)" },
  lim: { n: "lim<sub>N→∞</sub> — limit", t: "operator", s: "What the relative frequency tends to with infinitely many repetitions. In practice we can only approach it; the deviation shrinks like 1/√N.", py: "freq = np.cumsum(hit) / np.arange(1, n + 1)", f: "freq(k) = real(count(hit(1:k)),dp)/k" },
  nE: { n: "n<sub>E</sub> — number of occurrences", t: "count", s: "How many of the N trials produced event E. Binomially distributed: mean Np, variance Np(1−p).", py: "nE = np.count_nonzero(trials)", f: "nE = count(trials)" },
  sdfreq: { n: "σ(n<sub>E</sub>/N) — spread of a frequency", t: "standard error", s: "Typical distance between the observed frequency and the true p. Shrinks as 1/√N: the law of large numbers in numbers.", py: "np.sqrt(p * (1 - p) / N)", f: "sqrt(p*(1.0_dp-p)/N)" },
  p: { n: "p — probability of success", t: "parameter", s: "The (true) probability of the event in one trial — here the quantity a frequency estimates.", py: "p_hat = nE / N", f: "p_hat = real(nE,dp)/N" },
  Omega: { n: "Ω — sample space", t: "set", s: "The set of all possible outcomes. Normalisation: P(Ω) = 1, something must happen.", py: "omega = [(a, b) for a in range(1, 7) for b in range(1, 7)]", f: "! all outcomes enumerated in nested DO loops" },
  Ei: { n: "E<sub>i</sub> — events", t: "subsets of Ω", s: "Pairwise disjoint events (no two can happen together). Their probabilities add — countably many of them, for continuous variables.", py: "P_union = sum(P[i] for i in disjoint)", f: "P_union = sum(P(disjoint))" },
  PAB: { n: "P(A | B) — conditional probability", t: "probability", s: "Probability of A once we know B happened: restrict the world to B. In Bayes' theorem: the quantity we want (the posterior).", py: "p_A_given_B = p_AB / p_B", f: "pAgB = pAB / pB" },
  cap: { n: "P(A ∩ B) — joint probability", t: "probability", s: "Probability that A and B happen together. Product rule: P(A|B)P(B) = P(B|A)P(A).", py: "p_AB = p_A_given_B * p_B", f: "pAB = pAgB * pB" },
  PB: { n: "P(B) — probability of the data / evidence", t: "normalisation", s: "Total probability of what we observed, summed over all hypotheses. It normalises the posterior.", py: "p_B = np.sum(like * prior)", f: "pB = sum(like*prior)" },
  PBA: { n: "P(B | A) — likelihood", t: "probability", s: "Probability of the observed effect B if the cause A is true. Usually easy to model (e.g. 0.3 for a blue ball from urn 1).", py: "like = np.array([0.3, 0.6])", f: "like = [0.3_dp, 0.6_dp]" },
  PA: { n: "P(A) — prior", t: "probability", s: "Probability of A before seeing B.", py: "prior = np.array([0.5, 0.5])", f: "prior = [0.5_dp, 0.5_dp]" },
  PBAi: { n: "P(B | A<sub>i</sub>) — likelihood of each hypothesis", t: "array", s: "One number per member of the partition.", py: "like[i]", f: "like(i)" },
  PAi: { n: "P(A<sub>i</sub>) — prior of each hypothesis", t: "array", s: "The A<sub>i</sub> form a partition: exactly one is true, so these priors sum to 1.", py: "prior[i]", f: "prior(i)" },
  post: { n: "p(θ | x) — posterior", t: "density", s: "What we know about the parameters after the data: a full probability distribution. All answers (best values, intervals, correlations) are summaries of it.", py: "post = np.exp(lp - lp.max()); post /= post.sum()", f: "post = exp(lp - maxval(lp)); post = post/sum(post)" },
  like: { n: "p(x | θ) — likelihood", t: "density in x", s: "Probability of the observed data if the parameters were θ. Read as a function of θ it is 𝓛(θ): not a density in θ.", py: "lnL = -neg_log_like(theta, f, P)", f: "lnL = -sum(log(S) + P/S)" },
  prior: { n: "p(θ) = π(θ) — prior", t: "density", s: "What we knew (or assume) about θ before the data. Flat priors are flat in one variable only.", py: "log_prior = np.where(in_box, 0.0, -np.inf)", f: "ln_prior = merge(0.0_dp, -huge(1.0_dp), in_box)" },
  pi: { n: "π(θ) — prior", t: "density", s: "Same as p(θ). Ignorance has no unique prior: choose it, state it, test the sensitivity.", py: "log_prior(theta)", f: "ln_prior(theta)" },
  evid: { n: "p(x) — evidence (marginal likelihood)", t: "number", s: "∫ 𝓛(θ)π(θ) dθ: normalises the posterior. Irrelevant for parameter estimation, central for model comparison (Bayes factor).", py: "lnZ = logsumexp(lnL + ln_prior) + np.log(dtheta)", f: "lnZ = maxval(lp) + log(sum(exp(lp-maxval(lp)))*dtheta)" },
  PRA: { n: "P(real | alarm)", t: "posterior", s: "Probability the event is real once the alarm fired. Dominated by the base rate when real events are rare.", py: "sens*prev / (sens*prev + fa*(1-prev))", f: "sens*prev/(sens*prev + fa*(1.0_dp-prev))" },
  sens: { n: "P(alarm | real) — sensitivity", t: "likelihood", s: "Fraction of real events that trigger the alarm (true-positive rate).", py: "sens = 0.99", f: "sens = 0.99_dp" },
  prev: { n: "P(real) — prevalence / base rate", t: "prior", s: "Fraction of candidates that are real before any test. The part people forget.", py: "prev = 0.01", f: "prev = 0.01_dp" },
  fa: { n: "P(alarm | noise) — false-alarm rate", t: "likelihood", s: "Fraction of non-events that still trigger the alarm (false-positive rate).", py: "fa = 0.05", f: "fa = 0.05_dp" },
  L: { n: "𝓛(θ) — likelihood function", t: "function of θ", s: "p(x | θ) at fixed data, as a function of θ. Can be rescaled by any constant; invariant under re-parametrisation (no Jacobian).", py: "L = np.exp(lnL - lnL.max())   # rescaled", f: "L = exp(lnL - maxval(lnL))" },
  thetahat: { n: "θ̂ — estimate (MLE or mode)", t: "estimator", s: "The value maximising the likelihood (or posterior). A random variable: another record gives another θ̂.", py: "theta_hat = minimize(nll, x0, args=(f, P)).x", f: "theta_hat = grid(maxloc(lnL,1))" },
  argmax: { n: "arg max — the argument of the maximum", t: "operator", s: "Not the maximum value, but the θ where it is reached.", py: "theta[np.argmax(lnL)]", f: "theta(maxloc(lnL, dim=1))" },
  argmin: { n: "arg min", t: "operator", s: "The θ minimising a function — optimisers minimise, so we minimise −ln 𝓛.", py: "theta[np.argmin(nll)]", f: "theta(minloc(nll, dim=1))" },
  lnL: { n: "ln 𝓛 — log-likelihood", t: "function", s: "Sum of log-probabilities of the data. Products → sums, no underflow. −ln 𝓛 for Whittle: Σ(ln S + P/S).", py: "lnL = np.sum(np.log(p_i))", f: "lnL = sum(log(p_i))" },
  sigmai: { n: "σ<sub>i</sub> — error of datum i", t: "known", s: "Standard deviation of the i-th measurement. Must be known (and θ-independent) for least squares to equal maximum likelihood.", py: "sigma_i = np.full(N, 0.2)", f: "sigma_i = 0.2_dp" },
  xi: { n: "x<sub>i</sub> — the i-th datum", t: "data", s: "One measured value.", py: "x[i]", f: "x(i)" },
  mui: { n: "μ<sub>i</sub>(θ) — model prediction", t: "model", s: "What the model predicts for datum i given parameters θ (e.g. a light curve, an orbit).", py: "mu = model(t, theta)", f: "mu = model(t, theta)" },
  prod: { n: "∏ — product over data", t: "operator", s: "Independent data: joint probability = product. Always switch to a sum of logs.", py: "np.sum(np.log(p))", f: "sum(log(p))" },
  chi2: { n: "χ²(θ) — chi-squared", t: "misfit", s: "Sum of squared normalised residuals. ln 𝓛 = −χ²/2 + const for Gaussian data with known errors. Expected value ≈ number of degrees of freedom at the best fit.", py: "chi2 = np.sum(((x - mu) / sigma_i) ** 2)", f: "chi2 = sum(((x - mu)/sigma_i)**2)" },
  Pj: { n: "P<sub>j</sub> — periodogram bin", t: "data", s: "|DFT|² at frequency f<sub>j</sub>: an exponential random variable with mean S<sub>j</sub>. One record → 3199 such numbers.", py: "P = np.abs(np.fft.rfft(x))**2 * norm", f: "! from FFTW: P(j) = abs(X(j))**2 * norm" },
  Sj: { n: "S<sub>j</sub> — model at bin j", t: "model", s: "S(f<sub>j</sub>; θ): the expected value of P<sub>j</sub>.", py: "S = spectrum(f, *theta)", f: "S = s2*(1.0_dp + (fk/f)**alpha)" },
  F: { n: "F — Fisher information matrix", t: "d × d matrix", s: "Expected curvature of −ln 𝓛. Its inverse bounds the covariance of any unbiased estimator (Cramér–Rao). Computable before the experiment: forecasts.", py: "F = G @ G.T   # G[a, j] = d ln S_j / d theta_a", f: "F(a,b) = sum(dlnS(:,a)*dlnS(:,b))" },
  E: { n: "𝔼[·] — expectation over data sets", t: "operator", s: "Average over all records the experiment could have produced. Fisher averages the curvature; the Hessian H is the curvature of the one record we have.", py: "np.mean([stat(simulate(rng)) for _ in range(500)])", f: "! loop over simulated records and average" },
  s2ml: { n: "σ̂²<sub>ML</sub> — ML variance estimate", t: "estimator", s: "Divides by N: biased low by (N−1)/N. ddof=1 (N−1) is unbiased.", py: "x.var(ddof=0)", f: "sum((x-sum(x)/n)**2)/n" },
  thetatrue: { n: "θ<sub>true</sub> — the true value", t: "unknown (known in a simulation)", s: "Fixed but unknown in real life. In a simulation we know it — the only way to test coverage and bias.", py: "theta_true = np.array([-3.0, 0.0, 1.0])", f: "th_true = [-3.0_dp, 0.0_dp, 1.0_dp]" },
  ax: { n: "a(x) — lower end of the interval", t: "random", s: "Depends on the data, hence random: the frequentist probability is about the end points.", py: "a = th - err", f: "a = th - err" },
  bx: { n: "b(x) — upper end", t: "random", s: "Upper end point, random through the data.", py: "b = th + err", f: "b = th + err" },
  gamma: { n: "γ — miss probability", t: "level", s: "1 − γ is the confidence (or credibility) level: γ = 0.317 for “1σ” (68.3%), 0.05 for 95%.", py: "z = scipy.stats.norm.ppf(1 - gamma/2)", f: "! z = 1.0 for 68.3%, 1.96 for 95%" },
  spost: { n: "σ<sub>post</sub> — posterior width", t: "result", s: "Width of the posterior. For Gaussians: 1/σ²<sub>post</sub> = 1/σ²<sub>L</sub> + 1/σ²₀ (precisions add).", py: "s_post = 1/np.sqrt(1/s_L**2 + 1/s_0**2)", f: "s_post = 1.0_dp/sqrt(1.0_dp/s_L**2 + 1.0_dp/s_0**2)" },
  sL: { n: "σ<sub>L</sub> — width of the likelihood", t: "data precision", s: "How precisely the data alone constrain θ (e.g. 0.12 for α on one record).", py: "s_L = 0.12", f: "s_L = 0.12_dp" },
  s0: { n: "σ₀ — width of the prior", t: "prior precision", s: "How precisely θ was known before (e.g. 0.10 from a previous experiment).", py: "s_0 = 0.10", f: "s_0 = 0.10_dp" },
  mpost: { n: "μ<sub>post</sub> — posterior mean", t: "result", s: "Precision-weighted average of the likelihood peak and the prior centre.", py: "mu_post = s_post**2 * (mu_L/s_L**2 + mu_0/s_0**2)", f: "mu_post = s_post**2*(mu_L/s_L**2 + mu_0/s_0**2)" },
  mL: { n: "μ<sub>L</sub> — likelihood peak (MLE)", t: "data", s: "What the data alone say (0.82 for α).", py: "mu_L = 0.82", f: "mu_L = 0.82_dp" },
  m0: { n: "μ₀ — prior centre", t: "prior", s: "What we believed before (1.00 for α).", py: "mu_0 = 1.00", f: "mu_0 = 1.0_dp" },
  propto: { n: "∝ — proportional to", t: "relation", s: "Equal up to a θ-independent constant (here the evidence).", py: "post = like * prior; post /= post.sum()", f: "post = like*prior; post = post/sum(post)" },
  a: { n: "a — prior pseudo-successes + 1", t: "Beta parameter", s: "Beta(a, b) prior on a probability: worth a−1 successes seen before. a = b = 1: flat.", py: "a = 1.0", f: "a = 1.0_dp" },
  b: { n: "b — prior pseudo-failures + 1", t: "Beta parameter", s: "Worth b−1 failures seen before.", py: "b = 1.0", f: "b = 1.0_dp" },
  h: { n: "h — observed successes (heads)", t: "data", s: "Count of successes in the data.", py: "h = np.count_nonzero(tosses)", f: "h = count(tosses)" },
  t: { n: "t — observed failures (tails)", t: "data", s: "Count of failures in the data.", py: "t = tosses.size - h", f: "t = size(tosses) - h" },
  pphi: { n: "p<sub>φ</sub>(φ) — density in the new variable", t: "density", s: "The same probability, expressed per unit of φ instead of θ.", py: "p_phi = p_theta(g_inv(phi)) * abs(dginv_dphi(phi))", f: "p_phi = p_theta * abs(dtheta_dphi)" },
  ptheta: { n: "p<sub>θ</sub>(θ) — density in the old variable", t: "density", s: "Probability per unit of θ.", py: "p_theta(theta)", f: "p_theta(theta)" },
  jac: { n: "|dθ/dφ| — Jacobian", t: "factor", s: "Stretching factor between the variables. It's why a flat prior in f<sub>k</sub> is ∝ f<sub>k</sub> in log f<sub>k</sub>, and why MAP and mean move under re-parametrisation.", py: "jac = 10**v * np.log(10)    # d fk / d log10 fk", f: "jac = 10.0_dp**v * log(10.0_dp)" },
  thetamap: { n: "θ̂<sub>MAP</sub> — maximum a posteriori", t: "estimator", s: "Mode of the posterior density. Depends on the parametrisation.", py: "theta[np.argmax(lnL + lnprior)]", f: "theta(maxloc(lnL + lnprior, 1))" },
  R: { n: "R — credible region", t: "set", s: "A region holding 1−γ of the posterior probability. Not unique: equal-tailed or HPD.", py: "np.interp([0.16, 0.84], cdf, theta)", f: "! scan the cumulative sum" },
  H: { n: "H — Hessian of −ln p", t: "d × d matrix", s: "Curvature of the log-posterior at its mode, from this one record (observed information). C = H⁻¹ is the Laplace covariance.", py: "H = hessian(neg_log_post, theta_hat)", f: "H(i,j) = (f(pp) - f(pm) - f(mp) + f(mm))/(4*h*h)" },
  Ngauss: { n: "𝒩(m, C) — Gaussian distribution", t: "distribution", s: "Multivariate normal with mean m and covariance C.", py: "scipy.stats.multivariate_normal(m, C)", f: "! sample: m + matmul(L, z), L = chol(C), z ~ N(0,1)" },
  C: { n: "C = H⁻¹ — covariance matrix", t: "d × d matrix", s: "Diagonal: variances (errors²). Off-diagonal: covariances; correlation C<sub>ij</sub>/√(C<sub>ii</sub>C<sub>jj</sub>).", py: "C = np.linalg.inv(H)", f: "call dpotrf('U',d,C,d,info); call dpotri('U',d,C,d,info)" },
  pk: { n: "p<sub>k</sub> — posterior at grid point k", t: "number", s: "Normalised so that Σ<sub>k</sub> p<sub>k</sub> = 1.", py: "post.ravel()[k]", f: "post(i,j,k)" },
  lpk: { n: "ln p<sub>k</sub> — log-posterior at point k", t: "number", s: "ln 𝓛 + ln π: huge negative numbers (≈ −1.85×10⁴ here). Never exponentiate them directly.", py: "lp[k]", f: "lp(i,j,k)" },
  K: { n: "K — points per axis", t: "count", s: "Grid resolution. Total cost K<sup>d</sup>.", py: "K = 50", f: "integer, parameter :: K = 50" },
  teval: { n: "t<sub>eval</sub> — time of one likelihood", t: "seconds", s: "30 µs for Whittle on 3199 bins in NumPy on one core.", py: "%timeit neg_log_like(theta, f, P)", f: "call system_clock(t0); ...; call system_clock(t1)" },
  Z: { n: "Z — evidence of a model", t: "number", s: "Average of the likelihood over the prior: how much probability the model gave to the data before seeing them. Ratio of Z's = Bayes factor.", py: "lnZ = lnL.max() + np.log(np.mean(np.exp(lnL - lnL.max())))", f: "lnZ = maxval(lnL) + log(sum(exp(lnL-maxval(lnL)))/size(lnL))" },
  B10: { n: "B₁₀ — Bayes factor", t: "ratio", s: "Z₁/Z₀. Kass–Raftery scale on 2 ln B: 0–2 bare mention, 2–6 positive, 6–10 strong, >10 very strong.", py: "lnB = lnZ1 - lnZ0", f: "lnB = lnZ1 - lnZ0" },
  Lmax: { n: "𝓛<sub>max</sub> — best-fit likelihood", t: "number", s: "How well the model can fit. More parameters always increase it — the Occam factor pays the price.", py: "lnLmax = lnL.max()", f: "lnLmax = maxval(lnL)" },
  Delta: { n: "Δ — prior width", t: "prior", s: "Width of a flat prior. The wider it is, the more the evidence penalises the extra parameter.", py: "Delta = 3.0   # alpha in [0, 3]", f: "Delta = 3.0_dp" },
  ck: { n: "c — a class (material)", t: "label", s: "Vegetation, soil, water, roof … in the hyperspectral project.", py: "classes = np.unique(labels)", f: "! integer class labels 1..K" },
  ypix: { n: "𝐲 — the spectrum of a pixel", t: "data vector", s: "B reflectance values, one per band (or its PCA components).", py: "y = cube[row, col, :]", f: "y = cube(:, col, row)   ! band index fastest" },
  Pc: { n: "P(c) — class prior", t: "prior", s: "Flat (1/K) or the class frequency in the training set. Affects rare classes most.", py: "prior = np.bincount(train_labels) / train_labels.size", f: "prior = real(counts,dp)/sum(counts)" },
  Amat: { n: "A — endmember matrix", t: "B × K matrix", s: "Each column is the pure spectrum of one material. The model of a pixel is A θ.", py: "A = endmembers.T", f: "real(dp) :: A(B, K)" },
  abund: { n: "θ — abundances", t: "parameters", s: "Fraction of the pixel covered by each material (physically ≥ 0 and summing to 1).", py: "theta = Sigma @ (A.T @ (y / sig**2))", f: "theta = matmul(Sigma, matmul(transpose(A), y/sig**2))" },
  Cn: { n: "C — noise covariance", t: "B × B matrix", s: "Here diagonal: σ²<sub>b</sub> per band, larger in the water-vapour bands.", py: "Cinv = 1 / sig**2", f: "Cinv = 1.0_dp/sig**2" },
  Sig: { n: "Σ — posterior covariance of the abundances", t: "K × K matrix", s: "(AᵀC⁻¹A + P⁻¹)⁻¹. With a flat prior it is exactly the inverse Fisher matrix.", py: "Sigma = np.linalg.inv(A.T @ (A * Cinv[:, None]) + P0inv)", f: "Sigma = inverse(matmul(transpose(A), A*spread(Cinv,2,K)) + P0inv)" },
  N: { n: "N — number of data / samples", t: "count", s: "Here also N = 6400 time samples → 3199 periodogram bins. Errors shrink like 1/√N.", py: "N = P.size", f: "n = size(P)" }
};

(function () {
  const W = (window.WIDGETS = {});
  const fmt = M.fmt;

  /* ---------------------------------------------------------- 1. law of large numbers */
  W.lln = (root) => {
    UI.head(root, "LAW OF LARGE NUMBERS · FLIP A BIASED COIN");
    const k = UI.kpis(root, [["n", "trials N"], ["f", "frequency nE/N"], ["d", "distance to p"], ["e", "expected spread √(p(1−p)/N)"]]);
    const P = UI.canvas(root, 280);
    UI.legend(root, [["blue", "running frequency"], ["ink", "true p"], ["green", "±1σ band  p ± √(p(1−p)/N)"]]);
    const c = UI.ctrls(root);
    let p = 0.3, rng = M.rng(7), N = 0, nE = 0, xs = [], ys = [];
    UI.slider(c, { label: "true probability p", min: 0.05, max: 0.95, step: 0.01, value: p, fmt: (v) => v.toFixed(2), on: (v) => { p = v; reset(); } });
    const b = UI.btns(root);
    [1, 10, 100, 1000, 10000, 100000].forEach((n) => UI.button(b, "+" + n.toLocaleString(), () => flip(n), n === 100 ? "go" : ""));
    UI.button(b, "reset", reset, "ghost");
    function reset() { N = 0; nE = 0; xs = []; ys = []; rng = M.rng((Math.random() * 1e9) | 0); draw(); }
    function flip(n) {
      for (let i = 0; i < n; i++) { N++; if (rng() < p) nE++; if (N < 100 || N % Math.ceil(N / 150) === 0) { xs.push(N); ys.push(nE / N); } }
      draw();
    }
    function draw() {
      const nmax = Math.max(10, N);
      P.begin({ xlim: [1, nmax], ylim: [0, 1], xlog: true, xlabel: "number of trials N (log scale)", ylabel: "nE / N" });
      const g = M.logspace(0, Math.log10(nmax), 120);
      P.fill(g, g.map((n) => p - Math.sqrt(p * (1 - p) / n)), g.map((n) => p + Math.sqrt(p * (1 - p) / n)), { c: "green", a: 0.18 });
      P.hline(p, { c: "ink", label: "p = " + p.toFixed(2) });
      if (xs.length) { P.line(xs, ys, { c: "blue", w: 2 }); P.pts([xs[xs.length - 1]], [ys[ys.length - 1]], { c: "blue", r: 4 }); }
      k.set("n", N.toLocaleString()); k.set("f", N ? (nE / N).toFixed(4) : "—");
      k.set("d", N ? Math.abs(nE / N - p).toFixed(4) : "—"); k.set("e", N ? Math.sqrt(p * (1 - p) / N).toFixed(4) : "—");
    }
    root._redraw = draw; draw();
  };

  /* ---------------------------------------------------------- 2. urns */
  W.urns = (root) => {
    UI.head(root, "TWO URNS · LEARN WHICH ONE FROM THE DRAWS");
    const k = UI.kpis(root, [["d", "draws"], ["a1", "P(A₁ | data)"], ["a2", "P(A₂ | data)"], ["s", "secret urn"]]);
    const box = UI.el("div", "split"); root.append(box);
    const L = new Plot(Object.assign(document.createElement("canvas")), { h: 250, pad: [10, 10, 10, 10] });
    const w1 = UI.el("div", "canvas-wrap"); w1.append(L.c); box.append(w1);
    const w2 = UI.el("div", "canvas-wrap"); const c2 = document.createElement("canvas"); w2.append(c2); box.append(w2);
    const R = new Plot(c2, { h: 250, pad: [16, 14, 38, 46] });
    const c = UI.ctrls(root);
    let prior = 0.5, hist = [], post = [prior], secret = null, rng = M.rng(3);
    UI.slider(c, { label: "prior P(A₁)", min: 0.05, max: 0.95, step: 0.05, value: 0.5, fmt: (v) => v.toFixed(2), on: (v) => { prior = v; recompute(); } });
    const b = UI.btns(root);
    UI.button(b, "observe ● blue", () => add("B"), "alt");
    UI.button(b, "observe ● red", () => add("R"));
    UI.button(b, "draw from a secret urn", () => { if (secret === null) secret = rng() < prior ? 1 : 2; const pb = secret === 1 ? 0.3 : 0.6; add(rng() < pb ? "B" : "R"); }, "go");
    UI.button(b, "reveal", () => { k.set("s", secret ? "A" + (secret === 1 ? "₁" : "₂") : "none yet"); });
    UI.button(b, "reset", () => { hist = []; secret = null; k.set("s", "hidden"); recompute(); }, "ghost");
    function add(d) { hist.push(d); recompute(); }
    function recompute() {
      post = [prior]; let p1 = prior;
      for (const d of hist) { const l1 = d === "B" ? 0.3 : 0.7, l2 = d === "B" ? 0.6 : 0.4; p1 = (l1 * p1) / (l1 * p1 + l2 * (1 - p1)); post.push(p1); }
      draw();
    }
    function drawUrn(g, x, y, w, h, nb, nr, label, prob) {
      g.strokeStyle = col("ink"); g.lineWidth = 2; g.fillStyle = col("paper");
      g.beginPath(); g.moveTo(x, y); g.lineTo(x, y + h - 14); g.quadraticCurveTo(x, y + h, x + 14, y + h); g.lineTo(x + w - 14, y + h); g.quadraticCurveTo(x + w, y + h, x + w, y + h - 14); g.lineTo(x + w, y); g.stroke();
      const balls = Array(nb).fill("blue").concat(Array(nr).fill("red"));
      balls.forEach((bc, i) => { const cx = x + 18 + (i % 4) * ((w - 36) / 3), cy = y + h - 18 - Math.floor(i / 4) * 24; g.fillStyle = col(bc); g.beginPath(); g.arc(cx, cy, 9, 0, 7); g.fill(); g.strokeStyle = col("ink"); g.lineWidth = 1; g.stroke(); });
      g.fillStyle = col("ink"); g.font = "700 13px JBM"; g.textAlign = "center"; g.fillText(label, x + w / 2, y - 22);
      g.font = "500 11.5px JBM"; g.fillText(`P(blue) = ${(nb / 10).toFixed(1)}`, x + w / 2, y - 7);
      // posterior bar under
      g.fillStyle = col("line"); g.fillRect(x, y + h + 12, w, 12); g.fillStyle = col("purple"); g.fillRect(x, y + h + 12, w * prob, 12);
      g.fillStyle = col("ink"); g.fillText((100 * prob).toFixed(1) + "%", x + w / 2, y + h + 40);
    }
    function draw() {
      L.size(); const g = L.ctx; g.clearRect(0, 0, L.W, L.H);
      const w = Math.min(130, L.W / 2 - 40), p1 = post[post.length - 1];
      drawUrn(g, L.W / 2 - w - 20, 50, w, 120, 3, 7, "A₁", p1); drawUrn(g, L.W / 2 + 20, 50, w, 120, 6, 4, "A₂", 1 - p1);
      // history of draws as dots
      g.font = "500 11px JBM"; g.textAlign = "left"; g.fillStyle = col("muted"); g.fillText("draws:", 8, 18);
      hist.slice(-28).forEach((d, i) => { g.fillStyle = col(d === "B" ? "blue" : "red"); g.beginPath(); g.arc(62 + i * 13, 14, 5, 0, 7); g.fill(); });
      const n = hist.length;
      R.begin({ xlim: [0, Math.max(6, n)], ylim: [0, 1], xlabel: "number of draws", ylabel: "P(A₁ | data)" });
      R.hline(0.5, { c: "faint" });
      R.line(post.map((_, i) => i), post, { c: "purple", w: 2.5 }); R.pts(post.map((_, i) => i), post, { c: "purple", r: 3.5 });
      k.set("d", n); k.set("a1", p1.toFixed(4)); k.set("a2", (1 - p1).toFixed(4));
    }
    k.set("s", "hidden"); root._redraw = draw; recompute();
    UI.note(root, "Check the lecture: from a ½ prior, blue → 0.3333, blue again → 0.2000. Change the prior and see how many draws it takes for the data to win.");
  };

  /* ---------------------------------------------------------- 2b. alarm */
  W.alarm = (root) => {
    UI.head(root, "THE BASE-RATE TRAP · 1000 CANDIDATES");
    const k = UI.kpis(root, [["p", "P(real | alarm)"], ["tp", "true alarms"], ["fp", "false alarms"], ["miss", "missed events"]]);
    const P = UI.canvas(root, 230, { pad: [6, 6, 6, 6] });
    UI.legend(root, [["green", "real & detected"], ["red", "real & missed"], ["orange", "false alarm"], ["line", "quiet noise"]]);
    const c = UI.ctrls(root);
    let prev = 0.01, sens = 0.99, fa = 0.05;
    UI.slider(c, { label: "prevalence P(real)", min: -3, max: -0.3, step: 0.01, value: -2, fmt: (v) => (100 * Math.pow(10, v)).toFixed(2) + "%", on: (v) => { prev = Math.pow(10, v); draw(); } });
    UI.slider(c, { label: "sensitivity P(alarm | real)", min: 0.5, max: 1, step: 0.005, value: 0.99, fmt: (v) => (100 * v).toFixed(1) + "%", on: (v) => { sens = v; draw(); } });
    UI.slider(c, { label: "false-alarm rate P(alarm | noise)", min: 0.001, max: 0.3, step: 0.001, value: 0.05, fmt: (v) => (100 * v).toFixed(1) + "%", on: (v) => { fa = v; draw(); } });
    function draw() {
      P.size(); const g = P.ctx; g.clearRect(0, 0, P.W, P.H);
      const n = 1000, real = Math.round(n * prev), tp = Math.round(real * sens), miss = real - tp, fp = Math.round((n - real) * fa);
      const cols = 50, s = Math.min((P.W - 12) / cols, (P.H - 12) / 20), r = s * 0.36;
      const kinds = []; for (let i = 0; i < n; i++) kinds.push(i < tp ? "green" : i < real ? "red" : i < real + fp ? "orange" : "line");
      kinds.forEach((kd, i) => { g.fillStyle = col(kd); g.beginPath(); g.arc(6 + (i % cols) * s + s / 2, 6 + Math.floor(i / cols) * s + s / 2, r, 0, 7); g.fill(); });
      const post = (sens * prev) / (sens * prev + fa * (1 - prev));
      k.set("p", (100 * post).toFixed(1) + "%"); k.set("tp", tp); k.set("fp", fp); k.set("miss", miss);
    }
    root._redraw = draw; draw();
  };

  /* ---------------------------------------------------------- 3. likelihood explorer */
  W.like = (root) => {
    UI.head(root, "LIKELIHOOD EXPLORER · MEAN OF GAUSSIAN DATA (σ = 1 KNOWN)");
    const k = UI.kpis(root, [["mu", "trial μ"], ["nll", "−ln 𝓛(μ)"], ["chi", "χ²(μ)"], ["mle", "MLE μ̂ = x̄"], ["err", "Fisher error 1/√N"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 250), B = UI.canvas(box, 250);
    const c = UI.ctrls(root);
    let n = 8, mu = 4, x = [], rng = M.rng(11);
    const gen = () => { x = []; for (let i = 0; i < n; i++) x.push(5 + rng.normal()); };
    UI.slider(c, { label: "trial value μ (drag me)", min: 2, max: 8, step: 0.01, value: mu, fmt: (v) => v.toFixed(2), on: (v) => { mu = v; draw(); } });
    UI.slider(c, { label: "number of data N", min: 2, max: 60, step: 1, value: n, on: (v) => { n = v; gen(); draw(); } });
    const b = UI.btns(root); UI.button(b, "new data set", () => { gen(); draw(); }, "go");
    UI.button(b, "jump to the MLE", () => { mu = M.mean(x); c.querySelector("input").set(mu); draw(); });
    const nll = (m) => x.reduce((s, v) => s + 0.5 * (v - m) ** 2, 0) + 0.5 * x.length * Math.log(2 * Math.PI);
    function draw() {
      const t = M.linspace(1.5, 8.5, 300), xb = M.mean(x);
      A.begin({ xlim: [1.5, 8.5], ylim: [0, 0.5], xlabel: "x", ylabel: "p(x | μ)", title: "model N(μ, 1) and the data" });
      A.fill(t, 0, t.map((v) => M.npdf(v, mu, 1)), { c: "purple", a: 0.18 }); A.line(t, t.map((v) => M.npdf(v, mu, 1)), { c: "purple" });
      x.forEach((v) => { A.seg(v, 0, v, M.npdf(v, mu, 1), { c: "orange", w: 1.5, a: 0.8 }); });
      A.pts(x, x.map(() => 0.012), { c: "ink", r: 3.5 });
      const ms = M.linspace(2, 8, 200), ys = ms.map(nll), ymin = nll(xb);
      B.begin({ xlim: [2, 8], ylim: [ymin - 1, ymin + Math.max(8, 0.5 * n * 4)], xlabel: "μ", ylabel: "−ln 𝓛(μ)", title: "the negative log-likelihood" });
      B.line(ms, ys, { c: "blue" }); B.vline(xb, { c: "green", label: "MLE" });
      B.hline(ymin + 0.5, { c: "faint", label: "min + ½  → ±1σ" });
      B.pts([mu], [nll(mu)], { c: "orange", r: 6, stroke: "ink" });
      const chi = x.reduce((s, v) => s + (v - mu) ** 2, 0);
      k.set("mu", mu.toFixed(2)); k.set("nll", nll(mu).toFixed(2)); k.set("chi", chi.toFixed(2)); k.set("mle", xb.toFixed(3)); k.set("err", (1 / Math.sqrt(x.length)).toFixed(3));
    }
    gen(); root._redraw = draw; draw();
    UI.note(root, "The orange segments are the probabilities p(xᵢ | μ) of each datum; 𝓛 is their product. Where −ln 𝓛 rises by ½ above its minimum you are 1σ away: that is the Fisher error 1/√N.");
  };

  /* ---------------------------------------------------------- 3b. the knee spectrum */
  function kneeData(T, seed) {
    const n = Math.round(T * 100) / 2 - 1, f = new Float64Array(n), P = new Float64Array(n), rng = M.rng(seed);
    for (let j = 0; j < n; j++) { f[j] = (j + 1) / T; const S = 1e-3 * (1 + 1 / f[j]); P[j] = -S * Math.log(1 - rng()); }
    return { f, P, n, lnf: Float64Array.from(f, Math.log) };
  }
  function kneeFit(d) { // grid over (log10 fk, alpha), sigma2 profiled: s2_hat = mean(P/g)
    let best = { nll: Infinity };
    const pass = (v0, v1, a0, a1, nv, na) => {
      for (let iv = 0; iv < nv; iv++) for (let ia = 0; ia < na; ia++) {
        const v = v0 + (v1 - v0) * iv / (nv - 1), a = a0 + (a1 - a0) * ia / (na - 1);
        let A = 0, B = 0; const L10 = v * Math.LN10;
        for (let j = 0; j < d.n; j++) { const g = 1 + Math.exp(a * (L10 - d.lnf[j])); A += Math.log(g); B += d.P[j] / g; }
        const s2 = B / d.n, nll = d.n * Math.log(s2) + A + d.n;
        if (nll < best.nll) best = { nll, v, a, s2 };
      }
    };
    pass(-1.5, 1.2, 0.2, 2.5, 28, 24);
    pass(best.v - 0.12, best.v + 0.12, best.a - 0.12, best.a + 0.12, 13, 13);
    return best;
  }
  W.knee = (root) => {
    UI.head(root, "FIT THE KNEE BY HAND · WHITTLE LIKELIHOOD ON A SIMULATED PERIODOGRAM");
    const k = UI.kpis(root, [["nll", "Δ(−ln 𝓛) vs the MLE"], ["fit", "MLE  (σ², fₖ, α)"], ["n", "bins"]]);
    const P = UI.canvas(root, 330);
    UI.legend(root, [["faint", "periodogram Pⱼ"], ["blue", "log-bin averages"], ["ink", "truth (10⁻³, 1 Hz, 1)"], ["orange", "your model"], ["green", "MLE"]]);
    const c = UI.ctrls(root);
    let T = 64, seed = 12, d, fit, showBest = false, th = { s2: -3, v: 0.3, a: 1.4 };
    const load = () => { d = kneeData(T, seed); fit = kneeFit(d); showBest = false; k.set("fit", `${fit.s2.toExponential(2)}, ${Math.pow(10, fit.v).toFixed(2)} Hz, ${fit.a.toFixed(2)}`); k.set("n", d.n.toLocaleString()); };
    const s1 = UI.slider(c, { label: "log₁₀ σ²", min: -3.5, max: -2.5, step: 0.005, value: th.s2, fmt: (v) => v.toFixed(3), on: (v) => { th.s2 = v; draw(); } });
    const s2 = UI.slider(c, { label: "log₁₀ fₖ  [Hz]", min: -1.5, max: 1.2, step: 0.01, value: th.v, fmt: (v) => v.toFixed(2) + "  (fₖ = " + Math.pow(10, v).toFixed(2) + " Hz)", on: (v) => { th.v = v; draw(); } });
    const s3 = UI.slider(c, { label: "α", min: 0.2, max: 2.5, step: 0.01, value: th.a, fmt: (v) => v.toFixed(2), on: (v) => { th.a = v; draw(); } });
    UI.select(c, { label: "record length T", options: [["8", "8 s (399 bins)"], ["64", "64 s (3199 bins)"], ["512", "512 s (25599 bins)"]], value: "64", on: (v) => { T = +v; load(); draw(); } });
    const b = UI.btns(root);
    UI.button(b, "show the MLE", () => { showBest = true; th = { s2: Math.log10(fit.s2), v: fit.v, a: fit.a }; s1.set(th.s2); s2.set(th.v); s3.set(th.a); draw(); }, "go");
    UI.button(b, "new record (new noise)", () => { seed++; load(); draw(); });
    const nllOf = (s2, v, a) => { let s = 0; const fk = Math.pow(10, v), S2 = Math.pow(10, s2); for (let j = 0; j < d.n; j++) { const S = S2 * (1 + Math.pow(fk / d.f[j], a)); s += Math.log(S) + d.P[j] / S; } return s; };
    function draw() {
      const fmin = d.f[0], fmax = d.f[d.n - 1];
      P.begin({ xlim: [fmin, fmax], ylim: [1e-6, 3], xlog: true, ylog: true, xlabel: "f [Hz]", ylabel: "S(f)" });
      const step = Math.max(1, Math.floor(d.n / 3000)), xs = [], ys = [];
      for (let j = 0; j < d.n; j += step) { xs.push(d.f[j]); ys.push(Math.max(d.P[j], 1.1e-6)); }
      P.pts(xs, ys, { c: "faint", r: 1.2, a: 0.5 });
      const edges = M.logspace(Math.log10(fmin), Math.log10(fmax), 26), bx = [], by = [];
      for (let i = 0; i < 25; i++) { let s = 0, m = 0; for (let j = 0; j < d.n; j++) if (d.f[j] >= edges[i] && d.f[j] < edges[i + 1]) { s += d.P[j]; m++; } if (m) { bx.push(Math.sqrt(edges[i] * edges[i + 1])); by.push(s / m); } }
      P.pts(bx, by, { c: "blue", r: 4 });
      const g = M.logspace(Math.log10(fmin), Math.log10(fmax), 200);
      P.line(g, g.map((f) => 1e-3 * (1 + 1 / f)), { c: "ink", w: 2 });
      if (showBest) P.line(g, g.map((f) => fit.s2 * (1 + Math.pow(Math.pow(10, fit.v) / f, fit.a))), { c: "green", w: 2.5, dash: [7, 4] });
      P.line(g, g.map((f) => Math.pow(10, th.s2) * (1 + Math.pow(Math.pow(10, th.v) / f, th.a))), { c: "orange", w: 2.5 });
      P.vline(Math.pow(10, th.v), { c: "orange", label: "fₖ" });
      k.set("nll", (nllOf(th.s2, th.v, th.a) - fit.nll).toFixed(2));
    }
    load(); root._redraw = draw; draw();
    UI.note(root, "Goal: bring Δ(−ln 𝓛) close to 0 by hand. A rise of ½ is “1σ” for one parameter. Try T = 8 s: the knee is barely visible and many (fₖ, α) pairs fit almost equally well — the prior will decide (lesson 5).");
  };

  /* ---------------------------------------------------------- 4. coverage */
  W.coverage = (root) => {
    UI.head(root, "COVERAGE · REPEAT THE EXPERIMENT MANY TIMES");
    const k = UI.kpis(root, [["n", "experiments"], ["c", "measured coverage"], ["nom", "nominal"], ["last", "last interval"]]);
    const P = UI.canvas(root, 300);
    UI.legend(root, [["green", "interval contains μ_true"], ["red", "interval misses it"], ["ink", "μ_true = 0"]]);
    const c = UI.ctrls(root);
    let N = 5, lev = 0.6827, mode = "known", rng = M.rng(5), runs = [], cov = 0;
    UI.slider(c, { label: "data per experiment N", min: 2, max: 100, step: 1, value: N, on: (v) => { N = v; reset(); } });
    UI.select(c, { label: "confidence level", options: [["0.6827", "68.3% (±1σ)"], ["0.95", "95% (±1.96σ)"]], on: (v) => { lev = +v; reset(); } });
    UI.select(c, { label: "σ in the interval", options: [["known", "σ known (= 1)"], ["est", "σ estimated from the N data"]], on: (v) => { mode = v; reset(); } });
    const b = UI.btns(root); [1, 50, 1000].forEach((n) => UI.button(b, "+" + n, () => run(n), n === 50 ? "go" : "")); UI.button(b, "reset", reset, "ghost");
    function reset() { runs = []; cov = 0; draw(); }
    function run(n) {
      const z = -M.nppf((1 - lev) / 2);
      for (let r = 0; r < n; r++) {
        const x = []; for (let i = 0; i < N; i++) x.push(rng.normal());
        const m = M.mean(x), s = mode === "known" ? 1 : M.std(x), h = z * s / Math.sqrt(N), ok = Math.abs(m) <= h;
        runs.push([m - h, m + h, ok]); if (ok) cov++;
      }
      draw();
    }
    function draw() {
      const show = runs.slice(-40), w = 3.2 / Math.sqrt(Math.max(2, N)) * 2.2;
      P.begin({ xlim: [-w, w], ylim: [0, 41], xlabel: "interval for μ  (one row = one experiment, newest on top)", yticks: 1 });
      P.vline(0, { c: "ink", dash: [] });
      show.forEach((r, i) => P.seg(r[0], i + 1, r[1], i + 1, { c: r[2] ? "green" : "red", w: 3 }));
      const n = runs.length; k.set("n", n.toLocaleString()); k.set("c", n ? (100 * cov / n).toFixed(1) + "%" : "—"); k.set("nom", (100 * lev).toFixed(1) + "%");
      k.set("last", n ? `[${runs[n - 1][0].toFixed(2)}, ${runs[n - 1][1].toFixed(2)}]` : "—");
    }
    root._redraw = draw; run(40);
  };

  /* ---------------------------------------------------------- 4b. bias of the ML variance */
  W.bias = (root) => {
    UI.head(root, "THE MLE OF A VARIANCE IS BIASED · σ² = 4");
    const k = UI.kpis(root, [["ml", "mean of σ̂²_ML (÷N)"], ["th", "theory (N−1)/N·σ²"], ["un", "mean of ÷(N−1)"]]);
    const P = UI.canvas(root, 250); const c = UI.ctrls(root);
    let N = 5; const rng = M.rng(19);
    UI.slider(c, { label: "sample size N", min: 2, max: 30, step: 1, value: N, on: (v) => { N = v; draw(); } });
    function draw() {
      const ml = [], un = []; let sm = 0, su = 0;
      for (let r = 0; r < 20000; r++) { let s = 0, q = 0; const x = []; for (let i = 0; i < N; i++) { const v = 2 * rng.normal(); x.push(v); s += v; } const m = s / N; for (const v of x) q += (v - m) ** 2; ml.push(q / N); sm += q / N; su += q / (N - 1); }
      const h = M.hist(ml, 0, 14, 70);
      P.begin({ xlim: [0, 14], ylim: [0, Math.max(...h.dens) * 1.15], xlabel: "σ̂²_ML from one data set", ylabel: "density" });
      P.bars(h.edges, h.dens, { c: "blue", a: 0.4 });
      P.vline(4, { c: "ink", label: "true σ² = 4", dash: [] }); P.vline(sm / 20000, { c: "red", label: "mean of σ̂²_ML", dy: 16 });
      k.set("ml", (sm / 20000).toFixed(3)); k.set("th", ((N - 1) / N * 4).toFixed(3)); k.set("un", (su / 20000).toFixed(3));
    }
    root._redraw = draw; draw();
  };

  /* ---------------------------------------------------------- 5. conjugate Gaussian */
  W.conj = (root) => {
    UI.head(root, "PRIOR × LIKELIHOOD = POSTERIOR · α OF THE KNEE");
    const k = UI.kpis(root, [["post", "posterior μ ± σ"], ["wL", "weight of the data"], ["w0", "weight of the prior"]]);
    const P = UI.canvas(root, 290);
    UI.legend(root, [["blue", "likelihood (this record)"], ["orange", "prior (previous experiment)"], ["purple", "posterior"]]);
    const c = UI.ctrls(root);
    const s = { mL: 0.82, sL: 0.12, m0: 1.0, s0: 0.1 };
    UI.slider(c, { label: "likelihood peak μ_L", min: 0.4, max: 1.4, step: 0.01, value: s.mL, fmt: (v) => v.toFixed(2), on: (v) => { s.mL = v; draw(); } });
    UI.slider(c, { label: "likelihood width σ_L", min: 0.02, max: 0.4, step: 0.005, value: s.sL, fmt: (v) => v.toFixed(3), on: (v) => { s.sL = v; draw(); } });
    UI.slider(c, { label: "prior centre μ₀", min: 0.4, max: 1.4, step: 0.01, value: s.m0, fmt: (v) => v.toFixed(2), on: (v) => { s.m0 = v; draw(); } });
    UI.slider(c, { label: "prior width σ₀", min: 0.02, max: 1, step: 0.005, value: s.s0, fmt: (v) => v.toFixed(3), on: (v) => { s.s0 = v; draw(); } });
    function draw() {
      const wL = 1 / s.sL ** 2, w0 = 1 / s.s0 ** 2, sp = 1 / Math.sqrt(wL + w0), mp = sp * sp * (wL * s.mL + w0 * s.m0);
      const t = M.linspace(0.2, 1.6, 400), mx = M.npdf(mp, mp, sp) * 1.1;
      P.begin({ xlim: [0.2, 1.6], ylim: [0, mx], xlabel: "α", ylabel: "density" });
      P.fill(t, 0, t.map((v) => M.npdf(v, mp, sp)), { c: "purple", a: 0.2 });
      P.line(t, t.map((v) => M.npdf(v, s.mL, s.sL)), { c: "blue" }); P.line(t, t.map((v) => M.npdf(v, s.m0, s.s0)), { c: "orange", dash: [6, 4] });
      P.line(t, t.map((v) => M.npdf(v, mp, sp)), { c: "purple", w: 3 });
      k.set("post", `${mp.toFixed(3)} ± ${sp.toFixed(3)}`); k.set("wL", (100 * wL / (wL + w0)).toFixed(0) + "%"); k.set("w0", (100 * w0 / (wL + w0)).toFixed(0) + "%");
    }
    root._redraw = draw; draw();
  };

  /* ---------------------------------------------------------- 5b. beta-binomial */
  W.beta = (root) => {
    UI.head(root, "A COIN AND ITS PRIOR · BETA–BINOMIAL");
    const k = UI.kpis(root, [["ht", "heads / tails"], ["mle", "MLE h/(h+t)"], ["mean", "posterior mean"], ["ci", "68% credible interval"]]);
    const P = UI.canvas(root, 280);
    UI.legend(root, [["orange", "prior"], ["blue", "likelihood (scaled)"], ["purple", "posterior"]]);
    const c = UI.ctrls(root);
    const PR = { flat: [1, 1], jeff: [0.5, 0.5], fair: [20, 20], sk: [2, 8] };
    let pr = PR.flat, ptrue = 0.7, h = 0, t = 0, rng = M.rng(21);
    UI.select(c, { label: "prior", options: [["flat", "flat Beta(1,1)"], ["jeff", "Jeffreys Beta(½,½)"], ["fair", "“coins are fair” Beta(20,20)"], ["sk", "skeptic Beta(2,8)"]], on: (v) => { pr = PR[v]; draw(); } });
    UI.slider(c, { label: "true p of the coin", min: 0.05, max: 0.95, step: 0.01, value: ptrue, fmt: (v) => v.toFixed(2), on: (v) => { ptrue = v; } });
    const b = UI.btns(root); [1, 10, 100, 1000].forEach((n) => UI.button(b, "toss ×" + n, () => { for (let i = 0; i < n; i++) rng() < ptrue ? h++ : t++; draw(); }, n === 10 ? "go" : ""));
    UI.button(b, "reset", () => { h = 0; t = 0; draw(); }, "ghost");
    function draw() {
      const x = M.linspace(0.001, 0.999, 500), a = pr[0] + h, bb = pr[1] + t;
      const post = x.map((v) => M.betaPdf(v, a, bb)), prior = x.map((v) => M.betaPdf(v, pr[0], pr[1]));
      const ll = x.map((v) => Math.exp(h * Math.log(v) + t * Math.log(1 - v))), lmax = Math.max(...ll), pmax = Math.max(...post);
      P.begin({ xlim: [0, 1], ylim: [0, Math.min(60, Math.max(pmax, 3) * 1.1)], xlabel: "p (probability of heads)", ylabel: "density" });
      P.line(x, prior, { c: "orange", dash: [6, 4] });
      if (h + t) P.line(x, ll.map((v) => (v / lmax) * pmax * 0.9), { c: "blue", w: 1.5 });
      P.fill(x, 0, post, { c: "purple", a: 0.18 }); P.line(x, post, { c: "purple", w: 3 });
      P.vline(ptrue, { c: "ink", label: "true p" });
      const lo = M.gridQuantile(x, post, 0.16), hi = M.gridQuantile(x, post, 0.84);
      k.set("ht", `${h} / ${t}`); k.set("mle", h + t ? (h / (h + t)).toFixed(3) : "—"); k.set("mean", (a / (a + bb)).toFixed(3)); k.set("ci", `[${lo.toFixed(3)}, ${hi.toFixed(3)}]`);
    }
    root._redraw = draw; draw();
    UI.note(root, "Toss 10 times with the “fair” prior: the posterior barely moves. Toss 1000 times: every prior gives the same answer — the data decide.");
  };

  /* ---------------------------------------------------------- 5c. jacobian */
  W.jacobian = (root) => {
    UI.head(root, "FLAT IN WHAT? · THE SAME PRIOR SEEN IN TWO VARIABLES");
    const k = UI.kpis(root, [["hi", "prior mass with fₖ > 1 Hz"], ["med", "prior median of fₖ"]]);
    const P = UI.canvas(root, 260);
    const c = UI.ctrls(root);
    let which = "lin", axis = "log"; const rng = M.rng(4);
    UI.select(c, { label: "prior", options: [["lin", "flat in fₖ on [0.01, 50] Hz"], ["log", "flat in log₁₀ fₖ on [0.01, 50] Hz"]], on: (v) => { which = v; draw(); } });
    UI.select(c, { label: "show the histogram in", options: [["log", "log₁₀ fₖ"], ["lin", "fₖ"]], on: (v) => { axis = v; draw(); } });
    function draw() {
      const s = []; for (let i = 0; i < 40000; i++) s.push(which === "lin" ? 0.01 + 49.99 * rng() : Math.pow(10, -2 + Math.log10(5000) * rng()));
      const v = axis === "log" ? s.map(Math.log10) : s, lo = axis === "log" ? -2 : 0, hi = axis === "log" ? Math.log10(50) : 50;
      const h = M.hist(v, lo, hi, 60);
      P.begin({ xlim: [lo, hi], ylim: [0, Math.max(...h.dens) * 1.15], xlabel: axis === "log" ? "log₁₀ fₖ" : "fₖ [Hz]", ylabel: "prior density" });
      P.bars(h.edges, h.dens, { c: which === "lin" ? "orange" : "green", a: 0.55 });
      P.vline(axis === "log" ? 0 : 1, { c: "ink", label: "1 Hz (truth)" });
      const sorted = s.slice().sort((a, b) => a - b);
      k.set("hi", (100 * s.filter((x) => x > 1).length / s.length).toFixed(1) + "%"); k.set("med", M.quantile(sorted, 0.5).toFixed(2) + " Hz");
    }
    root._redraw = draw; draw();
    UI.note(root, "“Flat in fₖ” puts 98% of its weight above 1 Hz: with only 8 s of data this belief drags the median to 4 Hz. Flat in log₁₀ fₖ gives each decade equal weight.");
  };

  /* ---------------------------------------------------------- 6. summaries of a skewed posterior */
  W.summ = (root) => {
    UI.head(root, "ONE POSTERIOR, SEVERAL ANSWERS · MARGINAL OF fₖ");
    const k = UI.kpis(root, [["map", "MAP"], ["med", "median"], ["mean", "mean"], ["et", "68% equal-tailed"], ["hpd", "68% HPD"]]);
    const P = UI.canvas(root, 290);
    UI.legend(root, [["green", "equal-tailed 68%"], ["red", "HPD 68%"], ["blue", "MAP"], ["ink", "median"], ["orange", "mean"]]);
    const c = UI.ctrls(root);
    let m = 0.92, s = 0.11, axis = "lin";
    UI.slider(c, { label: "median of fₖ [Hz]", min: 0.4, max: 2, step: 0.01, value: m, fmt: (v) => v.toFixed(2), on: (v) => { m = v; draw(); } });
    UI.slider(c, { label: "width in log₁₀ fₖ (dex)", min: 0.03, max: 0.35, step: 0.005, value: s, fmt: (v) => v.toFixed(3), on: (v) => { s = v; draw(); } });
    UI.select(c, { label: "variable", options: [["lin", "fₖ (skewed)"], ["log", "log₁₀ fₖ (symmetric)"]], on: (v) => { axis = v; draw(); } });
    function draw() {
      const L = Math.log10(m), lo = axis === "log" ? L - 5 * s : Math.max(1e-3, Math.pow(10, L - 4 * s)), hi = axis === "log" ? L + 5 * s : Math.pow(10, L + 5 * s);
      const x = M.linspace(lo, hi, 1500);
      const p = x.map((v) => (axis === "log" ? M.npdf(v, L, s) : M.npdf(Math.log10(v), L, s) / (v * Math.LN10)));
      const dx = x[1] - x[0], Z = M.sum(p) * dx, pn = p.map((v) => v / Z);
      const imap = pn.indexOf(Math.max(...pn)), mean = M.sum(x.map((v, i) => v * pn[i])) * dx;
      const q = (t) => M.gridQuantile(x, pn, t), med = q(0.5), e1 = q(0.1587), e2 = q(0.8413);
      const ord = pn.map((v, i) => i).sort((a, b) => pn[b] - pn[a]); let acc = 0, h1 = Infinity, h2 = -Infinity;
      for (const i of ord) { acc += pn[i] * dx; h1 = Math.min(h1, x[i]); h2 = Math.max(h2, x[i]); if (acc >= 0.6827) break; }
      const ymax = Math.max(...pn) * 1.25;
      P.begin({ xlim: [lo, hi], ylim: [0, ymax], xlabel: axis === "log" ? "log₁₀ fₖ" : "fₖ [Hz]", ylabel: "posterior density" });
      const sel = (a, b) => x.map((v, i) => (v >= a && v <= b ? pn[i] : 0));
      P.fill(x, 0, sel(e1, e2), { c: "green", a: 0.25 });
      P.seg(h1, ymax * 0.93, h2, ymax * 0.93, { c: "red", w: 4 }); P.text(h1, ymax * 0.985, "HPD", { c: "red", bold: true });
      P.line(x, pn, { c: "ink", w: 2.5 });
      P.vline(x[imap], { c: "blue", dash: [] }); P.vline(med, { c: "ink", dash: [3, 3] }); P.vline(mean, { c: "orange", dash: [] });
      const show = (v) => (axis === "log" ? `${v.toFixed(3)} (${Math.pow(10, v).toFixed(2)} Hz)` : v.toFixed(3) + " Hz");
      k.set("map", show(x[imap])); k.set("med", show(med)); k.set("mean", show(mean));
      k.set("et", `[${e1.toFixed(2)}, ${e2.toFixed(2)}]`); k.set("hpd", `[${h1.toFixed(2)}, ${h2.toFixed(2)}]`);
    }
    root._redraw = draw; draw();
    UI.note(root, "Switch the variable: the median and the equal-tailed interval map exactly (10^x), the MAP, mean and HPD do not.");
  };

  /* ---------------------------------------------------------- 6b. marginalisation */
  W.marg = (root) => {
    UI.head(root, "MARGINAL vs SLICE · A CORRELATED 2-D POSTERIOR");
    const k = UI.kpis(root, [["m", "marginal σ₁"], ["s", "slice σ₁ at θ₂ fixed"], ["r", "slice / marginal"]]);
    const P = UI.canvas(root, 330, { h: (w) => Math.min(380, w * 0.62) });
    UI.legend(root, [["purple", "posterior samples"], ["green", "marginal of θ₁ (projection)"], ["red", "slice at θ₂ = 0 (nuisance fixed)"]]);
    const c = UI.ctrls(root); let rho = -0.56; const rng = M.rng(30);
    UI.slider(c, { label: "correlation ρ", min: -0.95, max: 0.95, step: 0.01, value: rho, fmt: (v) => v.toFixed(2), on: (v) => { rho = v; draw(); } });
    function draw() {
      P.begin({ xlim: [-4, 4], ylim: [-4, 4], xlabel: "θ₁ (e.g. log fₖ, standardised)", ylabel: "θ₂ (e.g. α)" });
      const xs = [], ys = []; for (let i = 0; i < 1500; i++) { const a = rng.normal(), b = rng.normal(); xs.push(a); ys.push(rho * a + Math.sqrt(1 - rho * rho) * b); }
      P.pts(xs, ys, { c: "purple", r: 1.8, a: 0.45 });
      for (const kk of [1, 2]) { const t = M.linspace(0, 2 * Math.PI, 120), r2 = kk === 1 ? 2.3 : 6.18; const L = [[1, 0], [rho, Math.sqrt(1 - rho * rho)]]; const ex = t.map((a) => Math.sqrt(r2) * Math.cos(a)), ey = t.map((a) => Math.sqrt(r2) * Math.sin(a)); P.line(ex.map((u, i) => L[0][0] * u), ex.map((u, i) => L[1][0] * u + L[1][1] * ey[i]), { c: "ink", w: 1.5 }); }
      const g = M.linspace(-4, 4, 200);
      P.line(g, g.map((v) => -4 + 3.2 * Math.exp(-0.5 * v * v)), { c: "green", w: 2.5 });
      const ss = Math.sqrt(1 - rho * rho); P.line(g, g.map((v) => -4 + 3.2 * Math.exp(-0.5 * (v / ss) ** 2)), { c: "red", w: 2.5, dash: [6, 4] });
      P.hline(0, { c: "red", w: 1 });
      k.set("m", "1.000"); k.set("s", ss.toFixed(3)); k.set("r", ss.toFixed(2));
    }
    root._redraw = draw; draw();
  };

  /* ---------------------------------------------------------- 7. grid posterior 2-D */
  W.grid2d = (root) => {
    UI.head(root, "POSTERIOR ON A GRID · (μ, σ) OF 12 GAUSSIAN DATA");
    const k = UI.kpis(root, [["n", "grid points K²"], ["u", "points holding 99%"], ["f", "useful fraction"], ["mu", "posterior μ"], ["sg", "posterior σ"]]);
    const P = UI.canvas(root, 330, { h: (w) => Math.min(380, w * 0.6) });
    const c = UI.ctrls(root); let K = 40; const rng = M.rng(77), x = []; for (let i = 0; i < 12; i++) x.push(3 + 1.5 * rng.normal());
    UI.slider(c, { label: "points per axis K", min: 5, max: 140, step: 1, value: K, on: (v) => { K = v; draw(); } });
    function draw() {
      const mu = M.linspace(0, 6, K), sg = M.linspace(0.3, 4, K), lp = [];
      let mx = -Infinity;
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) { let s = 0; for (const v of x) s += -0.5 * ((v - mu[i]) / sg[j]) ** 2 - Math.log(sg[j]); lp.push(s); if (s > mx) mx = s; }
      const p = lp.map((v) => Math.exp(v - mx)), Z = M.sum(p), pn = p.map((v) => v / Z);
      const ord = pn.map((v, i) => i).sort((a, b) => pn[b] - pn[a]); let acc = 0, nU = 0; const useful = new Uint8Array(pn.length);
      for (const i of ord) { acc += pn[i]; useful[i] = 1; nU++; if (acc >= 0.99) break; }
      P.begin({ xlim: [0, 6], ylim: [0.3, 4], xlabel: "μ", ylabel: "σ" });
      const dmu = 6 / (K - 1), dsg = 3.7 / (K - 1), pm = Math.max(...pn);
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
        const q = i * K + j; P.rect(mu[i] - dmu / 2, sg[j] - dsg / 2, mu[i] + dmu / 2, sg[j] + dsg / 2, { c: "purple", a: 0.08 + 0.85 * Math.sqrt(pn[q] / pm), stroke: useful[q] ? "orange" : null, w: 1 });
      }
      if (K <= 40) { const xs = [], ys = []; for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) { xs.push(mu[i]); ys.push(sg[j]); } P.pts(xs, ys, { c: "ink", r: 1.2, a: 0.5 }); }
      let m1 = 0, s1 = 0; for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) { m1 += mu[i] * pn[i * K + j]; s1 += sg[j] * pn[i * K + j]; }
      k.set("n", (K * K).toLocaleString()); k.set("u", nU); k.set("f", (100 * nU / (K * K)).toFixed(2) + "%"); k.set("mu", m1.toFixed(3)); k.set("sg", s1.toFixed(3));
    }
    root._redraw = draw; draw();
    UI.note(root, "Orange outlines: the cells that hold 99% of the probability. Increase K: the useful fraction falls. In 3-D the lecture's grid had 0.25% useful points; in 10-D almost none.");
  };

  /* ---------------------------------------------------------- 7b. cost calculator */
  W.cost = (root) => {
    UI.head(root, "THE COST OF A GRID · K^d EVALUATIONS");
    const k = UI.kpis(root, [["e", "evaluations K^d"], ["t", "time on one core"]]);
    const P = UI.canvas(root, 260); const c = UI.ctrls(root);
    let K = 50, d = 5, te = 30e-6;
    UI.slider(c, { label: "points per axis K", min: 5, max: 200, step: 1, value: K, on: (v) => { K = v; draw(); } });
    UI.slider(c, { label: "dimension d", min: 1, max: 12, step: 1, value: d, on: (v) => { d = v; draw(); } });
    UI.slider(c, { label: "time per likelihood", min: -7, max: 0, step: 0.1, value: Math.log10(te), fmt: (v) => M.human(Math.pow(10, v)), on: (v) => { te = Math.pow(10, v); draw(); } });
    function draw() {
      const ds = M.linspace(1, 12, 12), tt = ds.map((x) => Math.pow(K, x) * te);
      P.begin({ xlim: [1, 12], ylim: [1e-6, 1e18], ylog: true, xlabel: "dimension d", ylabel: "time [s]" });
      [[3600, "1 hour"], [86400 * 365, "1 year"], [4.35e17, "age of the Universe"]].forEach(([v, l]) => P.hline(v, { c: "faint", label: l }));
      P.line(ds, tt, { c: "red", w: 2.5 }); P.pts(ds, tt, { c: "red", r: 3 }); P.pts([d], [Math.pow(K, d) * te], { c: "orange", r: 7, stroke: "ink" });
      k.set("e", M.fmt(Math.pow(K, d), 3)); k.set("t", M.human(Math.pow(K, d) * te));
    }
    root._redraw = draw; draw();
  };

  /* ---------------------------------------------------------- 8. the knee grid in the browser */
  W.kneegrid = (root) => {
    UI.head(root, "knee_grid.f90 IN YOUR BROWSER · 3-D POSTERIOR ON ONE 64 s RECORD");
    const k = UI.kpis(root, [["g", "grid points"], ["t", "time"], ["a", "α: mean ± std"], ["v", "fₖ: median [68%]"], ["u", "points with 99% mass"]]);
    const box = UI.el("div", "split"); root.append(box);
    const A = UI.canvas(box, 280), B = UI.canvas(box, 280);
    const b = UI.btns(root); let seed = 12;
    UI.button(b, "run the grid", () => run(), "go"); UI.button(b, "new record + run", () => { seed++; run(); });
    function run() {
      const d = kneeData(64, seed), nu = 21, nv = 81, na = 61, t0 = performance.now();
      const u = M.linspace(-3.25, -2.75, nu), v = M.linspace(-2, Math.log10(50), nv), a = M.linspace(0, 3, na);
      const lp = new Float64Array(nu * nv * na); let mx = -Infinity;
      for (let ka = 0; ka < na; ka++) for (let jv = 0; jv < nv; jv++) {
        let As = 0, Bs = 0; const L10 = v[jv] * Math.LN10;
        for (let l = 0; l < d.n; l++) { const g = 1 + Math.exp(a[ka] * (L10 - d.lnf[l])); As += Math.log(g); Bs += d.P[l] / g; }
        for (let iu = 0; iu < nu; iu++) { const s2 = Math.pow(10, u[iu]), val = -(d.n * Math.log(s2) + As + Bs / s2); lp[(ka * nv + jv) * nu + iu] = val; if (val > mx) mx = val; }
      }
      let Z = 0; for (let i = 0; i < lp.length; i++) { lp[i] = Math.exp(lp[i] - mx); Z += lp[i]; }
      const mv = new Float64Array(nv), ma = new Float64Array(na), mva = new Float64Array(nv * na);
      for (let ka = 0; ka < na; ka++) for (let jv = 0; jv < nv; jv++) for (let iu = 0; iu < nu; iu++) { const p = lp[(ka * nv + jv) * nu + iu] / Z; mv[jv] += p; ma[ka] += p; mva[ka * nv + jv] += p; }
      const sorted = Array.from(lp).sort((x, y) => y - x); let acc = 0, nU = 0; for (const p of sorted) { acc += p / Z; nU++; if (acc >= 0.99) break; }
      const t1 = performance.now();
      const meanA = a.reduce((s, x, i) => s + x * ma[i], 0), sdA = Math.sqrt(a.reduce((s, x, i) => s + (x - meanA) ** 2 * ma[i], 0));
      const vm = Array.from(mv), med = M.gridQuantile(v, vm, 0.5), q1 = M.gridQuantile(v, vm, 0.16), q2 = M.gridQuantile(v, vm, 0.84);
      k.set("g", lp.length.toLocaleString()); k.set("t", (t1 - t0).toFixed(0) + " ms"); k.set("a", `${meanA.toFixed(3)} ± ${sdA.toFixed(3)}`);
      k.set("v", `${Math.pow(10, med).toFixed(2)} Hz [${Math.pow(10, q1).toFixed(2)}, ${Math.pow(10, q2).toFixed(2)}]`); k.set("u", `${nU} (${(100 * nU / lp.length).toFixed(2)}%)`);
      const draw = () => {
        A.begin({ xlim: [0, 3], ylim: [-2, Math.log10(50)], xlabel: "α", ylabel: "log₁₀ fₖ", title: "2-D marginal (fₖ, α)" });
        const pm = Math.max(...mva), da = 3 / (na - 1), dv = (Math.log10(50) + 2) / (nv - 1);
        for (let ka = 0; ka < na; ka++) for (let jv = 0; jv < nv; jv++) { const p = mva[ka * nv + jv] / pm; if (p > 1e-3) A.rect(a[ka] - da / 2, v[jv] - dv / 2, a[ka] + da / 2, v[jv] + dv / 2, { c: "green", a: Math.min(1, 0.1 + p) }); }
        A.pts([1], [0], { c: "red", r: 6, stroke: "ink" });
        B.begin({ xlim: [0, 3], ylim: [0, Math.max(...ma) * 1.15], xlabel: "α  (red: truth)", ylabel: "marginal", title: "1-D marginal of α" });
        B.fill(a, 0, Array.from(ma), { c: "purple", a: 0.25 }); B.line(a, Array.from(ma), { c: "purple" }); B.vline(1, { c: "red", dash: [] });
      };
      root._redraw = draw; draw();
    }
    run();
  };
})();
