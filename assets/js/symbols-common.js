/* Symbols shared by every lesson. Each entry: n = name, t = type tag, s = significance,
   py / f = how it looks in Python / Fortran. Lessons add their own in window.SYMBOLS. */
window.SYMBOLS_COMMON = {
  theta: { n: "θ — the parameters", t: "unknown", s: "The numbers of the model we want to learn, e.g. θ = (σ², f<sub>k</sub>, α) for the knee noise. A vector when there are several.", py: "theta = np.array([-3.0, 0.0, 1.0])", f: "real(dp) :: theta(3)" },
  x: { n: "x — the data", t: "observed", s: "What we measured: a set of numbers x = (x₁, …, x_N). Fixed once observed (Bayesian view) or one random draw among possible repetitions (frequentist view).", py: "x = np.loadtxt('data.txt')", f: "real(dp), allocatable :: x(:)" },
  N: { n: "N — number of samples / data", t: "count", s: "How many data points, trials or random points. Errors of averages shrink as 1/√N.", py: "N = x.size", f: "N = size(x)" },
  sigma: { n: "σ — standard deviation", t: "spread", s: "The typical size of a fluctuation around the mean. σ² is the variance. For a measurement: its error bar.", py: "sigma = x.std(ddof=1)", f: "sigma = sqrt(sum((x-sum(x)/n)**2)/(n-1))" },
  mu: { n: "μ — mean / expected value", t: "centre", s: "The centre of a distribution: the value the average tends to with many samples.", py: "mu = x.mean()", f: "mu = sum(x)/size(x)" },
  P: { n: "P(·) — probability", t: "number in [0,1]", s: "Probability of an event: 0 = impossible, 1 = certain. Same rules (Kolmogorov) for every interpretation.", py: "p = n_event / n_total", f: "p = real(n_event,dp)/n_total" },
  E: { n: "𝔼[·] — expectation (average)", t: "operator", s: "The probability-weighted average of a quantity: 𝔼[g] = ∫ g(θ) p(θ) dθ. With samples it becomes a plain mean.", py: "Eg = np.mean(g(samples))", f: "Eg = sum(g)/n" },
  integral: { n: "∫ … dθ — integral", t: "operator", s: "A continuous sum. Every Bayesian answer (means, intervals, marginals) is an integral over the posterior — Lecture 13 computes them with random points.", py: "from scipy.integrate import quad\nI, err = quad(f, a, b)", f: "! a sum on a grid or a Monte Carlo average\nI = (b-a)*sum(f(x))/n" },
  sum: { n: "Σ — sum", t: "operator", s: "Add the terms for every index. On a grid, integrals become sums.", py: "s = np.sum(a)", f: "s = sum(a)" },
  prod: { n: "∏ — product", t: "operator", s: "Multiply the terms for every index. Independent data: the likelihood is the product of each datum's probability. Take the log to turn it into a sum (no underflow).", py: "logL = np.sum(np.log(p_i))   # never np.prod(p_i)", f: "logL = sum(log(p_i))" },
  propto: { n: "∝ — proportional to", t: "relation", s: "Equal up to a constant factor that does not depend on the variable of interest (here θ). Lets us ignore normalisations.", py: "post = like * prior   # then post /= post.sum()", f: "post = like*prior; post = post/sum(post)" },
  d: { n: "d — dimension", t: "count", s: "Number of parameters (or of integration variables). Grids cost K^d; Monte Carlo errors do not depend on d directly.", py: "d = theta.size", f: "d = size(theta)" },
  given: { n: "| — “given”", t: "conditioning", s: "p(A | B) is the probability of A when we know B is true. Read the bar as “given”.", py: "# p_A_given_B = p_AB / p_B", f: "p_A_given_B = p_AB / p_B" },
  ln: { n: "ln — natural logarithm", t: "function", s: "Logs turn products into sums and keep tiny numbers representable: exp(−745) is already 0 in double precision.", py: "np.log(x)", f: "log(x)" },
  exp: { n: "exp — exponential", t: "function", s: "e^x. Always subtract the maximum before exponentiating log-probabilities.", py: "np.exp(lp - lp.max())", f: "exp(lp - maxval(lp))" }
};
