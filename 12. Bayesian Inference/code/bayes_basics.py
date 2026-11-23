"""
bayes_basics.py  --  the small numerical examples of Lecture 12, one function each.
Run:  python bayes_basics.py
"""
import numpy as np


def urns(draws="BB", prior_A1=0.5):
    """Two urns: A1 has 3 blue / 7 red, A2 has 6 blue / 4 red (with replacement).
    Yesterday's posterior is today's prior."""
    p_blue = {"A1": 0.3, "A2": 0.6}
    post = np.array([prior_A1, 1 - prior_A1])
    for d in draws:
        like = np.array([p_blue["A1"], p_blue["A2"]]) if d == "B" else \
               np.array([1 - p_blue["A1"], 1 - p_blue["A2"]])
        evidence = np.sum(like * post)           # law of total probability
        post = like * post / evidence            # Bayes' theorem
        print(f"  after '{d}': P(A1|data) = {post[0]:.4f}   P(A2|data) = {post[1]:.4f}")
    return post


def alarm(prevalence=0.01, sensitivity=0.99, false_alarm=0.05):
    """P(real event | alarm): the base-rate trap."""
    num = sensitivity * prevalence
    return num / (num + false_alarm * (1 - prevalence))


def gaussian_update(mu_L, s_L, mu_0, s_0):
    """Gaussian likelihood x Gaussian prior -> Gaussian posterior (precisions add)."""
    w_L, w_0 = 1 / s_L**2, 1 / s_0**2
    s_post = 1 / np.sqrt(w_L + w_0)
    mu_post = s_post**2 * (w_L * mu_L + w_0 * mu_0)
    return mu_post, s_post


def mle_variance_bias(N=5, sigma2=4.0, n_sets=200_000, seed=19):
    """The MLE of a Gaussian variance divides by N: biased by (N-1)/N."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, np.sqrt(sigma2), size=(n_sets, N))
    s2_ml = x.var(axis=1, ddof=0)      # 1/N
    s2_un = x.var(axis=1, ddof=1)      # 1/(N-1)
    return s2_ml.mean(), s2_un.mean(), (N - 1) / N * sigma2


if __name__ == "__main__":
    print("Urns, two blue draws (lecture: 1/3 then 0.2):")
    urns("BB")
    print(f"\nAlarm: P(real | alarm) = {alarm():.4f}")
    m, s = gaussian_update(0.82, 0.12, 1.00, 0.10)
    print(f"\nalpha: likelihood 0.82+-0.12, prior 1.00+-0.10 -> posterior {m:.3f} +- {s:.3f}")
    ml, un, th = mle_variance_bias()
    print(f"\nVariance N=5, sigma2=4: <MLE> = {ml:.3f} (theory {th:.3f}),  <1/(N-1)> = {un:.3f}")
