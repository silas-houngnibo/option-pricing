"""Monte Carlo pricing of European options under geometric Brownian motion.

The terminal price is simulated in a single step using the exact solution
of the SDE, so the estimator carries no discretisation bias - only
statistical error, which the reported standard error quantifies.

Antithetic variates are available: pairing each draw Z with -Z leaves the
estimator unbiased and reduces variance whenever the payoff is monotone
in Z, which is the case for vanilla calls and puts.
"""

import numpy as np


def monte_carlo_price(S, K, T, r, sigma, option_type="call", n_paths=100_000,
                      q=0.0, antithetic=True, seed=None):
    """Price a European option by Monte Carlo simulation.

    Parameters
    
    S : float
        Spot price, strictly positive.
    K : float
        Strike, strictly positive.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate, continuously compounded.
    sigma : float
        Annualised volatility, strictly positive.
    option_type : {"call", "put"}
    n_paths : int
        Number of simulated terminal prices.
    q : float
        Continuous dividend yield (default 0).
    antithetic : bool
        Use antithetic variates for variance reduction (default True).
    seed : int, optional
        Seed for reproducibility.

    Returns
    
    tuple of float
        (price, standard_error).
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T and sigma must be strictly positive")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")

    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    S_T = S * np.exp(drift + diffusion)

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    discounted = np.exp(-r * T) * payoff

    if antithetic:
        # Pairs are correlated: average within each pair before computing
        # the standard error, otherwise it is understated.
        pairs = discounted.reshape(2, -1).mean(axis=0)
        return pairs.mean(), pairs.std(ddof=1) / np.sqrt(pairs.size)

    return discounted.mean(), discounted.std(ddof=1) / np.sqrt(n_paths)
