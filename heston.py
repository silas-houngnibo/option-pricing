"""Heston stochastic volatility model  simulation and Monte Carlo pricing.

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2_t
    corr(dW1, dW2) = rho

Variance is discretised with the full truncation Euler scheme: v is floored
at zero before use, which keeps the square root defined when the Feller
condition 2*kappa*theta >= xi^2 fails - as it usually does on calibrated
market parameters.

Negative rho tilts the risk-neutral density to the left and produces the
downward-sloping smile observed on equity indices, which Black-Scholes
cannot reproduce with a single volatility.
"""

import numpy as np


def simulate_heston(S0, v0, r, kappa, theta, xi, rho, T,
                    n_steps, n_paths=1, q=0.0, seed=None):
    """Simulate price and variance paths under the Heston model.

    Parameters
    
    S0 : float
        Initial spot, strictly positive.
    v0 : float
        Initial variance (not volatility), strictly positive.
    r : float
        Risk-free rate, continuously compounded.
    kappa : float
        Mean reversion speed of the variance.
    theta : float
        Long-run variance.
    xi : float
        Volatility of volatility.
    rho : float
        Correlation between the two Brownian motions, in [-1, 1].
    T : float
        Maturity in years.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of independent paths (default 1).
    q : float
        Continuous dividend yield (default 0).
    seed : int, optional
        Seed for reproducibility.

    Returns
   
    tuple of numpy.ndarray
        (prices, variances), each of shape (n_paths, n_steps + 1).
    """
    if S0 <= 0 or v0 <= 0 or T <= 0:
        raise ValueError("S0, v0 and T must be strictly positive")
    if kappa <= 0 or theta <= 0 or xi <= 0:
        raise ValueError("kappa, theta and xi must be strictly positive")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    prices = np.empty((n_paths, n_steps + 1))
    variances = np.empty((n_paths, n_steps + 1))
    prices[:, 0] = S0
    variances[:, 0] = v0

    for i in range(1, n_steps + 1):
        z2 = rng.standard_normal(n_paths)
        z_ind = rng.standard_normal(n_paths)
        z1 = rho * z2 + np.sqrt(1.0 - rho**2) * z_ind

        v_prev = np.maximum(variances[:, i - 1], 0.0)
        sqrt_v = np.sqrt(v_prev)

        variances[:, i] = (
            variances[:, i - 1]
            + kappa * (theta - v_prev) * dt
            + xi * sqrt_v * sqrt_dt * z2
        )

        prices[:, i] = prices[:, i - 1] * np.exp(
            (r - q - 0.5 * v_prev) * dt + sqrt_v * sqrt_dt * z1
        )

    return prices, variances


def heston_price(S0, K, v0, r, kappa, theta, xi, rho, T,
                 option_type="call", n_steps=252, n_paths=100_000,
                 q=0.0, seed=None):
    """Price a European option by Monte Carlo under Heston.

    Only the terminal price matters for a European payoff, but the whole
    path is simulated because the variance is path-dependent.

    Returns
    
    tuple of float
        (price, standard_error). The standard error is the Monte Carlo
        confidence half-width divided by 1.96 - report it, a Monte Carlo
        price without it is meaningless.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    prices, _ = simulate_heston(
        S0, v0, r, kappa, theta, xi, rho, T,
        n_steps=n_steps, n_paths=n_paths, q=q, seed=seed,
    )
    S_T = prices[:, -1]

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    discounted = np.exp(-r * T) * payoff
    return discounted.mean(), discounted.std(ddof=1) / np.sqrt(n_paths)


def feller_condition(kappa, theta, xi):
    """Return True when 2*kappa*theta >= xi**2, i.e. variance stays positive."""
    return 2.0 * kappa * theta >= xi**2
