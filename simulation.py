"""Geometric Brownian Motion path simulation.

dS_t = mu * S_t * dt + sigma * S_t * dW_t

Simulated with the exact solution, so the scheme carries no
discretisation bias: S_{t+dt} = S_t * exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z).

For pricing, mu must be the risk-neutral drift (r - q), not the
historical return.
"""

import numpy as np


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths=1, seed=None):
    """Simulate Geometric Brownian Motion paths.

    Parameters
    
    S0 : float
        Initial price, strictly positive.
    mu : float
        Drift. Use r - q for risk-neutral pricing.
    sigma : float
        Annualised volatility, strictly positive.
    T : float
        Horizon in years.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of independent paths (default 1).
    seed : int, optional
        Seed for reproducibility.

    Returns
    
    numpy.ndarray
        Array of shape (n_paths, n_steps + 1), including S0 at t=0.
    """
    if S0 <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S0, sigma and T must be strictly positive")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be at least 1")

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(
        (n_paths, n_steps)
    )

    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


def time_grid(T, n_steps):
    """Return the time grid in years matching simulate_gbm output."""
    return np.linspace(0.0, T, n_steps + 1)
