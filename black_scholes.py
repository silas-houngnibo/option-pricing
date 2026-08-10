"""Black-Scholes closed-form pricing for European options.

Assumes constant volatility and constant rates.
The constant-volatility assumption is the one the market
contradicts see implied_volatility.py for the resulting smile.
"""

from math import log, sqrt, exp
from scipy.stats import norm


def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Price a European option under Black-Scholes.

    Parameters
   
    S : float
        Spot price of the underlying.
    K : float
        Strike.
    T : float
        Time to maturity, in years.
    r : float
        Risk-free rate, continuously compounded.
    sigma : float
        Annualised volatility.
    option_type : {"call", "put"}
    q : float
        Continuous dividend yield (default 0).

    Returns
    
    float
        Option price.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T and sigma must be strictly positive")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    discounted_spot = S * exp(-q * T)
    discounted_strike = K * exp(-r * T)

    if option_type == "call":
        return discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    return discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
