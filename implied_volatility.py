"""Implied volatility by bisection on the Black-Scholes price.

The Black-Scholes price is strictly increasing in sigma: vega is positive
everywhere. So for a price lying within the no-arbitrage bounds, there is
exactly one volatility reproducing it, and bracketing it is enough to find
it - no derivative needed, no divergence possible.

The function whose root we seek is the residual

    f(sigma) = BS(sigma) - market_price

negative when the model underprices the market, positive when it
overprices it. Bisection halves the bracket at each step, so the error
after n iterations is (sigma_high - sigma_low) / 2^n: linear convergence,
but unconditionally safe.
"""

from black_scholes import black_scholes_price


def residual(sigma, S, K, T, r, market_price, option_type="call"):
    """Gap between the Black-Scholes price at sigma and the observed price.

    Parameters
    
    sigma : float
        Candidate volatility.
    S, K, T, r : float
        Spot, strike, maturity in years, risk-free rate.
    market_price : float
        Observed option price.
    option_type : {"call", "put"}

    Returns
    
    float
        BS(sigma) - market_price. Strictly increasing in sigma.
    """
    return black_scholes_price(S, K, T, r, sigma, option_type) - market_price


def implied_vol(S, K, T, r, market_price, option_type="call",
                sigma_low=1e-6, sigma_high=10.0, tol=1e-6, max_iter=200):
    """Recover the volatility implied by an observed option price.

    Parameters
    
    S, K, T, r : float
        Spot, strike, maturity in years, risk-free rate.
    market_price : float
        Observed option price.
    option_type : {"call", "put"}
    sigma_low, sigma_high : float
        Initial bracket. Must contain the root.
    tol : float
        Absolute tolerance on the price residual.
    max_iter : int
        Iteration cap.

    Returns
    
    float
        Implied volatility.

    Raises
    
    ValueError
        If the root is not bracketed, or if no root is found within
        max_iter iterations.
    """
    f_low = residual(sigma_low, S, K, T, r, market_price, option_type)
    f_high = residual(sigma_high, S, K, T, r, market_price, option_type)

    if f_low * f_high > 0:
        raise ValueError(
            f"price {market_price:.4f} not bracketed on "
            f"[{sigma_low}, {sigma_high}] - no implied volatility here"
        )

    for _ in range(max_iter):
        sigma_mid = 0.5 * (sigma_low + sigma_high)
        f_mid = residual(sigma_mid, S, K, T, r, market_price, option_type)

        if abs(f_mid) < tol:
            return sigma_mid

        if f_low * f_mid < 0:
            sigma_high = sigma_mid
        else:
            sigma_low, f_low = sigma_mid, f_mid

    raise ValueError(f"no convergence after {max_iter} iterations")
