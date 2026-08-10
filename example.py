"""Demonstration script for the option-pricing repository.

Runs every model, checks each one against a result that can be verified
independently, and writes the figures used in the README.

    python example.py

Expected module names in this folder:
    black_scholes.py      black_scholes_price
    simulation.py         simulate_gbm, time_grid
    monte_carlo.py        monte_carlo_price
    heston.py             simulate_heston, heston_price, feller_condition
    implied_volatility.py implied_vol
"""

import os
from math import exp

import numpy as np
import matplotlib

matplotlib.use("Agg")  # write files without opening a window
import matplotlib.pyplot as plt
from black_scholes import black_scholes_price
from simulation import simulate_gbm, time_grid
from monte_carlo import monte_carlo_price
from heston import simulate_heston, heston_price, feller_condition
from implied_volatility import implied_vol



SEED = 42
IMAGES = "images"


# ----------------------------------------------------------------------
# 1. Black-Scholes: put-call parity
# ----------------------------------------------------------------------
def check_put_call_parity():
    """C - P = S - K exp(-rT). Residual should be at machine precision."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.02, 0.20

    call = black_scholes_price(S, K, T, r, sigma, "call")
    put = black_scholes_price(S, K, T, r, sigma, "put")
    gap = (call - put) - (S - K * exp(-r * T))

    print("1. Black-Scholes")
    print(f"   call {call:.4f}   put {put:.4f}")
    print(f"   put-call parity residual: {gap:.2e}")


# ----------------------------------------------------------------------
# 2. Monte Carlo against the closed form
# ----------------------------------------------------------------------
def check_monte_carlo():
    """The closed-form price should sit inside the Monte Carlo 95% interval."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.02, 0.20

    exact = black_scholes_price(S, K, T, r, sigma, "call")
    mc, se = monte_carlo_price(S, K, T, r, sigma, "call",
                               n_paths=200_000, antithetic=True, seed=SEED)
    plain, se_plain = monte_carlo_price(S, K, T, r, sigma, "call",
                                        n_paths=200_000, antithetic=False,
                                        seed=SEED)

    print("\n2. Monte Carlo")
    print(f"   closed form           {exact:.4f}")
    print(f"   MC antithetic         {mc:.4f} +/- {1.96 * se:.4f}")
    print(f"   MC plain              {plain:.4f} +/- {1.96 * se_plain:.4f}")
    print(f"   variance reduction    {se_plain / se:.1f}x lower standard error")
    print(f"   exact within 95% CI:  {abs(mc - exact) < 1.96 * se}")


# ----------------------------------------------------------------------
# 3. Implied volatility round-trip
# ----------------------------------------------------------------------
def check_implied_vol():
    """Price at a known sigma, invert, and recover the same sigma."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.02, 0.20

    price = black_scholes_price(S, K, T, r, sigma, "call")
    recovered = implied_vol(S, K, T, r, price, "call")

    print("\n3. Implied volatility")
    print(f"   input sigma     {sigma:.6f}")
    print(f"   recovered sigma {recovered:.6f}")
    print(f"   absolute error  {abs(recovered - sigma):.2e}")


# ----------------------------------------------------------------------
# 4. Heston degenerating into Black-Scholes
# ----------------------------------------------------------------------
def check_heston_limit():
    """With xi -> 0 and v0 = theta, variance is frozen and Heston is BS."""
    S, K, T, r = 100.0, 100.0, 1.0, 0.02
    theta = 0.04  # 20% volatility

    h, se = heston_price(S, K, v0=theta, r=r, kappa=2.0, theta=theta,
                         xi=1e-6, rho=-0.7, T=T, option_type="call",
                         n_steps=100, n_paths=50_000, seed=SEED)
    bs = black_scholes_price(S, K, T, r, np.sqrt(theta), "call")

    print("\n4. Heston")
    print(f"   Heston (xi -> 0) {h:.4f} +/- {1.96 * se:.4f}")
    print(f"   Black-Scholes    {bs:.4f}")
    print(f"   consistent:      {abs(h - bs) < 1.96 * se}")
    print(f"   Feller condition (kappa=2, theta=0.04, xi=0.5): "
          f"{feller_condition(2.0, 0.04, 0.5)}")


# ----------------------------------------------------------------------
# Figure 1: GBM paths
# ----------------------------------------------------------------------
def figure_gbm_paths():
    T, n_steps = 1.0, 252
    paths = simulate_gbm(S0=100.0, mu=0.02, sigma=0.20, T=T,
                         n_steps=n_steps, n_paths=30, seed=SEED)
    t = time_grid(T, n_steps)

    plt.figure(figsize=(9, 5))
    plt.plot(t, paths.T, linewidth=0.8, alpha=0.7)
    plt.title("Geometric Brownian motion 30 paths, sigma = 20%")
    plt.xlabel("Time (years)")
    plt.ylabel("Price")
    plt.grid(alpha=0.3)
    plt.savefig(f"{IMAGES}/gbm_paths.png", dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# Figure 2: Heston price and volatility paths
# ----------------------------------------------------------------------
def figure_heston_paths():
    T, n_steps = 1.0, 252
    theta = 0.04
    prices, variances = simulate_heston(
        S0=100.0, v0=0.09, r=0.02, kappa=2.0, theta=theta,
        xi=0.5, rho=-0.7, T=T, n_steps=n_steps, n_paths=20, seed=SEED,
    )
    t = time_grid(T, n_steps)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    top.plot(t, prices.T, linewidth=0.8, alpha=0.7)
    top.set_ylabel("Price")
    top.set_title("Heston  20 paths under stochastic volatility")
    top.grid(alpha=0.3)

    bottom.plot(t, np.sqrt(np.maximum(variances, 0.0)).T,
                linewidth=0.8, alpha=0.7)
    bottom.axhline(np.sqrt(theta), color="black", linestyle="--",
                   label="long-run volatility  sqrt(theta)")
    bottom.set_ylabel("Instantaneous volatility")
    bottom.set_xlabel("Time (years)")
    bottom.legend()
    bottom.grid(alpha=0.3)

    plt.savefig(f"{IMAGES}/heston_paths.png", dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# Figure 3: Monte Carlo convergence
# ----------------------------------------------------------------------
def figure_mc_convergence():
    """RMSE over 30 independent runs per sample size.

    A single run per size measures one realisation of a random error: the
    sign flips at random and the log scale turns every near-zero crossing
    into a spike. Averaging squared errors over repetitions estimates the
    quantity that actually decays as 1/sqrt(n).
    """
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.02, 0.20
    exact = black_scholes_price(S, K, T, r, sigma, "call")

    sizes = np.logspace(2, 5, 12).astype(int)
    n_repeats = 30
    rmse = []

    for n in sizes:
        errors = [
            monte_carlo_price(S, K, T, r, sigma, "call", n_paths=int(n),
                              antithetic=False, seed=SEED + j)[0] - exact
            for j in range(n_repeats)
        ]
        rmse.append(np.sqrt(np.mean(np.square(errors))))

    reference = rmse[0] * np.sqrt(sizes[0] / sizes)

    plt.figure(figsize=(9, 5))
    plt.loglog(sizes, rmse, "o-", markersize=5,
               label=f"RMSE over {n_repeats} runs")
    plt.loglog(sizes, reference, "--", color="black",
               label=r"$1/\sqrt{n}$ reference")
    plt.title("Monte Carlo convergence against the closed-form price")
    plt.xlabel("Number of paths")
    plt.ylabel("Root mean squared error")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.savefig(f"{IMAGES}/mc_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# Figure 4: the smile Heston produces and Black-Scholes cannot
# ----------------------------------------------------------------------
def figure_volatility_smile():
    """Price a strike ladder under Heston, then read it back through
    Black-Scholes. A flat line would mean the two models agree; the curve
    is the exact sense in which they do not.

    All strikes are priced off a single set of paths (common random
    numbers), which keeps the curve smooth instead of noisy.
    """
    S0, r, T = 100.0, 0.02, 1.0
    v0, kappa, theta, xi, rho = 0.09, 2.0, 0.04, 0.5, -0.7
    n_steps, n_paths = 100, 50_000

    prices, _ = simulate_heston(S0, v0, r, kappa, theta, xi, rho, T,
                                n_steps=n_steps, n_paths=n_paths, seed=SEED)
    S_T = prices[:, -1]
    discount = np.exp(-r * T)

    strikes = np.arange(70.0, 131.0, 5.0)
    smile = np.full(strikes.size, np.nan)

    for i, K in enumerate(strikes):
        heston_call = discount * np.maximum(S_T - K, 0.0).mean()
        try:
            smile[i] = implied_vol(S0, K, T, r, heston_call, "call")
        except ValueError:
            pass  # outside the no-arbitrage bounds, leave as NaN

    plt.figure(figsize=(9, 5))
    plt.plot(strikes, 100 * smile, "o-", label="implied volatility (Heston prices)")
    plt.axhline(100 * np.sqrt(theta), color="black", linestyle="--",
                label="Black-Scholes, single volatility")
    plt.axvline(S0, color="grey", linewidth=0.8, alpha=0.6)
    plt.title(f"Volatility smile implied by Heston (rho = {rho})")
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{IMAGES}/smile.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n5. Volatility smile")
    for K, iv in zip(strikes, smile):
        print(f"   K = {K:6.1f}   implied vol = {100 * iv:5.2f}%")


def main():
    os.makedirs(IMAGES, exist_ok=True)

    check_put_call_parity()
    check_monte_carlo()
    check_implied_vol()
    check_heston_limit()

    figure_gbm_paths()
    figure_heston_paths()
    figure_mc_convergence()
    figure_volatility_smile()

    print(f"\nFigures written to {IMAGES}/")


if __name__ == "__main__":
    main()
