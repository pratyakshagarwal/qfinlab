import numpy as np
from numpy.random import default_rng
from typing import Optional

class MonteCarloSimulation:
    """
    Monte Carlo simulation engine for Geometric Brownian Motion (GBM).

    S_t = S_0 * exp((r - 0.5 * sigma^2) * t + sigma * W_t)
    """

    def __init__(self, S0: float, r: float, sigma: float, seed: Optional[int] = None):
        """
        Initialize the Monte Carlo simulator.

        Args:
            S0 (float): Initial stock price.
            r (float): Risk-free rate.
            sigma (float): Volatility of the asset.
            seed (int, optional): Random seed for reproducibility.
        """
        self.S0, self.r, self.sigma = S0, r, sigma
        self.rng = default_rng(seed)

    def simulate_euler(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        """
        Simulate GBM paths using Euler discretization.

        Args:
            T (float): Time horizon.
            n_paths (int): Number of simulated paths.
            n_steps (int): Number of time steps.

        Returns:
            np.ndarray: Simulated paths of shape (n_steps+1, n_paths).
        """
        dt = T / n_steps
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = self.S0

        for t in range(1, n_steps + 1):
            Z = self.rng.standard_normal(n_paths)
            paths[t] = paths[t - 1] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z
            )

        return paths

    def simulate_exact(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        """
        Simulate GBM paths using the exact log-normal solution.

        Args:
            T (float): Time horizon.
            n_paths (int): Number of simulated paths.
            n_steps (int): Number of time steps.

        Returns:
            np.ndarray: Simulated paths of shape (n_steps+1, n_paths).
        """
        dt = T / n_steps
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = self.S0

        for t in range(1, n_steps + 1):
            mu = (self.r - 0.5 * self.sigma ** 2) * dt
            sigma_dt = self.sigma * np.sqrt(dt)
            paths[t] = paths[t - 1] * np.exp(self.rng.normal(mu, sigma_dt, n_paths))

        return paths