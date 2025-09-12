import numpy as np
from numpy.random import default_rng
from typing import Optional

class MertonJumpDiffusion:
    """
    Monte Carlo simulation of Merton's Jump-Diffusion model.
    
    dS_t = (r - r_J) S_t dt + sigma S_t dW_t + J_t S_t
    """

    def __init__(self, S0: float, r: float, sigma: float, lamb: float, 
                 mu: float, delta: float, seed: Optional[int] = None):
        """
        Args:
            S0 (float): Initial stock price.
            r (float): Risk-free rate.
            sigma (float): Volatility of diffusion.
            lamb (float): Jump intensity (Poisson rate).
            mu (float): Mean of log-normal jump size.
            delta (float): Std dev of log-normal jump size.
            seed (int, optional): Random seed.
        """
        self.S0, self.r, self.sigma = S0, r, sigma
        self.lamb, self.mu, self.delta = lamb, mu, delta
        self.rng = default_rng(seed)

        # Jump risk adjustment
        self.rj = lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)

    def simulate_paths(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        """
        Simulate paths of Merton's jump-diffusion model.

        Args:
            T (float): Time horizon.
            n_paths (int): Number of simulated paths.
            n_steps (int): Number of time steps.

        Returns:
            np.ndarray: Simulated paths (n_steps+1, n_paths).
        """
        dt = T / n_steps
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = self.S0

        # Generate random numbers
        Z1 = self.rng.standard_normal((n_steps + 1, n_paths))
        Z2 = self.rng.standard_normal((n_steps + 1, n_paths))
        N  = self.rng.poisson(self.lamb * dt, (n_steps + 1, n_paths))

        for t in range(1, n_steps + 1):
            drift = (self.r - self.rj - 0.5 * self.sigma ** 2) * dt
            diffusion = self.sigma * np.sqrt(dt) * Z1[t]
            jumps = (np.exp(self.mu + self.delta * Z2[t]) - 1) * N[t]

            paths[t] = paths[t - 1] * np.exp(drift + diffusion) + paths[t - 1] * jumps

            paths[t] = np.maximum(paths[t], 0)  # enforce positivity

        return paths