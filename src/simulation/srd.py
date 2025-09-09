import numpy as np
import numpy.random as npr
from typing import Tuple

class SquareRootDiffusion:
    """
    Square Root Diffusion (CIR process) simulator.

    dX_t = kappa * (theta - X_t) dt + sigma * sqrt(X_t) dW_t
    """

    def __init__(self, x0: float, kappa: float, theta: float, sigma: float, seed: int = None):
        """
        Initialize SRD process parameters.

        Args:
            x0 (float): Initial value of the process.
            kappa (float): Mean reversion speed.
            theta (float): Long-term mean level.
            sigma (float): Volatility parameter.
            seed (int, optional): Random seed for reproducibility.
        """
        self.x0, self.kappa, self.theta, self.sigma = x0, kappa, theta, sigma
        self.rng = npr.default_rng(seed)

    def simulate_euler(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        """
        Simulate paths using Euler discretization.

        Args:
            T (float): Time horizon.
            n_paths (int): Number of simulated paths.
            n_steps (int): Number of time steps.

        Returns:
            np.ndarray: Simulated paths of shape (n_steps+1, n_paths).
        """
        dt = T / n_steps
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = self.x0

        for t in range(1, n_steps + 1):
            xt = np.maximum(paths[t - 1], 0)
            dW = self.rng.standard_normal(n_paths)
            paths[t] = xt + self.kappa * (self.theta - xt) * dt \
                       + self.sigma * np.sqrt(xt) * np.sqrt(dt) * dW

        return np.maximum(paths, 0)

    def simulate_exact(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        """
        Simulate paths using the exact non-central chi-square method.

        Args:
            T (float): Time horizon.
            n_paths (int): Number of simulated paths.
            n_steps (int): Number of time steps.

        Returns:
            np.ndarray: Simulated paths of shape (n_steps+1, n_paths).
        """
        dt = T / n_steps
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = self.x0

        df = 4 * self.theta * self.kappa / self.sigma**2

        for t in range(1, n_steps + 1):
            c = (self.sigma**2 * (1 - np.exp(-self.kappa * dt))) / (4 * self.kappa)
            nc = (4 * self.kappa * np.exp(-self.kappa * dt) * paths[t - 1]) / \
                 (self.sigma**2 * (1 - np.exp(-self.kappa * dt)))
            paths[t] = c * self.rng.noncentral_chisquare(df, nc, n_paths)

        return np.maximum(paths, 0)
