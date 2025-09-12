import numpy as np
from numpy.random import default_rng
from typing import Optional

class HestonSimulation:
    """
    Heston model Monte Carlo simulation.
    
    dS_t = r S_t dt + sqrt(v_t) S_t dW_t^1
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_t^2
    Corr(dW_t^1, dW_t^2) = rho
    """

    def __init__(self, S0: float, v0: float, r: float, kappa: float, theta: float,
                 sigma: float, rho: float, seed: Optional[int] = None):
        """
        Initialize the Heston simulator.

        Args:
            S0 (float): Initial stock price.
            v0 (float): Initial variance.
            r (float): Risk-free rate.
            kappa (float): Mean reversion speed of variance.
            theta (float): Long-term variance level.
            sigma (float): Volatility of volatility.
            rho (float): Correlation between price and variance.
            seed (int, optional): Random seed for reproducibility.
        """
        self.S0, self.v0 = S0, v0
        self.r, self.kappa, self.theta, self.sigma, self.rho = r, kappa, theta, sigma, rho
        self.rng = default_rng(seed)

    def simulate_euler(self, T: float, n_paths: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate asset and variance paths using Euler discretization.

        Args:
            T (float): Time horizon.
            n_paths (int): Number of simulated paths.
            n_steps (int): Number of time steps.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - S (n_steps+1, n_paths): Asset price paths
                - v (n_steps+1, n_paths): Variance paths
        """
        dt = T / n_steps

        # Precompute Cholesky for correlation
        corr_mat = np.array([[1.0, self.rho], [self.rho, 1.0]])
        cho_mat = np.linalg.cholesky(corr_mat)

        S = np.zeros((n_steps + 1, n_paths))
        v = np.zeros((n_steps + 1, n_paths))
        S[0] = self.S0
        v[0] = self.v0

        Z = self.rng.standard_normal((2, n_steps+1, n_paths))
        for t in range(1, n_steps + 1):
            # Correlated random numbers
            dW = cho_mat @ Z[:, t, :]

            v_prev = np.maximum(v[t - 1], 0)
            v[t] = v_prev + self.kappa * (self.theta - v_prev) * dt + \
                   self.sigma * np.sqrt(v_prev * dt) * dW[1]
            v[t] = np.maximum(v[t], 0)  # enforce positivity

            S[t] = S[t - 1] * np.exp(
                (self.r - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * dW[0]
            )

        return S, v