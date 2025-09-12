import numpy as np
import math
from typing import Tuple

class BinomialTree:
    """
    Binomial tree simulator for asset price evolution.

    S_t evolves according to:
    S_t+1 = S_t * u  (up move)
    S_t+1 = S_t * d  (down move)
    """

    def __init__(self, S0: float, r: float, sigma: float):
        """
        Initialize the Binomial Tree model.

        Args:
            S0 (float): Initial stock price.
            r (float): Risk-free rate.
            sigma (float): Volatility of the asset.
        """
        self.S0, self.r, self.sigma = S0, r, sigma

    def simulate_tree(self, T: float, n_steps: int) -> np.ndarray:
        """
        Build the binomial price tree.

        Args:
            T (float): Time horizon.
            n_steps (int): Number of steps in the tree.

        Returns:
            np.ndarray: Price tree matrix of shape (n_steps+1, n_steps+1),
                        with zeros for unused cells.
        """
        dt = T / n_steps
        u = math.exp(self.sigma * math.sqrt(dt))
        d = 1 / u
        tree = np.zeros((n_steps + 1, n_steps + 1))
        tree[0, 0] = self.S0

        for t in range(1, n_steps + 1):
            for i in range(t + 1):
                if i == 0:
                    tree[i, t] = tree[i, t - 1] * u
                else:
                    tree[i, t] = tree[i - 1, t - 1] * d

        return tree