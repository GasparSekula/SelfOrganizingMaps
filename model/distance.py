import numpy as np

class Distance:
    """Abstract base class for distance functions."""
    
    def __call__(self, distance: np.ndarray) -> np.ndarray:
        """
        Calculate the distance transformation.

        Args:
            distance (np.ndarray): The input distance array.

        Returns:
            np.ndarray: The transformed distance array.
        """
        raise NotImplementedError("Distance function must be implemented in subclasses.")


class GaussianDistance(Distance):
    """Gaussian distance function."""
    
    def __init__(self, sigma: float):
        """
        Initialize the GaussianDistance.

        Args:
            sigma (float): The standard deviation for the Gaussian function.
        """
        self.sigma = sigma
    
    def __call__(self, distance: np.ndarray) -> np.ndarray:
        """
        Apply the Gaussian distance transformation.

        Args:
            distance (np.ndarray): The input distance array.

        Returns:
            np.ndarray: The transformed distance array.
        """
        return np.exp(-distance**2 / (2 * self.sigma**2))


class MexicanHatDistance(Distance):
    """Mexican Hat (Ricker wavelet) distance function."""
    
    def __init__(self, sigma: float):
        """
        Initialize the MexicanHatDistance.

        Args:
            sigma (float): The standard deviation for the function.
        """
        self.sigma = sigma
    
    def __call__(self, distance: np.ndarray) -> np.ndarray:
        """
        Apply the Mexican Hat distance transformation.

        Args:
            distance (np.ndarray): The input distance array.

        Returns:
            np.ndarray: The transformed distance array.
        """
        return (1 - (distance ** 2) / (self.sigma ** 2)) * np.exp(-distance**2 / (2 * self.sigma**2))
