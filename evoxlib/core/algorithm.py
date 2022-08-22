import types
import chex
from .module import *


class Algorithm(Module):
    """Base class for all algorithms

    The ``ask`` and ``tell`` methods are automatically wrapped with
    ``use_state``
    """

    def ask(self, state: dict):
        """Ask the algorithm

        Ask the algorithm for points to explore

        Parameters
        ----------
        state : dict
            The state of this algorithm.

        Returns
        -------
        dict : dict
            The new state of the algorithm.
        """
        pass

    def tell(self, state: dict, X: chex.Array, F: chex.Array):
        """Tell the algorithm more information

        Tell the algorithm about the points it chose and their corresponding fitness

        Parameters
        ----------
        state : dict
            The state of this algorithm
        X : ndarray
            The points given by ``ask``
        F : ndarray
            The fitness

        Returns
        -------
        dict
            The new state of the algorithm
        """
        pass