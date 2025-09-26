"""
Code for the general heterogeneous Moran process

This corresponds to the model described in `main.tex`

TODO: Add mathematical details here.
"""
import itertools

def get_state_space(N, k):
    """
    Return state space for... TODO

    Parameters:
    -----------
    TODO

    Returns:
    --------
    TODO
    """
    return list(itertools.product(range(k), repeat=N))
