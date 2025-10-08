"""
Code for the general heterogeneous Moran process

This corresponds to the model described in `main.tex`

Assume we have N ordered individuals of k types: $A_1$, ... $A_K$

We have a state v \in S = [$v_1$, .... $v_n$] as the set of types of individuals in the population, so that $v_i$ is the type of individual i. |S| = $k^N$

There is also a fitness function f: S -> $R^N$, giving us an array of the fitness of individuals
"""

import itertools
import numpy as np
import sympy as sym


def get_state_space(N, k):
    """
    Return state space for a given N and K

    Parameters:
    -----------
    N: integer, number of individuals

    k: integer, number of possible types

    Returns:
    --------
    Array of possible states within the system
    """
    return list(itertools.product(range(k), repeat=N))


def compute_transition_probability(source, target, fitness_function):
    """
    Given two states and a fitness function, returns the transition probability.

    Parameters
    ----------
    source: numpy array: the starting state

    target: numpy array: what the source transitions to

    fitness_function: function: returns the fitness of a given state
    """
    different_indices = np.where(source != target)
    if len(different_indices[0]) > 1:
        return 0
    if len(different_indices[0]) == 0:
        return None
    fitness = fitness_function(source)
    denominator = fitness.sum() * len(source)
    numerator = fitness[source == target[different_indices]].sum()
    return numerator / denominator


def generate_transition_matrix(state_space, fitness_function):
    """"""
    N = len(state_space)
    transition_matrix = np.zeros(shape=(N, N))
    for row_index, source in enumerate(state_space):
        for col_index, target in enumerate(state_space):
            if row_index != col_index:
                transition_matrix[row_index, col_index] = (
                    compute_transition_probability(
                        source=source, target=target, fitness_function=fitness_function
                    )
                )
    for diag in range(N):
        transition_matrix[diag, diag] = 1 - np.sum(transition_matrix[diag])
    return transition_matrix
