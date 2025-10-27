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
    return np.array(list(itertools.product(range(k), repeat=N)))


def compute_transition_probability(source, target, fitness_function, **kwargs):
    """
    Given two states and a fitness function, returns the transition probability

    when moving from the source state to the target state. Must move between

    states with a Hamming distance of 1. Returns 0 if Hamming distance > 1.

    Returns None if Hamming distance = 0. For an absorbing state, this will

    naturally return 0 for all off-diagonal entries, and None on the diagonal.

    This is adressed in the get_transition_matrix function.

    $\frac{\sum_{v_i = u_{i*}}{f(v_i)}}{\sum_{v_i}f(v_i)}$

    Parameters
    ----------
    source: numpy.array, the starting state

    target: numpy.array, what the source transitions to

    fitness_function: func, The fitness function which maps a state to a numpy.array

    where each entry represents the fitness of the given individual

    Returns
    ---------
    Float: the transition pobability
    """
    different_indices = np.where(source != target)
    if len(different_indices[0]) > 1:
        return 0
    if len(different_indices[0]) == 0:
        return None
    fitness = fitness_function(source, **kwargs)
    denominator = fitness.sum() * len(source)
    numerator = fitness[source == target[different_indices]].sum()
    return numerator / denominator


def generate_transition_matrix(state_space, fitness_function, **kwargs):
    """
    Given a state space and a fitness function, returns the transition matrix

    for the heterogeneous Moran process.

    Parameters
    ----------
    state_space: numpy.array, the state space for the transition matrix.

    fitness_function: function, should return a size N numpy.array when passed a state

    Returns
    ----------
    numpy.array: the transition matrix
    """
    N = len(state_space)
    transition_matrix = np.zeros(shape=(N, N))
    for row_index, source in enumerate(state_space):
        for col_index, target in enumerate(state_space):
            if row_index != col_index:
                try:
                    transition_matrix[row_index, col_index] = (
                        compute_transition_probability(
                            source=source,
                            target=target,
                            fitness_function=fitness_function,
                            **kwargs,
                        )
                    )
                except TypeError:
                    transition_matrix = transition_matrix.astype(object)
                    transition_matrix[row_index, col_index] = (
                        compute_transition_probability(
                            source=source,
                            target=target,
                            fitness_function=fitness_function,
                            **kwargs,
                        )
                    )
    np.fill_diagonal(transition_matrix, 1 - transition_matrix.sum(axis=1))
    return transition_matrix


def get_absorption_probability(transition_matrix, rounding_places):
    """Given a transition matrix, returns the matrix of absorption probabilities.

    By default rounded to 5 d.p.

    Parameters
    -------------
    transition_matrix: np.array, an MxM matrix with rows summing to exactly 1

    rounding_places: rounds transition probability to this many places

    Returns
    ---------
    numpy.array: a matrix with the absorption probabilities for the given transition matrix.
    """

    return np.round(
        (np.linalg.matrix_power(transition_matrix, 100)), decimals=rounding_places
    )
