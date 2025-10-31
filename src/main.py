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
    unsorted_state_space = np.array(list(itertools.product(range(k), repeat=N)))

    sorted_state_space = unsorted_state_space[
        np.argsort(unsorted_state_space.sum(axis=1))
    ]

    return sorted_state_space


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


def get_absorbing_state_index(state_space):
    """Given a state space, returns the indexes of the absorbing states
    (i.e, states with only one value repeated).

    Parameters
    -------------
    state_space: numpy.array, an array of states

    Returns
    --------------
    numpy.array of index values for the absorbing states"""

    absorbing_index = np.where(np.all(state_space == state_space[:, [0]], axis=1))[0]

    return absorbing_index if len(absorbing_index) >= 1 else None


def get_absorbing_states(state_space):
    """Given a state space, returns the absorbing states

    Parameters
    -----------
    state_space: numpy.array, a state space

    Returns
    ---------
    numpy.array, a list of absorbing states, in order"""

    index_array = get_absorbing_state_index(state_space=state_space)

    return np.array([state_space[index] for index in index_array])


def get_absorption_probabilities(transition_matrix, state_space):
    """Given a transition matrix and a corresponding state space

    generate the absorption probabilities.

    Parameters
    -------------
    state_space: numpy.array, a state space

    transition matrix: numpy.array, a matrix of transition probabilities corresponding to the state space

    Returns
    -------------
    dictionary of values: tuple([starting state]): [[absorbing state 1, absorption probability 1], [absorbing state 2, absorption probability 2]]
    """

    # get absorption indexes
    # get absorption probabilities per index per starting state
    # formulate dictionary

    absorption_index = get_absorbing_state_index(state_space=state_space)
    absorbing_states = get_absorbing_states(state_space=state_space)

    absorbing_transition_matrix = np.linalg.matrix_power(transition_matrix, 50)
    # this method of getting absorption probabilities will change, but we need to set up benchmarks first

    absorbing_collums = np.array(
        [absorbing_transition_matrix[:, index] for index in absorption_index]
    )
    # returns a numpy.array of the absorption probabilities for each absorbing state

    combined_values = np.array(
        [
            np.ravel(np.column_stack((absorption_index, absorbing_collums[:, k])))
            for k, y in enumerate(absorbing_collums.transpose())
        ]
    )

    return {tuple(state): combined_values[x] for x, state in enumerate(state_space)}
