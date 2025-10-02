"""
Code for the general heterogeneous Moran process

This corresponds to the model described in `main.tex`

Assume we have N ordered individuals of k types: $A_1$, ... $A_K$

We have a state v \in S = [$v_1$, .... $v_n$] as the set of types of individuals in the population, so that $v_i$ is the type of individual i. |S| = $k^N$

There is also a fitness function f: S -> $R^N$, giving us an array of the fitness of individuals
"""

import itertools
import numpy as np


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

def fitness(S1, S2):
    """
    TODO
    """
    return 1


def get_where_different(S1, S2):
    """
    Given a pair of states, identify where they are different, and return the position of the difference
    
    Parameters:
    --------------
    S1, S2: Arrays, states from the state space
    
    Returns:
    Position where arrays differ"""

    return np.where(np.array(S1) != np.array(S2))

def get_transition_prob(S1, S2):
    """
    Given two states, return the probability of transitioning from one state to the next

    Parameters:
    -----------
    S1, S2: Arrays, states from the state space
    """
    Diff = get_where_different(S1, S2)[0]
    if len(Diff) <= 1:
        return fitness(S1, S2)
    else:
        return 0
    
def gen_transition_matrix(S):
    """
    Returns the transition matrix given a state space
    
    Parameters:
    -----------

    N: integer, number of individuals

    k: integer, number of types

    S: array, state space

    Returns:
    Transition matrix for the given state space
    """


    T_Mat = np.zeros((len(S), len(S)))

    for x in range(len(S)):
        for y in range(len(S)):

            T_Mat[x,y] = get_transition_prob(S[x], S[y])
    

    return(T_Mat)