import numpy as np
import sympy as sym


def heterogeneous_contribution_pgg_fitness_function(
    state, r, contribution_vector, **kwargs
):
    """Public goods fitness function where players contribute a different
    amount.

    Parameters
    -----------

    state: numpy.array, the ordered set of actions each player takes; 1 for
    contributing, 0 for free-riding.

    r: float, the parameter which the public goods is multiplied by

    contribution_vector: numpy.array, the value which each player contributes

    Returns
    -------

    numpy.array: an ordered vector of each player's fitness."""

    total_goods = (
        r
        * sum(
            action * contribution
            for action, contribution in zip(state, contribution_vector)
        )
        / len(state)
    )

    payoff_vector = np.array(
        [
            total_goods - (action * contribution)
            for action, contribution in zip(state, contribution_vector)
        ]
    )

    return payoff_vector


def homogeneous_pgg_fitness_function(state, alpha, r, **kwargs):
    """
    Public goods fitness function where all players contribute the same amount.
    They therefore have a return of 1 + (selection_intensity * payoff), This
    is the selection intensity $\epsilon$, which determines the effect of
    payoff on a player's fitness.

    Parameters
    -----------
    state: numpy.array, the ordered set of actions each player takes

    alpha: float, each player's contribution

    r: float, the parameter which the public goods is multiplied by

    epsilon: float, the selection intensity determining the effect of payoff on
    a player's fitness. Must satisfy $0 < \epsilon < \frac{N}{(N-r)\alpha}$ if r<N

    Returns
    -------
    numpy.array: an ordered array of each player's fitness"""
    homogeneous_contribution_vector = np.array([alpha for _ in enumerate(state)])
    return heterogeneous_contribution_pgg_fitness_function(
        state=state,
        r=r,
        contribution_vector=homogeneous_contribution_vector,
        **kwargs,
    )


def general_four_state_fitness_function(state, **kwargs):
    """
    Returns a general fitness function for each player in a general 2 player population,
    according to the rule
    $f_i(x)$ is the fitness of player i in state x, indexed from 1.

    In this case, the states correspond to:
    $a=(0,0)$, $b=(0,1)$, $c=(1,0)$, $d=(1,1)$
    This is the same state space as we have in the Population Dynamics section
    of main.tex.

    Parameters
    -----------
    state: numpy.array, the ordered set of actions each player takes

    Returns
    -------
    numpy.array: an ordered array of each player's fitness"""

    f = sym.Function("f")
    if (state == np.array([0, 0])).all():
        state_symbol = sym.Symbol("a")
    if (state == np.array([0, 1])).all():
        state_symbol = sym.Symbol("b")
    if (state == np.array([1, 0])).all():
        state_symbol = sym.Symbol("c")
    if (state == np.array([1, 1])).all():
        state_symbol = sym.Symbol("d")

    return np.array(
        [sym.Function(f"f_{i+1}")(state_symbol) for i, j in enumerate(state)]
    )
