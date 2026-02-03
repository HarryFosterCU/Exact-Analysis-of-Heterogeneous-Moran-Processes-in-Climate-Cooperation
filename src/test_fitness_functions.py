import main
import sympy as sym
import numpy as np
import pytest
import fitness_functions as ff


def test_heterogeneous_contribution_pgg_fitness_function_for_homogeneous_contribution():
    """
    Tests that heterogeneous_contribution_pgg_fitness_function correctly
    handles a homogeneous contribution. Simultaneously shows that the function
    takes into account each player's choice of contribution"""

    homogeneous_contributions = np.array([2, 2, 2])
    state = np.array([1, 0, 1])
    r = 1.8

    actual_return = ff.heterogeneous_contribution_pgg_fitness_function(
        state=state, contribution_vector=homogeneous_contributions, r=r
    )

    expected_return = np.array([0.4, 2.4, 0.4])

    np.testing.assert_allclose(actual_return, expected_return)


def test_heterogeneous_contribution_pgg_fitness_function_for_heterogeneous_numeric_contribution():
    """
    Tests that heterogeneous_contribution_pgg_fitness_function returns the
    correct values for a purely numerical hetereogeneous contribution"""

    heterogeneous_contributions = np.array([1, 3, 4, 8])
    state = np.array([1, 1, 1, 1])
    r = 2.1

    actual_return = ff.heterogeneous_contribution_pgg_fitness_function(
        state=state,
        contribution_vector=heterogeneous_contributions,
        r=r,
    )

    expected_return = np.array([7.4, 5.4, 4.4, 0.4])

    np.testing.assert_allclose(actual_return, expected_return)


def test_heterogeneous_contribution_pgg_fitness_function_for_heterogeneous_symbolic_contribution():
    """
    Tests that heterogeneous_contribution_pgg_fitness_function returns the
    correct values for a purely numerical hetereogeneous contribution"""

    a = sym.Symbol("a")
    b = sym.Symbol("b")

    heterogeneous_contributions = np.array([a, b])
    state = np.array([1, 1])
    r = sym.Symbol("r")

    actual_return = ff.heterogeneous_contribution_pgg_fitness_function(
        state=state,
        contribution_vector=heterogeneous_contributions,
        r=r,
    )

    expected_return_p1 = r * (a + b) / 2 - a
    expected_return_p2 = r * (a + b) / 2 - b

    expected_return = np.array([expected_return_p1, expected_return_p2])

    np.testing.assert_array_equal(actual_return, expected_return)


def test_heterogeneous_contribution_pgg_fitness_function_for_heterogeneous_no_contribution():
    """
    Tests that heterogeneous_contribution_pgg_fitness_function returns the
    correct values when no players contribute"""

    heterogeneous_contributions = np.array([8, 41, 28, 19])
    state = np.array([0, 0, 0, 0])
    r = 3

    actual_return = ff.heterogeneous_contribution_pgg_fitness_function(
        state=state,
        contribution_vector=heterogeneous_contributions,
        r=r,
    )

    expected_return = np.array([0, 0, 0, 0])

    np.testing.assert_allclose(actual_return, expected_return)


def test_homogeneous_pgg_fitness_function_for_numeric_value():
    """
    Tests that homogeneous_pgg_fitness_function returns the correct value for a
    purely numerical system."""

    alpha = 2
    state = np.array([1, 1, 0, 1, 0, 0])
    r = 2

    actual_return = ff.homogeneous_pgg_fitness_function(state=state, alpha=alpha, r=r)

    expected_return = np.array([0, 0, 2, 0, 2, 2])

    np.testing.assert_allclose(actual_return, expected_return)


def test_homogeneous_pgg_fitness_function_for_symbolic_values():
    """
    Tests that homogeneous_pgg_fitness_function returns the correct value for
    purely symbolic values."""

    alpha = sym.Symbol("alpha")
    state = np.array([1, 1, 0])
    r = sym.Symbol("r")

    actual_return = ff.homogeneous_pgg_fitness_function(state=state, alpha=alpha, r=r)

    contributor_payment = (2 * r * alpha / 3) - alpha
    defector_payment = 2 * r * alpha / 3

    expected_return = np.array(
        [contributor_payment, contributor_payment, defector_payment]
    )

    np.testing.assert_array_equal(actual_return, expected_return)


def test_homogeneous_pgg_fitness_function_for_no_contribution():
    """
    Tests that homogeneous_pgg_fitness_function returns the correct value when
    no players contribute."""

    alpha = 2
    state = np.array([0, 0, 0])
    r = 1.8

    actual_return = ff.homogeneous_pgg_fitness_function(state=state, alpha=alpha, r=r)

    expected_return = np.array([0, 0, 0])

    np.testing.assert_array_equal(actual_return, expected_return)


def test_general_four_state_player_fitness_function_for_whole_state_space():
    """Tests that all players are assigned the correct values in all possible
    states of the four-state system where N=2, k=2"""
    four_state_space = main.get_state_space(N=2, k=2)

    f = sym.Function("f")

    expected_return_a = np.array(
        [
            sym.Function(f"f_{1}")(sym.Symbol("a")),
            sym.Function(f"f_{2}")(sym.Symbol("a")),
        ]
    )

    expected_return_b = np.array(
        [
            sym.Function(f"f_{1}")(sym.Symbol("b")),
            sym.Function(f"f_{2}")(sym.Symbol("b")),
        ]
    )

    expected_return_c = np.array(
        [
            sym.Function(f"f_{1}")(sym.Symbol("c")),
            sym.Function(f"f_{2}")(sym.Symbol("c")),
        ]
    )

    expected_return_d = np.array(
        [
            sym.Function(f"f_{1}")(sym.Symbol("d")),
            sym.Function(f"f_{2}")(sym.Symbol("d")),
        ]
    )

    expected_returns = [
        expected_return_a,
        expected_return_b,
        expected_return_c,
        expected_return_d,
    ]

    for i in range(4):
        np.testing.assert_array_equal(
            expected_returns[i],
            ff.general_four_state_fitness_function(four_state_space[i]),
        )
