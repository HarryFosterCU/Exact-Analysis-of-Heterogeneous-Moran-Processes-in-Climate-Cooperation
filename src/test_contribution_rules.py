import contribution_rules as cr
import numpy as np
import sympy as sym


def test_dirichlet_linear_alpha_rule_for_N_eq_3():
    """Tests that the diriclet_linear_alpha_rule function correctly returns the
    numpy.array [1,2,3] for a population with 3 individuals."""

    N = 3

    expected_alphas = np.array([1,2,3])

    obtained_alphas = cr.dirichlet_linear_alpha_rule(N)

    np.testing.assert_array_equal(expected_alphas, obtained_alphas)


def test_dirichlet_binomial_alpha_rule_for_N_eq_5_n_eq_3():
    """Tests that the diriclet_binomial_alpha_rule function correctly returns
    the numpy.array [1,1,1,3,3] for a population with 5 individuals, 2
    contributing high and 3 contributing low, with a difference of 2 between them."""

    N = 5
    n = 3
    low_alpha = 1
    high_alpha = 3

    expected_alphas = np.array([1,1,1,3,3])

    obtained_alphas = cr.dirichlet_bonmial_alpha_rule(N=N, n=n, low_alpha=low_alpha, high_alpha=high_alpha)

    np.testing.assert_array_equal(expected_alphas, obtained_alphas)


def test_dirichlet_log_alpha_rule_for_N_eq_3():
    """Tests that the dirichlet_log_alpha_rule correctly returns the
    numpy.array (log(1) + 1, log(2) + 1, log(3) + 1) for a population with 3 individuals.
    """

    N = 3

    expected_alphas = np.array([1, 1.693147, 2.098612])

    obtained_alphas = cr.dirichlet_log_alpha_rule(N=N)

    np.testing.assert_array_almost_equal(expected_alphas, obtained_alphas)

def test_log_contribution_rule_for_player_2_N_eq_3_M_eq_12_contributing():
    """
    Tests that the log_contribution_rule function correctly calculates the
    contribution of player 2 of 3 when M = 12 and when they contribute"""

    N = 3
    M = 12
    index = 1
    action = 1

    expected_contribution = 4.095894024

    obtained_contribution = cr.log_contribution_rule(index=index, action=action, M=M, N=N)

    np.testing.assert_almost_equal(expected_contribution,obtained_contribution)


def test_linear_contribution_rule_for_N_eq_3_M_eq_12_contributing():
    """
    Tests that the linear_contribution_rule function correctly calculates
    the contribution of player 2 of 3 when M=12 and when they contribute
    """
    
    N = 3
    M = 12
    index = 1
    action = 1

    expected_contribution = 4

    obtained_contribution = cr.linear_contribution_rule(index=index, action=action, M=M, N=N)

    assert expected_contribution==obtained_contribution

def test_binomial_contribution_rule_for_N_eq_5_n_eq_3():
    """
    Tests that the binomial_contribution_rule function correctly calculates the
    contribution of two players, player 2 and player 4, when N=5 and n=3, and
    when they contribute. We take alpha_h = 3 and M=9
    """

    N = 5
    M = 9
    n=3
    index_1 = 1
    index_2 = 3
    action_1 = 1
    action_2 = 1
    alpha_h = 3

    expected_contribution_1 = 1
    expected_contribution_2 = 3

    obtained_contribution_1 = cr.binomial_contribution_rule(index=index_1, action=action_1, M=M, N=N, alpha_h=alpha_h, n=3)

    obtained_contribution_2 = cr.binomial_contribution_rule(index=index_2, action=action_2, M=M, N=N, alpha_h=alpha_h, n=3)

    assert expected_contribution_1==obtained_contribution_1
    assert expected_contribution_2==obtained_contribution_2