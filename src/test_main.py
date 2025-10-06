import main
import numpy as np


def test_generate_state_space_for_N_eq_3_and_k_eq_2():
    """
    Given a value of $N$: the number of individuals and a value of $k$: the
    number of types generate $S = [1, ..., k] ^ N$.

    This tests this for N = 3, k = 2.
    """
    k = 2
    N = 3
    expected_state_space = [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 0, 0),
        (1, 1, 1),
    ]
    obtained_state_space = main.get_state_space(N=N, k=k)
    assert sorted(expected_state_space) == sorted(obtained_state_space)


def test_generate_state_space_for_N_eq_3_and_k_eq_1():
    """
    Given a value of $N$: the number of individuals and a value of $k$: the
    number of types generate $S = [1, ..., k] ^ N$.

    This tests this for N = 3, k = 1.
    """
    k = 1
    N = 3
    expected_state_space = [
        (0, 0, 0),
    ]
    obtained_state_space = main.get_state_space(N=N, k=k)
    assert sorted(expected_state_space) == sorted(obtained_state_space)


def test_generate_state_space_for_N_eq_1_and_k_eq_3():
    """
    Given a value of $N$: the number of individuals and a value of $k$: the
    number of types generate $S = [1, ..., k] ^ N$.

    This tests this for N = 1, k = 3.
    """
    k = 3
    N = 1
    expected_state_space = [
        (0,),
        (1,),
        (2,),
    ]
    obtained_state_space = main.get_state_space(N=N, k=k)
    assert sorted(expected_state_space) == sorted(obtained_state_space)


def test_get_transition_matrix_for_given_state_space():
    """
    Given a state space S generate the transition matrix

    This tests this for N = 3, k = 2.
    """

    S = [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 0, 0),
        (1, 1, 1),
    ]

    expected_transition_matrix = np.array(
        [
            [1, 0, 0, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 1],
        ]
    )

    obtained_transition_matrix = main.gen_transition_matrix(S)
    assert (obtained_transition_matrix == expected_transition_matrix).all()


def test_compute_transition_probability_for_trivial_fitness_function():
    """"""
    state_space = [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 0, 0),
        (1, 1, 1),
    ]
    def trivial_fitness_function(state):
        return np.array([1 for _ in state])
    source = np.array((0,1,0))
    target = np.array((1,1,0))
    assert main.compute_transition_probability(source=source, target=target) == 1/64