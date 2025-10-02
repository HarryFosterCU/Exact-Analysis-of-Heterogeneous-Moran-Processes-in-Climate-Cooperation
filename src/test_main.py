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

    Expected_Transition_Matrix = np.array([[1, 0, 0, 1, 1, 0, 1, 0],
[0, 1, 0, 1, 0, 1, 1, 0],
[0, 0, 1, 0, 1, 1, 1, 0],
[1, 1, 0, 1, 0, 0, 0, 1],
[1, 0, 1, 0, 1, 0, 0, 1],
[0, 1, 1, 0, 0, 1, 0, 1],
[1, 1, 1, 0, 0, 0, 1, 0],
[0, 0, 0, 1, 1, 1, 0, 1]])

    Obtained_Transition_Matrix = main.gen_transition_matrix(S)
    assert (Obtained_Transition_Matrix == Expected_Transition_Matrix).all()


test_get_transition_matrix_for_given_state_space()