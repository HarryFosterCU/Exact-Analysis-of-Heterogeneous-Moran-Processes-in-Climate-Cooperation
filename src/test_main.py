import main

def test_generate_state_space():
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
    assert expected_state_space == obtained_state_space
