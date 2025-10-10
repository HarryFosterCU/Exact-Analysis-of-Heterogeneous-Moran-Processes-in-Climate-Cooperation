import main
import numpy as np
import sympy as sym


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


def test_compute_transition_probability_for_trivial_fitness_function():
    """
    Tests whether the compute_transition_probability

    works properly for a standard fitness function. Given two states

    (source and target, both numpy.arrays) and a trivial

    fitness function (returning 1 for all entries within the state),

    test that compute_transition_probability returns the

    correct value. Here we see (0,1,0) -> (1,1,0) with a correct

    value of 1/9, and then we see a transition with Hamming distance

    2, correct value 0, and then a transition with Hamming distance

    0, correct value None."""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    assert (
        main.compute_transition_probability(
            source=source, target=target, fitness_function=trivial_fitness_function
        )
        == 1 / 9
    )
    source = np.array((0, 1, 0))
    target = np.array((1, 1, 1))
    assert (
        main.compute_transition_probability(
            source=source, target=target, fitness_function=trivial_fitness_function
        )
        == 0
    )
    source = np.array((0, 0, 0))
    target = np.array((0, 0, 0))
    assert (
        main.compute_transition_probability(
            source=source, target=target, fitness_function=trivial_fitness_function
        )
        is None
    )


def test_compute_transition_probability_for_specific_fitness_function():
    """
    Tests to see that the compute_transition_probability

    function works correctly when the fitness function takes into account

    all entries within the state. Given two states (source and target, both numpy.arrays)

    and a specific fitness function (which returns the number of entries

    in the state sharing a type with a given entry (including itself)),

    test that compute_transition_probability returns the

    correct value. Here we see (0,1,0) -> (1,1,0) with a correct

    value of 1/15, and then we see a transition with Hamming distance

    2, correct value 0, and then a transition with Hamming distance

    0, correct value None.

    An example for the fitness function can be seen as in the state

    f((0,0,1)) = (2, 2, 1)"""

    def fitness_function(state):
        return np.array([np.count_nonzero(state == _) for _ in state])

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    assert (
        main.compute_transition_probability(
            source=source, target=target, fitness_function=fitness_function
        )
        == 1 / 15
    )
    source = np.array((0, 1, 1))
    target = np.array((0, 0, 0))
    assert (
        main.compute_transition_probability(
            source=source, target=target, fitness_function=fitness_function
        )
        == 0
    )
    source = np.array((1, 1, 0))
    target = np.array((1, 1, 0))
    assert (
        main.compute_transition_probability(
            source=source, target=target, fitness_function=fitness_function
        )
        is None
    )


def test_compute_transition_probability_for_ordered_fitness_function():
    """
    Tests to see that the compute_transition_probability

    function works correctly when the fitness function takes into account

    the position of entries within the state, both in relation to an entry and

    the position of an entry itself. Given two states (source and target, both numpy.arrays)

    and a specific fitness function (which for a given entry in position i

    (indexed from 0) will return the number of prior (self-included) entries

    with the same value as the entry + (i % 2)), tests that

    compute_transition_probability returns the correct value. Here we see (0,1,0) -> (1,1,0)

    with an expected value of 2/15, and then we see a transition with Hamming

    distance 2, correct value 0, and then a transition with Hamming distance

    0, correct value None.

    An example for the fitness function can be seen as in the state

    f((0,0,1)) = (1, 3, 1)"""

    def ordered_fitness_function(state):
        fitness = np.array([0 for _ in state])
        zero_encountered = 0
        one_encountered = 0
        for position, value in enumerate(state):
            if value == 0:
                zero_encountered += 1
                fitness[position] = zero_encountered + (position % 2)
            else:
                one_encountered += 1
                fitness[position] = one_encountered + (position % 2)
        return fitness

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    assert (
        main.compute_transition_probability(
            source=source, target=target, fitness_function=ordered_fitness_function
        )
        == 2 / 15
    )
    source = np.array((0, 1, 1))
    target = np.array((0, 0, 0))
    assert (
        main.compute_transition_probability(
            source=source, target=target, fitness_function=ordered_fitness_function
        )
        == 0
    )
    source = np.array((1, 1, 0))
    target = np.array((1, 1, 0))
    assert (
        main.compute_transition_probability(
            source=source, target=target, fitness_function=ordered_fitness_function
        )
        is None
    )


def test_compute_transition_probability_for_symbolic_fitness_function():
    """
    Tests for whether compute_transition_prbability returns the correct

    value for a fitness function which works symbolically.

    Given two states (source and target, both numpy.arrays) and a

    symbolic fitness function (i.e, replacing 1 with x and 0 with y, via

    sympy), tests that compute_transition_probability returns the correct

    value. tests (0,1,0) -> (1,1,0), with correct value

    x / ((3 * x) + (6 * y)), then transitions with Hamming distances

    2 and 0, with correct values 0 and None respectively."""

    def symbolic_fitness_function(state):
        return np.array(
            [
                sym.Symbol("x") if individual == 1 else sym.Symbol("y")
                for individual in state
            ]
        )

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    x = sym.symbols("x")
    y = sym.symbols("y")
    assert main.compute_transition_probability(
        source=source, target=target, fitness_function=symbolic_fitness_function
    ) == x / ((3 * x) + (6 * y))
    source = np.array((0, 1, 1))
    target = np.array((0, 0, 0))
    assert (
        main.compute_transition_probability(
            source=source, target=target, fitness_function=symbolic_fitness_function
        )
        == 0
    )
    source = np.array((1, 1, 0))
    target = np.array((1, 1, 0))
    assert (
        main.compute_transition_probability(
            source=source, target=target, fitness_function=symbolic_fitness_function
        )
        is None
    )
    source = np.array((0, 1))
    target = np.array((0, 0))
    assert main.compute_transition_probability(
        source=source, target=target, fitness_function=symbolic_fitness_function
    ) == y / (2 * x + 2 * y)

    source = np.array((0, 1))
    target1 = np.array((0, 0))
    target2 = np.array((1, 1))
    assert 1 - main.compute_transition_probability(
        source=source, target=target1, fitness_function=symbolic_fitness_function
    ) - main.compute_transition_probability(
        source=source, target=target2, fitness_function=symbolic_fitness_function
    ) == (
        1 - (y / (2 * x + 2 * y)) - x / (2 * x + 2 * y)
    )


def test_generate_transition_matrix_for_trivial_fitness_function():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a trivial fitness function an a state space N = 3, K = 2.
    """

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    state_space = np.array(
        [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (0, 0, 0),
            (1, 1, 1),
        ]
    )
    expected_transition_matrix = np.array(
        [
            [5 / 9, 0, 0, 1 / 9, 1 / 9, 0, 2 / 9, 0],
            [0, 5 / 9, 0, 1 / 9, 0, 1 / 9, 2 / 9, 0],
            [0, 0, 5 / 9, 0, 1 / 9, 1 / 9, 2 / 9, 0],
            [1 / 9, 1 / 9, 0, 5 / 9, 0, 0, 0, 2 / 9],
            [1 / 9, 0, 1 / 9, 0, 5 / 9, 0, 0, 2 / 9],
            [0, 1 / 9, 1 / 9, 0, 0, 5 / 9, 0, 2 / 9],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    assert np.array_equal(
        main.generate_transition_matrix(
            state_space=state_space, fitness_function=trivial_fitness_function
        ),
        expected_transition_matrix,
    )


def test_generate_transition_matrix_for_ordered_fitness_function():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a fitness function based on order (see test_compute_transition_matrix_for_ordered_fitness_function

    for a description of the fitness function) and a state space N = 3, K = 2.
    """

    def ordered_fitness_function(state):
        fitness = np.array([0 for _ in state])
        zero_encountered = 0
        one_encountered = 0
        for position, value in enumerate(state):
            if value == 0:
                zero_encountered += 1
                fitness[position] = zero_encountered + (position % 2)
            else:
                one_encountered += 1
                fitness[position] = one_encountered + (position % 2)
        return fitness

    state_space = np.array(
        [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (0, 0, 0),
            (1, 1, 1),
        ]
    )
    expected_transition_matrix = np.array(
        [
            [3 / 5, 0, 0, 1 / 15, 1 / 15, 0, 4 / 15, 0],
            [0, 8 / 15, 0, 2 / 15, 0, 2 / 15, 1 / 5, 0],
            [0, 0, 9 / 15, 0, 1 / 15, 1 / 15, 4 / 15, 0],
            [1 / 15, 1 / 15, 0, 9 / 15, 0, 0, 0, 4 / 15],
            [2 / 15, 0, 2 / 15, 0, 8 / 15, 0, 0, 1 / 5],
            [0, 1 / 15, 1 / 15, 0, 0, 3 / 5, 0, 4 / 15],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    np.testing.assert_allclose(
        main.generate_transition_matrix(
            state_space=state_space, fitness_function=ordered_fitness_function
        ),
        expected_transition_matrix,
    )


def test_generate_transition_matrix_for_different_state_space():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a state space N = 2, K = 3.
    """

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    state_space = np.array(
        [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2)]
    )
    expected_transition_matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1 / 4, 1 / 2, 0, 1 / 4, 0, 0, 0, 0, 0],
            [1 / 4, 0, 1 / 2, 1 / 4, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1 / 4, 0, 0, 0, 1 / 2, 0, 0, 0, 1 / 4],
            [1 / 4, 0, 0, 0, 0, 1 / 2, 0, 0, 1 / 4],
            [0, 0, 0, 1 / 4, 0, 0, 1 / 2, 0, 1 / 4],
            [0, 0, 0, 1 / 4, 0, 0, 0, 1 / 2, 1 / 4],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    np.testing.assert_allclose(
        main.generate_transition_matrix(
            state_space=state_space, fitness_function=trivial_fitness_function
        ),
        expected_transition_matrix,
    )


def test_generate_transition_matrix_for_symbolic_fitness_function():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a symbolic fitness function function based on (see test_compute_transition_matrix_for_symbolic_fitness_function

    for a description of the fitness function) and a smaller state space N = 2, K = 2.
    """

    def symbolic_fitness_function(state):
        return np.array(
            [
                sym.Symbol("x") if individual == 1 else sym.Symbol("y")
                for individual in state
            ]
        )

    state_space = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])

    x = sym.Symbol("x")
    y = sym.Symbol("y")

    expected_transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [
                y / (2 * x + 2 * y),
                (1 - (y / (2 * x + 2 * y)) - x / (2 * x + 2 * y)),
                0,
                x / (2 * x + 2 * y),
            ],
            [
                y / (2 * x + 2 * y),
                0,
                (1 - (y / (2 * x + 2 * y)) - x / (2 * x + 2 * y)),
                x / (2 * x + 2 * y),
            ],
            [0, 0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(
            main.generate_transition_matrix(
                state_space=state_space, fitness_function=symbolic_fitness_function
            ),
            expected_transition_matrix
    )
