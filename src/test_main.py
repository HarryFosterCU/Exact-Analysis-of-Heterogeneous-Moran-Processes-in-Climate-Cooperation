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
    """"""
    def trivial_fitness_function(state):
        return np.array([1 for _ in state])
    source = np.array((0,1,0))
    target = np.array((1,1,0))
    assert main.compute_transition_probability(source=source, target=target, fitness_function=trivial_fitness_function) == 1/9
    source = np.array((0,1,0))
    target = np.array((1,1,1))
    assert main.compute_transition_probability(source=source, target=target, fitness_function=trivial_fitness_function) == 0
    source = np.array((0,0,0))
    target = np.array((0,0,0))
    assert main.compute_transition_probability(source=source, target=target, fitness_function=trivial_fitness_function) is None

def test_compute_transition_probability_for_specific_fitness_function():
    """"""
    def fitness_function(state):
        return np.array([np.count_nonzero(state==_) for _ in state])
    source = np.array((0,1,0))
    target = np.array((1,1,0))
    assert main.compute_transition_probability(source=source, target=target, fitness_function=fitness_function) == 1/15
    source = np.array((0,1,1))
    target = np.array((0,0,0))
    assert main.compute_transition_probability(source=source, target=target, fitness_function=fitness_function) == 0
    source = np.array((1,1,0))
    target = np.array((1,1,0))
    assert main.compute_transition_probability(source=source, target=target, fitness_function=fitness_function) is None

def test_compute_transition_probability_for_ordered_fitness_function():
    """"""
    def ordered_fitness_function(state):
        fitness = np.array([0 for _ in state])
        zero_encountered = 0
        one_encountered = 0
        for position, value in enumerate(state):
            if value == 0:
                zero_encountered += 1
                fitness[position] = (zero_encountered + (position % 2))
            else:
                one_encountered += 1
                fitness[position] = (one_encountered + (position % 2))
        return(fitness)
    source = np.array((0,1,0))
    target = np.array((1,1,0))
    assert main.compute_transition_probability(source=source, target=target, fitness_function=ordered_fitness_function) == 2/15
    source = np.array((0,1,1))
    target = np.array((0,0,0))
    assert main.compute_transition_probability(source=source, target=target, fitness_function=ordered_fitness_function) == 0
    source = np.array((1,1,0))
    target = np.array((1,1,0))
    assert main.compute_transition_probability(source=source, target=target, fitness_function=ordered_fitness_function) is None

def test_compute_transition_probability_for_symbolic_fitness_function():
    """"""
    def symbolic_return(input):
        if input == 1:
            return sym.symbols('x')
        return sym.symbols('y')
    def symbolic_fitness_function(state):
        return np.array([symbolic_return(_) for _ in state])
    source = np.array((0,1,0))
    target = np.array((1,1,0))
    x = sym.symbols('x')
    y = sym.symbols('y')
    assert main.compute_transition_probability(source=source, target=target, fitness_function=symbolic_fitness_function) == x / ((3 * x) + (6 * y))
    source = np.array((0,1,1))
    target = np.array((0,0,0))
    assert main.compute_transition_probability(source=source, target=target, fitness_function=symbolic_fitness_function) == 0
    source = np.array((1,1,0))
    target = np.array((1,1,0))
    assert main.compute_transition_probability(source=source, target=target, fitness_function=symbolic_fitness_function) is None


def test_generate_transition_matrix_for_trivial_fitness_function():
    """"""
    def trivial_fitness_function(state):
        return np.array([1 for _ in state])
    state_space = np.array([
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 0, 0),
        (1, 1, 1),
    ])
    expected_transition_matrix = np.array([
        [5/9, 0, 0, 1/9, 1/9, 0, 2/9, 0], 
        [0, 5/9, 0, 1/9, 0, 1/9, 2/9, 0],
        [0, 0, 5/9, 0, 1/9, 1/9, 2/9, 0],
        [1/9, 1/9, 0, 5/9, 0, 0, 0, 2/9],
        [1/9, 0, 1/9, 0 ,5/9, 0, 0, 2/9],
        [0, 1/9, 1/9, 0, 0, 5/9, 0, 2/9],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
        ])
    assert np.array_equal(main.generate_transition_matrix(state_space=state_space, fitness_function=trivial_fitness_function), expected_transition_matrix)
    

def test_generate_transition_matrix_for_ordered_fitness_function():
    """TODO"""
    def ordered_fitness_function(state):  #From left to right, counts how many of the same type have been read, and adds the position % 2 to give fitness
        fitness = np.array([0 for _ in state])
        zero_encountered = 0
        one_encountered = 0
        for position, value in enumerate(state):
            if value == 0:
                zero_encountered += 1
                fitness[position] = (zero_encountered + (position % 2))
            else:
                one_encountered += 1
                fitness[position] = (one_encountered + (position % 2))
        return(fitness)
    state_space = np.array([
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 0, 0),
        (1, 1, 1),
    ])
    expected_transition_matrix = np.array([
        [3/5, 0, 0, 1/15, 1/15, 0, 4/15, 0], 
        [0, 8/15, 0, 2/15, 0, 2/15, 1/5, 0],
        [0, 0, 9/15, 0, 1/15, 1/15, 4/15, 0],
        [1/15, 1/15, 0, 9/15, 0, 0, 0, 4/15],
        [2/15, 0, 2/15, 0 ,8/15, 0, 0, 1/5],
        [0, 1/15, 1/15, 0, 0, 3/5, 0, 4/15],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
        ])
    np.testing.assert_allclose(main.generate_transition_matrix(state_space=state_space, fitness_function=ordered_fitness_function), expected_transition_matrix)
    
    
    