import pytest
import numpy as np
import pathlib
import sys

file_path = pathlib.Path(__file__)

root_path = (file_path / "../../").resolve()

sys.path.append(str(root_path))
import src.main as main


def generate_deterministic_cycle_matrix(n):
    return np.roll(np.eye(n), shift=1, axis=1)


def generate_stochastic_2_valency_cycle_matrix(n):
    I = np.eye(n)
    return 0.5 * (np.roll(I, 1, axis=1) + np.roll(I, -1, axis=1))


def generate_full_uniform_matrix(n):
    return np.full((n, n), 1 / n)


@pytest.mark.parametrize("n", range(2, 21))
def test_approximate_steady_state_for_deterministic_cycle_matrix(n, benchmark):
    """Benchmarks approximate_steady_state for the deterministic cycle
    matrix"""

    transition_matrix = generate_deterministic_cycle_matrix(n)
    benchmark(main.approximate_steady_state, transition_matrix)


@pytest.mark.parametrize("n", range(2, 21))
def test_approximate_steady_state_for_stochastic_2_valency_cycle_matrix(n, benchmark):
    """Benchmarks approximate_steady_state for the stochastic 2-valency cycle
    matrix"""

    transition_matrix = generate_stochastic_2_valency_cycle_matrix(n)
    benchmark(main.approximate_steady_state, transition_matrix)


@pytest.mark.parametrize("n", range(2, 21))
def test_approximate_steady_state_for_full_uniform_matrix(n, benchmark):
    """Benchmarks approximate_steady_state for the full uniform
    matrix"""

    transition_matrix = generate_full_uniform_matrix(n)
    benchmark(main.approximate_steady_state, transition_matrix)


def test_approximate_steady_state_for_specific_four_by_four(benchmark):
    """Benchmarks the approximate_steady_state function that uses the left
    eigenvector approximation"""

    transition_matrix = np.array(
        [
            [0.5, 0, 0.2, 0.3],
            [0.1, 0.7, 0.2, 0],
            [0.3, 0.3, 0.3, 0.1],
            [0.3, 0.1, 0.1, 0.5],
        ]
    )

    benchmark(main.approximate_steady_state, transition_matrix)
