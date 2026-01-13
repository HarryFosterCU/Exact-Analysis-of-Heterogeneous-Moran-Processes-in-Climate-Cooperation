import pytest
import numpy as np
import main


def test_approximate_steady_state_by_eigenvector_four_by_four(benchmark):
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
