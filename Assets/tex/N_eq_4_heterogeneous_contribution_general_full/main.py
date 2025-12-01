import numpy as np
import sympy as sym
import pathlib
import sys

file_path = pathlib.Path(__file__)

root_path = (file_path / "../../../../").resolve()

sys.path.append(str(root_path))
import src.main


def heterogeneous_contribution_fitness_function(
    state, omega, r, contribution_vector, **kwargs
):
    """Public goods fitness function where each player contributes H times

    their position in the state."""

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

    return 1 + (omega * payoff_vector)


r = sym.Symbol("r")
omega = sym.Symbol("w")
N = 4
alphas = [sym.Symbol(f"alpha_{i}") _ for i in range(1, N + 1)]
M = sum(alphas)
state_space = src.main.get_state_space(N=N, k=2)

general_heterogeneous_contribution_transition_matrix = (
    src.main.generate_transition_matrix(
        state_space=state_space,
        fitness_function=heterogeneous_contribution_fitness_function,
        r=r,
        omega=omega,
        N=N,
        contribution_vector=alphas,
    )
)

general_heterogeneous_absorption_matrix = src.main.generate_absorption_matrix(
    general_heterogeneous_contribution_transition_matrix, symbolic=True
)

with open(
    file_path.parent / "main.tex",
    "w",
) as f:
    f.write(sym.latex(sym.Matrix(general_heterogeneous_absorption_matrix)))
