import numpy as np
import sympy as sym
import pathlib
import sys

file_path = pathlib.Path(__file__)
root_path = (file_path / "../../../../").resolve()

sys.path.append(str(root_path))
import src.main as main
import src.fitness_functions as fitness_functions

r = sym.Symbol("r")
epsilon = sym.Symbol("epsilon")
N = 2
state_space = main.get_state_space(N=N, k=2)
beta = sym.Symbol("beta")

transition_matrix = main.generate_transition_matrix(
    state_space=state_space,
    fitness_function=fitness_functions.general_four_state_fitness_function,
    compute_transition_probability=main.compute_introspection_transition_probability,
    choice_intensity=beta,
    number_of_strategies=2
    )

steady_state = main.calculate_steady_state(transition_matrix)


with open(
    file_path.parent / "main.tex",
    "w",
) as f:
    f.write(sym.latex(sym.Matrix(steady_state)))
