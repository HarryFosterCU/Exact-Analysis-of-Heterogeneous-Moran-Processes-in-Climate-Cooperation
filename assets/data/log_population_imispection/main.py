import pandas as pd
import numpy as np
import sympy as sym
import pathlib
import sys
import uuid
import math

file_path = pathlib.Path(__file__)
root_path = (file_path / "../../../../").resolve()

sys.path.append(str(root_path))
import src.main as main
import src.fitness_functions as fitness_functions
import src.contribution_rules as contribution_rules


r_min = 0.5

df = pd.DataFrame(columns=["UID", "alpha_i", "i", "first_alpha", "N", "r", "epsilon", "beta", "p_C"])
df.to_csv(file_path.parent / "main.csv")
N = 3
while True:
    for M in np.linspace(N, 4*N, 30):
            alphas = main.get_deterministic_contribution_vector(N=N, contribution_rule=contribution_rules.log_contribution_rule, M=M)
            for r in np.linspace(0.5,1.5*N, 30):
                for selection_intensity in np.linspace(0,(1/alphas[-1]) * 0.99,30):
                    for choice_intensity in np.linspace(0,2,30):
                        id = uuid.uuid4()
                        state_space = main.get_state_space(N=N, k=2)

                        transition_matrix = main.generate_transition_matrix(state_space=state_space, fitness_function=fitness_functions.heterogeneous_contribution_pgg_fitness_function, compute_transition_probability=main.compute_imitation_introspection_transition_probability, r=r, contribution_vector=alphas, selection_intensity=selection_intensity,
                        choice_intensity=choice_intensity,
                        number_of_strategies=2)

                        absorption_matrix = main.approximate_absorption_matrix(transition_matrix)
                        
                        data = []
                        for j in range(0,N):
                            approximate_state = np.zeros(N)
                            approximate_state[j] = 1
                            starting_player_contribution = alphas[
                                np.where(state_space[np.where(np.all(state_space == approximate_state, axis = 1))][0] == 1)
                                ][0]
                            p_C = absorption_matrix[np.where(np.all(state_space == approximate_state, axis=1))[0] - 1, -1]
                            for i, alpha in enumerate(alphas):
                                row = [id, alpha, i, starting_player_contribution, N, r, selection_intensity, choice_intensity, p_C]
                                data.append(row)
                        df = pd.DataFrame(data)
                        df.to_csv(file_path.parent / "main.csv", mode='a', header=False)
    N += 1
        
