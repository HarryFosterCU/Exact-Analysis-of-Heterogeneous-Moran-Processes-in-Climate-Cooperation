import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

np.set_printoptions(suppress=False)
import matplotlib.patches as mpatches
import matplotlib.lines as mlinesge
import scipy
import math

import uuid
import csv

import sys
import pathlib
import argparse
import os

file_path = pathlib.Path(__file__)
root_path = (file_path / "../../../../").resolve()

sys.path.append(str(root_path))
import src.main as main
import src.contribution_rules as cr

"""Parse arguments given in command line"""

parser = argparse.ArgumentParser(
        description="Model parameters passed via command line"
    )

parser.add_argument("--N", type=int, default=None)
parser.add_argument("--n", type=int, default=None)
parser.add_argument("--M", type=int, default=None)
parser.add_argument("--alpha_h", type=float, default=None)
parser.add_argument("--r", type=float, default=None)
parser.add_argument("--beta", type=float, default=None)
parser.add_argument("--epsilon", type=float, default=None)

parser.add_argument("--lim", type=bool, default=False)
parser.add_argument("--max", type=int, default=None)
parser.add_argument("--inc", type=float, default=None)

args = parser.parse_args()

"""Set defaults for arguments not passed"""

N = args.N if args.N is not None else 2
n = args.n if args.n is not None else 1
M = args.M if args.M is not None else 20
alpha_h = args.alpha_h if args.alpha_h is not None else 0.2
r = args.r if args.r is not None else 1.5
beta = args.beta if args.beta is not None else 0.01
epsilon = args.epsilon if args.epsilon is not None else 0.001
lim = args.lim
max_runs = args.max if args.max is not None else None
inc = args.inc if args.inc is not None else 1.1

"""Check which values are to be incremented"""

parameter_variability = {
    "N": args.N is None,
    "n": args.n is None,
    "M": args.M is None,
    "alpha_h": args.alpha_h is None,
    "r": args.r is None,
    "beta": args.beta is None,
    "epsilon": args.epsilon is None,
}

variable_values = {
    "N": N,
    "M": M,
    "n": n,
    "alpha_h": alpha_h,
    "r": r,
    "beta": beta,
    "epsilon": epsilon,
}

processes = {
    "MP": main.compute_moran_transition_probability,
    "Fermi": main.compute_fermi_transition_probability,
    "II": main.compute_imitation_introspection_transition_probability,
    "Intro":main.compute_introspection_transition_probability
}

"""Define fitness function"""

def heterogeneous_contribution_fitness_function(
    state,
    epsilon,
    r,
    contribution_vector,
    **kwargs
):
    """Public goods fitness function where each player contributes H times

    their position in the state."""

    total_goods = (
        r
        * sum(action * contribution for action, contribution in zip(state, contribution_vector))
        / len(state)
    )

    payoff_vector = np.array([total_goods - (action * contribution) for action, contribution in zip(state, contribution_vector)])

    return 1 + (payoff_vector * epsilon)

"""Check if data file exists"""

file_exists = os.path.isfile(file_path.parent / "data.csv")
with open(file_path.parent / "data.csv", mode="a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(["UID", "N", "M", "i", r"$\alpha_{i}$", "First Contributor", "r", r"$\beta$", r"$\epsilon$", r"$\rho_{C}", r"$p_{C}$","Process"])

if lim:
    runs = 0

if __name__ == "__main__":
    while True:   
        for parameter in parameter_variability.keys():
            if parameter_variability[parameter] == False:
                continue
            for current_process in processes.keys():
                print(variable_values["N"])
                contributions = main.get_deterministic_contribution_vector(cr.binomial_contribution_rule, 
                N=variable_values["N"], 
                n=variable_values["n"],
                M=variable_values["M"], 
                alpha_h = variable_values["alpha_h"])

                transition_matrix = main.generate_transition_matrix(
                    state_space = main.get_state_space(N=variable_values["N"], k=2),
                    fitness_function = heterogeneous_contribution_fitness_function,
                    compute_transition_probability = processes[current_process],
                    contribution_vector = contributions,
                    r = variable_values["r"],
                    epsilon = variable_values["epsilon"],
                    selection_intensity = variable_values["beta"],
                    number_of_strategies = 2,
                )

                if current_process == "Intro":
                    
                    id = uuid.uuid4()

                    result = main.approximate_steady_state(transition_matrix)[-1]

                    with open(file_path.parent / "data.csv", mode="a", newline="") as f:
                        writer = csv.writer(f)

                        for i in range(variable_values["N"]):
                            writer.writerow([id, variable_values["N"], variable_values["M"], i, contributions[i], None, variable_values["r"], variable_values["beta"], variable_values["epsilon"], None, result, current_process])
                
                else:

                    absorption = main.approximate_absorption_matrix(transition_matrix)


                    with open(file_path.parent / "data.csv", mode="a", newline="") as f:
                        writer = csv.writer(f)

                        for starting_player in range(variable_values["N"]):
                            id = uuid.uuid4()
                            result = absorption[starting_player, -1]
                            for i in range(variable_values["N"]):
                                writer.writerow([id, variable_values["N"], variable_values["M"], i, contributions[i], starting_player, variable_values["r"], variable_values["beta"], variable_values["epsilon"], result, None, current_process])

                if isinstance(variable_values[parameter], int):
                    variable_values[parameter] = variable_values[parameter] * inc
                    variable_values[parameter] = math.ceil(variable_values[parameter])
                else:
                    variable_values[parameter] = variable_values[parameter] * inc
        if lim:
            runs += 1
            if runs == max_runs:
                break

                    

