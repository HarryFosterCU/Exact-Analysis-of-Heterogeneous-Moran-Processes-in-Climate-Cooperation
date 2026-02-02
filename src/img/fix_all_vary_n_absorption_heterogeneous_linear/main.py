import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

np.set_printoptions(suppress=False)
import matplotlib.patches as mpatches
import matplotlib.lines as mlinesge
import scipy
import sys
import pathlib

file_path = pathlib.Path(__file__)
root_path = (file_path / "../../../../").resolve()

sys.path.append(str(root_path))
import src.main as main
import src.contribution_rules as cr


def heterogeneous_contribution_pgg_fitness_function(
    state, epsilon, r, contribution_vector, **kwargs
):
    """Public goods fitness function where players contribute a different
    amount. They then have a return of 1 + (payoff * selection_intensity). This
    is the selection intensity $\epsilon$, which determines the effect of
    payoff on a player's fitness.
    Parameters
    -----------
    state: numpy.array, the ordered set of actions each player takes; 1 for
    contributing, 0 for free-riding.
    epsilon: float, the selection intensity determining the effect of payoff on
    a player's fitness. Must satisfy $0 < \epsilon < max_i(\frac{N}{(N-r)\alpha_i})$ if r<N
    r: float, the parameter which the public goods is multiplied by
    contribution_vector: numpy.array, the value which each player contributes
    Returns
    -------
    numpy.array: an ordered vector of each player's fitness."""

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

    return 1 + (payoff_vector * epsilon)


def get_absorption_probability_vector_1_contributor_low(
    n_values, r, M, probability_function, **kwargs
):
    y = []
    for N in n_values:
        state_space = main.get_state_space(N=N, k=2)

        transition_matrix = main.generate_transition_matrix(
            state_space=state_space,
            fitness_function=heterogeneous_contribution_pgg_fitness_function,
            r=1.8,
            epsilon=0.1,
            compute_transition_probability=probability_function,
            contribution_vector=main.get_deterministic_contribution_vector(
                cr.linear_contribution_rule, N=N, M=M
            ),
            **kwargs
        )

        state = np.zeros(N, dtype=int)
        state[0] = 1
        state_index = np.all(state_space == state, axis=1)
        print(state_space[np.where(state_index)[0]])

        absorption_probability = main.approximate_absorption_matrix(
            transition_matrix=transition_matrix
        )

        y.append(absorption_probability[np.where(state_index)[0] - 1, 1])
    return np.array(y)


def get_absorption_probability_vector_1_contributor_high(
    n_values, r, M, probability_function, **kwargs
):
    y = []
    for N in n_values:
        state_space = main.get_state_space(N=N, k=2)

        transition_matrix = main.generate_transition_matrix(
            state_space=state_space,
            fitness_function=heterogeneous_contribution_pgg_fitness_function,
            r=1.8,
            epsilon=0.1,
            compute_transition_probability=probability_function,
            contribution_vector=main.get_deterministic_contribution_vector(
                cr.linear_contribution_rule, N=N, M=M
            ),
            **kwargs
        )
        print(state_space[1])

        y.append(
            main.approximate_absorption_matrix(transition_matrix=transition_matrix)[
                0, 1
            ]
        )

    return np.array(y)


def get_absorption_probability_vector_n_minus_1_contributor_low(
    n_values, r, M, probability_function, **kwargs
):
    y = []
    for N in n_values:
        state_space = main.get_state_space(N=N, k=2)

        transition_matrix = main.generate_transition_matrix(
            state_space=state_space,
            fitness_function=heterogeneous_contribution_pgg_fitness_function,
            r=1.8,
            epsilon=0.1,
            compute_transition_probability=probability_function,
            contribution_vector=main.get_deterministic_contribution_vector(
                cr.linear_contribution_rule, N=N, M=M
            ),
            **kwargs
        )

        state = np.ones(N, dtype=int)
        state[0] = 0
        state_index = np.all(state_space == state, axis=1)
        print(state_space[np.where(state_index)[0]])

        absorption_probability = main.approximate_absorption_matrix(
            transition_matrix=transition_matrix
        )

        y.append(absorption_probability[np.where(state_index)[0] - 1, 1])

    return np.array(y)


def get_absorption_probability_vector_n_minus_1_contributor_high(
    n_values, r, M, probability_function, **kwargs
):
    y = []
    for N in n_values:
        state_space = main.get_state_space(N=N, k=2)

        transition_matrix = main.generate_transition_matrix(
            state_space=state_space,
            fitness_function=heterogeneous_contribution_pgg_fitness_function,
            r=1.8,
            epsilon=0.1,
            compute_transition_probability=probability_function,
            contribution_vector=main.get_deterministic_contribution_vector(
                cr.linear_contribution_rule, N=N, M=M
            ),
            **kwargs
        )

        state = np.ones(N, dtype=int)
        state[N - 1] = 0
        state_index = np.all(state_space == state, axis=1)
        print(state_space[np.where(state_index)[0]])

        absorption_probability = main.approximate_absorption_matrix(
            transition_matrix=transition_matrix
        )

        y.append(absorption_probability[np.where(state_index)[0] - 1, 1])

    return np.array(y)


n_range = 10
r = 1.8
M = 20
beta = 0.5
n_values = np.arange(3, n_range)


fig, axes = plt.subplots(3, 3, figsize=(12, 8))
axes = axes.flatten()
axes[5].axis("off")
axes[8].axis("off")
fig.suptitle(
    r"Absorption probabilities for Full Cooperation in a Linearly Heterogeneous PGG with varying population size."
)
fig.text(
    0.5,
    -0.02,
    r"$\beta = 0.5$, $\epsilon = 0.1$, $r = 1.8$, $M=20$",
    ha="right",
    va="bottom",
    fontsize=12,
)
fig.text(
    0.2,
    -0.02,
    "Dashed – High contributing mutant\nSolid – Low contributing mutant",
    ha="right",
    va="bottom",
    fontsize=12,
)


axes[2].scatter(
    range(10),
    np.array([cr.linear_contribution_rule(index=n, N=10, M=M) for n in range(10)]),
    label="",
)

axes[2].set_title(r"E.g. Contrbutions for $N=10$")
axes[2].set_xlabel(r"$N$")
axes[2].set_ylabel(r"$\rho_C$")

fermi_contributor_low = get_absorption_probability_vector_1_contributor_low(
    n_values=n_values,
    r=r,
    M=M,
    probability_function=main.compute_fermi_transition_probability,
    selection_intensity=beta,
)
moran_contributor_low = get_absorption_probability_vector_1_contributor_low(
    n_values=n_values,
    r=r,
    M=M,
    probability_function=main.compute_moran_transition_probability,
)
II_contributor_low = get_absorption_probability_vector_1_contributor_low(
    n_values=n_values,
    r=r,
    M=M,
    probability_function=main.compute_imitation_introspection_transition_probability,
    selection_intensity=beta,
)

axes[0].plot(
    n_values,
    fermi_contributor_low,
    label="Fermi Imitation Dynamics",
)
axes[0].plot(
    n_values,
    moran_contributor_low,
    label="Moran Process",
)
axes[0].plot(
    n_values,
    II_contributor_low,
    label="Introspective Imitation Dynamics",
)

axes[0].set_title("1 low-contributing initial contributor")
axes[0].set_xlabel(r"$N$")
axes[0].set_ylabel(r"$\rho_C$")
axes[0].set_ylim(0, 0.9)

fermi_contributor_high = get_absorption_probability_vector_1_contributor_high(
    n_values=n_values,
    r=r,
    M=M,
    probability_function=main.compute_fermi_transition_probability,
    selection_intensity=0.5,
)

moran_contributor_high = get_absorption_probability_vector_1_contributor_high(
    n_values=n_values,
    r=r,
    M=M,
    probability_function=main.compute_moran_transition_probability,
)

II_contributor_high = get_absorption_probability_vector_1_contributor_high(
    n_values=n_values,
    r=r,
    M=M,
    probability_function=main.compute_imitation_introspection_transition_probability,
    selection_intensity=0.5,
)

axes[3].plot(
    n_values,
    fermi_contributor_high,
    label="Fermi Imitation Dynamics",
)
axes[3].plot(
    n_values,
    moran_contributor_high,
    label="Moran Process",
)
axes[3].plot(
    n_values,
    II_contributor_high,
    label="Introspective Imitation Dynamics",
)

axes[3].set_title("1 high-contributing initial contributor")
axes[3].set_xlabel(r"$N$")
axes[3].set_ylabel(r"$\rho_C$")
axes[3].set_ylim(0, 0.9)

axes[6].plot(n_values, fermi_contributor_low, color="blue")
axes[6].plot(n_values, moran_contributor_low, color="orange")
axes[6].plot(n_values, II_contributor_low, color="green")

axes[6].plot(n_values, fermi_contributor_high, color="blue", linestyle="dashed")
axes[6].plot(n_values, moran_contributor_high, color="orange", linestyle="dashed")
axes[6].plot(n_values, II_contributor_high, color="green", linestyle="dashed")
axes[6].set_title("1 initial contributor")
axes[6].set_xlabel(r"$N$")
axes[6].set_ylabel(r"$\rho_C$")

fermi_defector_low = get_absorption_probability_vector_n_minus_1_contributor_low(
    n_values=n_values,
    r=r,
    M=M,
    probability_function=main.compute_fermi_transition_probability,
    selection_intensity=0.5,
)
moran_defector_low = get_absorption_probability_vector_n_minus_1_contributor_low(
    n_values=n_values,
    r=r,
    M=M,
    probability_function=main.compute_moran_transition_probability,
)
II_defector_low = get_absorption_probability_vector_n_minus_1_contributor_low(
    n_values=n_values,
    r=r,
    M=M,
    probability_function=main.compute_imitation_introspection_transition_probability,
    selection_intensity=0.5,
)

axes[1].plot(
    n_values,
    fermi_defector_low,
    label="Fermi Imitation Dynamics",
)
axes[1].plot(
    n_values,
    moran_defector_low,
    label="Moran Process",
)
axes[1].plot(
    n_values,
    II_defector_low,
    label="Introspective Imitation Dynamics",
)

axes[1].set_title("1 low-contributing initial defector")
axes[1].set_xlabel(r"$N$")
axes[1].set_ylabel(r"$\rho_C$")
axes[1].set_ylim(0, 0.9)

fermi_defector_high = get_absorption_probability_vector_n_minus_1_contributor_high(
    n_values=n_values,
    r=r,
    M=M,
    probability_function=main.compute_fermi_transition_probability,
    selection_intensity=0.5,
)

moran_defector_high = get_absorption_probability_vector_n_minus_1_contributor_high(
    n_values=n_values,
    r=r,
    M=M,
    probability_function=main.compute_moran_transition_probability,
)

II_defector_high = get_absorption_probability_vector_n_minus_1_contributor_high(
    n_values=n_values,
    r=r,
    M=M,
    probability_function=main.compute_imitation_introspection_transition_probability,
    selection_intensity=0.5,
)

axes[4].plot(
    n_values,
    fermi_defector_high,
    label="Fermi Imitation Dynamics",
)
axes[4].plot(
    n_values,
    moran_defector_high,
    label="Moran Process",
)
axes[4].plot(
    n_values,
    II_defector_high,
    label="Introspective Imitation Dynamics",
)

axes[4].set_title("1 high-contributing initial defector")
axes[4].set_xlabel(r"$N$")
axes[4].set_ylabel(r"$\rho_C$")
axes[4].set_ylim(0, 0.9)


axes[7].plot(n_values, fermi_defector_low, color="blue")
axes[7].plot(n_values, moran_defector_low, color="orange")
axes[7].plot(n_values, II_defector_low, color="green")

axes[7].plot(n_values, fermi_defector_high, color="blue", linestyle="dashed")
axes[7].plot(n_values, moran_defector_high, color="orange", linestyle="dashed")
axes[7].plot(n_values, II_defector_high, color="green", linestyle="dashed")
axes[7].set_title("1 initial defector")
axes[7].set_xlabel(r"$N$")
axes[7].set_ylabel(r"$\rho_C$")

plt.tight_layout()
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1, 0.5))
plt.savefig(file_path.parent / "main.pdf", bbox_inches="tight")
