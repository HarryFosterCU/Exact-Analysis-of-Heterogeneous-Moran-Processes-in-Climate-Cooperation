import pandas as pd
import numpy as np
import sympy as sym
import pathlib
import sys
import uuid
import math

file_path = pathlib.Path(__file__)
root_path = (file_path / "../../../../").resolve()
print(root_path)
data_path = (file_path /"../../../data").resolve()

sys.path.append(str(root_path))
import src.main as main
import src.fitness_functions as fitness_functions
import src.contribution_rules as contribution_rules



if pathlib.Path(data_path / "procecssed/r_effect_on_std_p_C/binomial_fermi/main.csv").exists():
    df = pd.read_csv(data_path / "procecssed/r_effect_on_std_p_C/binomial_fermi/main.csv")
else:
    csvs = [
        data_path/"raw/binomial_population_fermi/main.csv",
        data_path/"raw/binomial_population_moran/main.csv",
        data_path/"raw/binomial_population_introspection/main.csv",
        data_path/"raw/binomial_population_imispection/main.csv",
        data_path/"raw/linear_population_fermi/main.csv",
        data_path/"raw/linear_population_moran/main.csv",
        data_path/"raw/linear_population_introspection/main.csv",
        data_path/"raw/linear_population_imispection/main.csv",
    ]
    process_names = ["fermi", "moran", "introspection", "introspective imitation", "fermi", "moran", "introspection", "introspective imitation"]
    population_types = ["binomial", "binomial", "binomial", "binomial", "linear", "linear", "linear", "linear"]
    
    
    N = []
    r = []
    p_C = []
    process = []
    population = []
    beta = []
    epsilon = []
    std_alpha = []

    for (population,process,data_file) in zip(population_types, process_names, csvs):
        print(process)
        df_temp = pd.read_csv(data_file)
        if process == "introspection":
            group_statement = "UID"
        else:
            group_statement = ["UID", "first_alpha"]
        for (_, first_alpha), frame in df_temp.groupby(group_statement):
            p_C.append(float(frame["p_C"].iloc[0].strip("[]")))
            print("worked")
            N.append(frame["N"].iloc[0])
            r.append(frame["r"].iloc[0])
            std_alpha.append(frame["alpha_i"].std())
            if process in ("fermi","introspection"):
                beta.append(frame["beta"].iloc[0])
                epsilon.append(None)
            elif process == "moran":
                epsilon.append(frame["epsilon"].iloc[0])
            else:
                beta.append(frame["beta"].iloc[0])
                epsilon.append(frame["epsilon"].iloc[0])

    max_data = len(N)

    df = pd.DataFrame(
        {
            "N": N[:max_data],
            "epsilon": epsilon[:max_data],
            "r": r[:max_data],
            "process":process[:max_data],
            "population":population[:max_data],
            "beta":beta[:max_data],
            "epsilon":epsilon[:max_data],
            "p_C": p_C[:max_data],
        }
    )

    df.to_csv(data_path / "procecssed/r_effect_on_std_p_C/binomial_fermi/main.csv")




    

            
            
        
    
