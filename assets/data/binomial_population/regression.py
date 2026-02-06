import numpy as np
import pandas as pd
import statsmodels.api as sm
import pathlib

file_path = pathlib.Path(__file__)
df = pd.read_csv(file_path.parent / "data.csv")

df["Y"] = np.where(
    df["Process"] == "Introspection",
    df["$p_{C}$"],
    df["$\\rho_{C}$"]
)

numeric_cols = ["Y", "r", "$\\beta$", "$\\epsilon$", "M"]
for col in numeric_cols:
    df[col] = pd.to_numeric(np.real(df[col]), errors="coerce")

model_df = df[["Y", "r", "$\\beta$", "$\\epsilon$", "M"]].dropna()

y = model_df["Y"]
X = model_df[["r", "$\\beta$", "$\\epsilon$", "M"]]
X = sm.add_constant(X) 

model = sm.OLS(y, X).fit()

print(model.summary())
