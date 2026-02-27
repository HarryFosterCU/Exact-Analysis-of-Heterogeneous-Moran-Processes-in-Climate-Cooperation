import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np

here = Path(__file__).resolve()
assets_path = here.parents[3]
df = pd.read_csv(assets_path / "data/processed/comparison_tables/linear/moran_against_imispection/main.csv")



for N, N_frame in df.groupby("N"):
    fig,ax = plt.subplots()
    print(N)
    print(len(N_frame))
    N_frame = N_frame[N_frame["winner"]!="draw"]

    N_frame["diff"] = N_frame["p_C_in_imispection"] - N_frame["p_C_in_moran"]
    N_frame["r"] = N_frame["r"].round(3)

    sns.violinplot(data=N_frame, x="r", y="diff", ax=ax)

    #ax.set_ylabel(r"$\frac{p_C(\text{Introspective
    #Imitation})}{p_C(\text{Moran Process})}$")
    ax.set_ylabel(r"$p_C(\text{Introspective Imitation}) - p_C(\text{Moran process})$")
    plt.xticks(rotation=85)
    folder = Path(here.parent / f"N_eq_{N}")
    folder.mkdir(exist_ok=True)

    plt.tight_layout()
    plt.savefig(here.parent /f"N_eq_{N}/main.pdf")
    plt.close()