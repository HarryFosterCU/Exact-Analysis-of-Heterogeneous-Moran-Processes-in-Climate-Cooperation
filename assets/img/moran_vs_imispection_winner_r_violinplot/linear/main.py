import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

here = Path(__file__).resolve()
assets_path = here.parents[3]
df = pd.read_csv(assets_path / "data/processed/comparison_tables/linear/moran_against_imispection/main.csv")

for N, N_frame in df.groupby("N"):
    sns.violinplot(data=N_frame, x="winner", y="r")
    plt.axhline(y=N)

    folder = Path(here.parent / f"N_eq_{N}")
    folder.mkdir(exist_ok=True)

    plt.savefig(here.parent /f"N_eq_{N}/main.pdf")
    plt.close()