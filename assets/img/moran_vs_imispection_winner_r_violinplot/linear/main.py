print("started")
import pandas as pd
print("pandas")
import matplotlib.pyplot as plt
print("plt")
from pathlib import Path
print("path")
import seaborn as sns
print("sns")


here = Path(__file__).resolve()
print("herepath")
assets_path = here.parents[3]
print("imported")
df = pd.read_csv(assets_path / "data/processed/comparison_tables/linear/moran_against_imispection/main.csv")
print("read")
for N, N_frame in df.groupby("N"):
    print(N)
    sns.violinplot(data=N_frame, x="winner", y="r")
    plt.axhline(y=N)

    folder = Path(here.parent / f"N_eq_{N}")
    folder.mkdir(exist_ok=True)

    plt.savefig(here.parent /f"N_eq_{N}/main.pdf")
    plt.close()