import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

files = [
    "OSNN_30runs_all_non_uniform.csv",
    "OSNN_30runs_all_uniform.csv",
]

def make_wide(csv_path: Path):
    df = pd.read_csv(csv_path)

    wide = (
        df.pivot(
            index=["Run", "Dataset"],
            columns="Label_Ratio",
            values="GMean"
        )
        .reset_index()
    )

    # إعادة تسمية الأعمدة لتناسب SPSS
    new_cols = []
    for c in wide.columns:
        if isinstance(c, float):
            pct = int(round(c * 100))
            new_cols.append(f"LR_{pct}")
        else:
            new_cols.append(c)
    wide.columns = new_cols

    # ترتيب الأعمدة
    lr_cols = sorted(
        [c for c in wide.columns if c.startswith("LR_")],
        key=lambda x: int(x.split("_")[1])
    )
    wide = wide[["Run", "Dataset"] + lr_cols]

    out_path = csv_path.with_name(csv_path.stem + "_WIDE.csv")
    wide.to_csv(out_path, index=False)
    print(f"Saved: {out_path.name}")

for f in files:
    make_wide(BASE_DIR / f)

print("Done. Wide files saved in the same folder.")
