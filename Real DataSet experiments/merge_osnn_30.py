import glob
import pandas as pd

INPUT_DIR = "results_runs_osnn"
OUT_FILE = "OSNN_30runs_all.parquet"
OUT_CSV  = "OSNN_30runs_all.csv"

files = sorted(glob.glob(f"{INPUT_DIR}/run_*.parquet"))
assert len(files) > 0, "No run files found!"

dfs = []
for f in files:
    df = pd.read_parquet(f)
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

print("Before dedup:", len(all_df))

# الأعمدة التي تعرّف التجربة الواحدة بشكل فريد
KEY_COLS = [
    "Run", "Model", "Dataset",
    "Labeling_Strategy", "Label_Ratio",
    "H", "N", "lam", "alpha", "beta", "gamma"
]

# حذف النسخ المكررة
all_df = all_df.drop_duplicates(subset=KEY_COLS, keep="first")

print("After dedup:", len(all_df))

all_df.to_parquet(OUT_FILE, index=False)
all_df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_FILE)
print("Saved:", OUT_CSV)
