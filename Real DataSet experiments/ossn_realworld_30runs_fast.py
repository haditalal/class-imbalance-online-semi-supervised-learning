import os
import time
import tracemalloc
import numpy as np
import pandas as pd
import random
import argparse
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from OSSN import OSNN


# =========================
# Online Normalization (Welford) — IDENTICAL to OSLMF
# =========================
class OnlineStandardizer:
    """
    Online Z-score using Welford updates.
    - Strictly sequential
    - NaN-safe (skip updates on NaN)
    """
    def __init__(self, n_features: int, eps: float = 1e-8):
        self.n_features = n_features
        self.eps = eps
        self.count = np.zeros(n_features, dtype=np.int64)
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)

    def update_and_transform(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64, copy=True)
        out = x.copy()

        for j in range(self.n_features):
            v = x[j]
            if np.isnan(v):
                continue

            self.count[j] += 1
            c = self.count[j]
            delta = v - self.mean[j]
            self.mean[j] += delta / c
            delta2 = v - self.mean[j]
            self.M2[j] += delta * delta2

            if c > 1:
                var = self.M2[j] / (c - 1)
                std = np.sqrt(var + self.eps)
            else:
                std = 1.0
            out[j] = (v - self.mean[j]) / std

        return out


def online_normalize_stream(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    stdzr = OnlineStandardizer(p)
    Xn = np.zeros_like(X, dtype=np.float64)
    for i in range(n):
        Xn[i] = stdzr.update_and_transform(X[i])
    return Xn


# =========================
# Data Loader — IDENTICAL logic to OSLMF
# =========================
def load_dataset_auto_csv(path: str):
    df = pd.read_csv(path)

    class_column = None
    for col in df.columns[::-1]:
        if df[col].dtype == object:
            class_column = col
            break
        if df[col].dtype != object and df[col].nunique(dropna=True) <= 50:
            class_column = col
            break

    if class_column is None:
        raise ValueError(f"Could not detect label column in {path}")

    y_raw = df[class_column]
    X = df.drop(columns=[class_column]).copy()

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.Categorical(X[col]).codes.astype(float)
    X = X.astype(float).values

    y = np.full(len(df), -1, dtype=int)
    notna = ~pd.isna(y_raw)

    if y_raw.dtype == object:
        codes = pd.Categorical(y_raw[notna]).codes
        y[notna.values] = codes.astype(int)
    else:
        y[notna.values] = y_raw[notna].astype(int).values

    return X, y


# =========================
# One-vs-Rest binarization — IDENTICAL to OSLMF
# =========================
def binarize_labels_ovr(y: np.ndarray):
    y = y.astype(int, copy=True)
    mask = (y != -1)
    uniq = np.unique(y[mask])

    if len(uniq) < 2:
        raise ValueError("Not enough labeled classes after ignoring -1.")

    if len(uniq) == 2:
        u_sorted = np.sort(uniq)
        m = {int(u_sorted[0]): 0, int(u_sorted[1]): 1}
        y2 = y.copy()
        y2[mask] = np.vectorize(m.get)(y[mask])
        return y2, int(u_sorted[1])

    vals, counts = np.unique(y[mask], return_counts=True)
    order = np.lexsort((vals, counts))
    minority_class = int(vals[order[0]])

    y_bin = np.zeros_like(y, dtype=int)
    y_bin[~mask] = -1
    y_bin[mask] = (y[mask] == minority_class).astype(int)

    return y_bin, minority_class


# =========================
# Labeling — IDENTICAL to OSLMF
# =========================
def label_uniform_stream(y_true: np.ndarray, label_rate: float, seed: int):
    n = len(y_true)
    flags = np.full(n, -1, dtype=int)
    if label_rate >= 1.0:
        return y_true.copy()

    rng = np.random.default_rng(seed)
    num_labeled = int(round(label_rate * n))
    if num_labeled <= 0:
        return flags

    step = max(1, n // num_labeled)
    offset = int(rng.integers(0, step))
    idx = offset + step * np.arange(num_labeled)
    idx = idx[idx < n]
    flags[idx] = y_true[idx]
    return flags


def label_nonuniform_stream(X: np.ndarray, y_true: np.ndarray, label_rate: float, seed: int, k: int = 5):
    n = len(y_true)
    flags = np.full(n, -1, dtype=int)
    if label_rate >= 1.0:
        return y_true.copy()

    rng = np.random.default_rng(seed)

    for cls in [0, 1]:
        idx_cls = np.where(y_true == cls)[0]
        n_cls = len(idx_cls)
        if n_cls == 0:
            continue

        num_labels = int(round(label_rate * n_cls))
        if num_labels <= 0:
            continue

        X_cls = X[idx_cls]
        k_actual = min(k, n_cls)

        if k_actual > 1:
            km = KMeans(n_clusters=k_actual, n_init=5, random_state=seed)
            cluster_labels = km.fit_predict(X_cls)
        else:
            cluster_labels = np.zeros(n_cls, dtype=int)

        counts = np.array([np.sum(cluster_labels == i) for i in range(k_actual)])
        quota = np.round(num_labels * counts / (counts.sum() + 1e-12)).astype(int)

        diff = num_labels - quota.sum()
        if diff > 0:
            order = np.argsort(-counts)
            for i in range(diff):
                quota[order[i % k_actual]] += 1
        elif diff < 0:
            order = np.argsort(quota)
            i = 0
            while diff < 0 and i < k_actual:
                if quota[order[i]] > 0:
                    quota[order[i]] -= 1
                    diff += 1
                i += 1

        selected = []
        for c in range(k_actual):
            idx_c = np.where(cluster_labels == c)[0]
            n_select = min(quota[c], len(idx_c))
            if n_select > 0:
                selected.extend(rng.choice(idx_c, size=n_select, replace=False))

        if selected:
            sel_global = idx_cls[np.array(selected)]
            flags[sel_global] = y_true[sel_global]

    return flags


# =========================
# Metrics — same definition
# =========================
def compute_accuracy_gmean(y_true, y_pred):
    acc = (y_true == y_pred).mean()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    sensitivity = TP / (TP + FN + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    gmean = np.sqrt(sensitivity * specificity)

    return float(acc), float(gmean)


def stable_seed(run_seed: int, *parts) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    h = zlib.crc32(s) & 0xFFFFFFFF
    return int(run_seed + (h % 1000000))


# =========================
# Worker (one OSNN experiment)
# =========================
def _worker_osnn(task):
    (task_idx, stream_name, X_norm, y_true, y_semi,
     H, N, lam, alpha, beta, gamma,
     model_seed, fadingFactor, type_flag) = task

    np.random.seed(model_seed)
    random.seed(model_seed)

    D = np.column_stack([X_norm, y_semi, y_true])

    preds = OSNN(
        D, N, H, lam, alpha, beta, gamma, fadingFactor, type=type_flag
    )

    y_t = preds[:, 2].astype(int)
    y_p = preds[:, 1].astype(int)

    acc, gmean = compute_accuracy_gmean(y_t, y_p)

    return task_idx, acc, gmean


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--out_dir", type=str, default="results_runs_osnn")
    parser.add_argument("--dataset_dir", type=str, default="Real_datasets")
    parser.add_argument("--param_xlsx", type=str, default="best_param_allscenario.xlsx")
    parser.add_argument("--base_seed", type=int, default=12345)
    parser.add_argument("--no_mem_trace", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    run_id = int(args.task_id)
    run_seed = int(args.base_seed + run_id)

    np.random.seed(run_seed)
    random.seed(run_seed)

    print(f"[RUN] id={run_id} seed={run_seed}")

    # CPU workers
    cpus_total = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    threads_per_worker = int(os.environ.get("OMP_NUM_THREADS", "1"))
    N_WORKERS = max(1, cpus_total // threads_per_worker)
    print(f"[CPU] workers={N_WORKERS}")

    # -------------------------
    # Load params
    # -------------------------
    param_df = pd.read_excel(args.param_xlsx)
    param_df.columns = [c.strip() for c in param_df.columns]

    # -------------------------
    # Load + preprocess datasets (once per run)
    # -------------------------
    DATASETS = [
        "INSECTS incremental_imbalanced.csv",
        "INSECTS incremental-abrupt_imbalanced.csv",
        "Keystroke.csv",
        "NOAA.csv",
        "Ozone.csv",
    ]
    LABEL_RATIOS = [0.05, 0.10, 0.20, 0.50, 1.00]
    LABELING_STRATEGIES = ["uniform", "nonuniform"]

    dataset_cache = {}

    t_load0 = time.time()
    for dataset_file in DATASETS:
        path = os.path.join(args.dataset_dir, dataset_file)
        stream_name = dataset_file.replace(".csv", "")

        X_raw, y_raw = load_dataset_auto_csv(path)
        y_bin, minority_class = binarize_labels_ovr(y_raw)
        X_norm = online_normalize_stream(X_raw)

        dataset_cache[stream_name] = {
            "X_norm": X_norm,
            "y_true": y_bin.astype(int),
            "minority_class": minority_class,
        }

        print(f"[LOAD] {stream_name} n={X_norm.shape[0]} p={X_norm.shape[1]} minority_orig={minority_class}")

    print(f"[LOAD] done in {time.time() - t_load0:.2f}s")

    # -------------------------
    # Precompute label masks (once per condition)
    # -------------------------
    label_cache = {}
    for stream_name, pack in dataset_cache.items():
        X_norm = pack["X_norm"]
        y_true = pack["y_true"]

        for lr in LABEL_RATIOS:
            for strategy in LABELING_STRATEGIES:
                cond_seed = stable_seed(run_seed, stream_name, lr, strategy)

                if strategy == "uniform":
                    y_semi = label_uniform_stream(y_true, lr, cond_seed)
                else:
                    y_semi = label_nonuniform_stream(X_norm, y_true, lr, cond_seed)

                label_cache[(stream_name, lr, strategy)] = y_semi

    # -------------------------
    # Build OSNN tasks (parallel)
    # -------------------------
    tasks = []
    task_idx = 0

    fadingFactor = 0.999
    type_flag = 0

    for row_i in range(len(param_df)):
        row = param_df.iloc[row_i]
        model_name = str(row["Model"]).strip()

        if not model_name.startswith("OSNN"):
            continue

        H = int(row["H"])
        N = int(row["N"])
        lam = float(row["lam"])
        alpha = float(row["alpha"])
        beta = float(row["beta"])
        gamma = float(row["gamma"])

        for stream_name, pack in dataset_cache.items():
            X_norm = pack["X_norm"]
            y_true = pack["y_true"]

            for lr in LABEL_RATIOS:
                for strategy in LABELING_STRATEGIES:
                    cond_seed = stable_seed(run_seed, stream_name, lr, strategy)
                    model_seed = stable_seed(run_seed, stream_name, lr, strategy, "model")

                    y_semi = label_cache[(stream_name, lr, strategy)]

                    tasks.append((
                        task_idx, stream_name, X_norm, y_true, y_semi,
                        H, N, lam, alpha, beta, gamma,
                        model_seed, fadingFactor, type_flag
                    ))
                    task_idx += 1

    # -------------------------
    # Run OSNN in parallel
    # -------------------------
    results_by_idx = {}

    t0_all = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(_worker_osnn, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            idx, acc, gmean = fut.result()
            results_by_idx[idx] = (acc, gmean)
            if i % 10 == 0:
                print(f"[RUN {run_id}] completed {i}/{len(tasks)}")

    # -------------------------
    # Collect results (stable order)
    # -------------------------
    rows = []
    tptr = 0

    for row_i in range(len(param_df)):
        row = param_df.iloc[row_i]
        model_name = str(row["Model"]).strip()
        if not model_name.startswith("OSNN"):
            continue

        H = int(row["H"])
        N = int(row["N"])
        lam = float(row["lam"])
        alpha = float(row["alpha"])
        beta = float(row["beta"])
        gamma = float(row["gamma"])

        for stream_name, pack in dataset_cache.items():
            minority_class = pack["minority_class"]

            for lr in LABEL_RATIOS:
                for strategy in LABELING_STRATEGIES:
                    acc, gmean = results_by_idx[tptr]

                    rows.append({
                        "Run": int(run_id),
                        "Seed": int(run_seed),
                        "Model": model_name,
                        "Dataset": stream_name,
                        "Minority_Class_Orig": int(minority_class),
                        "Labeling_Strategy": strategy,
                        "Label_Ratio": float(lr),
                        "H": H, "N": N, "lam": lam, "alpha": alpha, "beta": beta, "gamma": gamma,
                        "Accuracy": float(acc),
                        "GMean": float(gmean),
                    })
                    tptr += 1

    out_path = os.path.join(args.out_dir, f"run_{run_id:02d}.parquet")
    pd.DataFrame(rows).to_parquet(out_path, index=False)

    print(f"\n[SAVE] Run {run_id} saved: {out_path}")
    print(f"[DONE] Wall time: {time.time() - t0_all:.2f} sec")


if __name__ == "__main__":
    main()
