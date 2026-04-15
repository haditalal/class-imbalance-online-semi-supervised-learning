import numpy as np
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_generation_osnn import generate_stream
from OSSN import OSNN # type: ignore
import os
import sys
import tempfile

def np_encoder(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError
# ============================================================
# 1. CONFIGURATION
# ============================================================

LABELING_STRATEGIES = ["uniform", "nonuniform"]
LABELING_RATIOS = [0.05, 0.10, 0.20, 0.50, 1.00]
IMBALANCE_RATIOS = [0.01, 0.10, 0.30, 0.50]

DATASETS = {
    "SINE": ["SINE1", "SINE2"],
    "AGRAWAL": ["AGRAWAL1", "AGRAWAL2", "AGRAWAL3", "AGRAWAL4"],
    "SEA": ["SEA1", "SEA2"],
    "STAGGER": ["STAGGER1", "STAGGER2"]
}

TUNING_SIZE = {
    "SINE": 20000,
    "AGRAWAL": 24000,
    "SEA": 20000,
    "STAGGER": 16000
}

# 📌 ثابت أثناء التونينغ
STREAM_SEED_TUNING = 42
#SEEDS_MODEL_TUNING = [101, 202, 303]  # للـ median G-Mean
SEEDS_MODEL_TUNING = [101]

# ============================================================
# 2. HYPERPARAMETER SPACE
# ============================================================

HYPERPARAM_SPACE = {
    "H":     (10, 100),
    "N":     (5, 50),
    "beta":  (0.1, 2.0),
    "gamma": (0.1, 2.0),
    "alpha": (1e-3, 1.0),
    "lam":   (1e-3, 1.0)
}

#N_HYPER_TRIALS = 1000  # عدد إعدادات التونينغ
N_HYPER_TRIALS = 1000  # عدد إعدادات التونينغ
HYPER_SEED = 999       # seed ثابت لتوليد البراميترات

# ============================================================
# 3. G-MEAN FUNCTION
# ============================================================

def gmean_score(predictions):
    y_true = predictions[:, 2]
    y_pred = predictions[:, 1]
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    sens = TP / (TP + FN + 1e-10)
    spec = TN / (TN + FP + 1e-10)
    return np.sqrt(sens * spec)

# ============================================================
# 4. توليد هايبر باراميترات عشوائية (ثابتة عبر كل الحالات)
# ============================================================

def sample_hyperparams(n_samples, seed):
    rng = np.random.default_rng(seed)
    params_list = []
    for _ in range(n_samples):
        H = rng.integers(*HYPERPARAM_SPACE["H"])
        N = rng.integers(*HYPERPARAM_SPACE["N"])
        beta = rng.uniform(*HYPERPARAM_SPACE["beta"])
        gamma = rng.uniform(*HYPERPARAM_SPACE["gamma"])
        alpha = 10 ** rng.uniform(np.log10(HYPERPARAM_SPACE["alpha"][0]), np.log10(HYPERPARAM_SPACE["alpha"][1]))
        lam = 10 ** rng.uniform(np.log10(HYPERPARAM_SPACE["lam"][0]), np.log10(HYPERPARAM_SPACE["lam"][1]))
        params_list.append((H, N, lam, alpha, beta, gamma))
    return params_list

# توليد نفس المجموعة مرة واحدة فقط
HYPERPARAM_CANDIDATES = sample_hyperparams(N_HYPER_TRIALS, HYPER_SEED)
# عدد العمّال = عدد الأنوية المخصصة من SLURM
cpus_total = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
threads_per_worker = int(os.environ.get("OMP_NUM_THREADS", "1"))
N_WORKERS = cpus_total // threads_per_worker

ALL_CASES = list(itertools.product(
    DATASETS.keys(), LABELING_STRATEGIES, LABELING_RATIOS, IMBALANCE_RATIOS
))

def get_case_from_index(idx: int):
    return ALL_CASES[idx]

# ============================================================
# 5. دالة تنفيذ تجربة واحدة
# ============================================================

# سيتم تحميل D مرة واحدة في كل process عبر initializer
D_GLOBAL = None

def _init_worker(D_path):
    """initializer لكل عامل: يحمّل D من الملف إلى الذاكرة مرة واحدة.s."""
    import numpy as _np
    global D_GLOBAL
    D_GLOBAL = _np.load(D_path)  # تحميل مباشر إلى RAM بدون mmap

def run_single_trial(params):
    """يشغّل تجربة واحدة مستخدمًا D_GLOBAL (memmap)."""
    H, N, lam, alpha, beta, gamma = params
    gmeans = []
    for model_seed in SEEDS_MODEL_TUNING:
        np.random.seed(model_seed)
        pred = OSNN(D_GLOBAL, N, H, lam, alpha, beta, gamma, fadingFactor=0.999, type=0)
        gmeans.append(gmean_score(pred))
    return np.median(gmeans), params

def run_batch(batch):
    best_g, best_p = -1.0, None
    for p in batch:
        g, params = run_single_trial(p)
        if g > best_g:
            best_g, best_p = g, params
    return best_g, best_p

# ============================================================
# 6. دالة التونينغ لحالة واحدة
# ============================================================

def build_family_stream(category, labeling_strategy, labeling_ratio, imbalance_ratio, seed):
    TARGET_SAMPLES = TUNING_SIZE[category]
    concept_len = 4000
    streams = DATASETS[category]  # كل الستريمات في هذه العائلة
    segments = []
    total = 0

    # نبني نسخة abrupt و gradual لكل ستريم
    built = []
    for s in streams:
        D_a, seq_a = generate_stream(
            stream_type=s,
            n_samples=None,
            drift_type='abrupt',
            imbalance_ratio=imbalance_ratio,
            label_rate=labeling_ratio,
            label_strategy=labeling_strategy,
            seed=seed
        )
        D_g, seq_g = generate_stream(
            stream_type=s,
            n_samples=None,
            drift_type='gradual',
            imbalance_ratio=imbalance_ratio,
            label_rate=labeling_ratio,
            label_strategy=labeling_strategy,
            seed=seed
        )
        built.append((D_a, D_g, len(seq_a)))

    # نأخذ المفاهيم بالتناوب حتى نصل للـ TARGET_SAMPLES
    i = 0
    while total < TARGET_SAMPLES:
        progressed = False
        for (D_a, D_g, n_concepts) in built:
            if i < n_concepts:
                start = i * concept_len
                end = start + concept_len
                seg = D_a[start:end] if (i % 2 == 0) else D_g[start:end]
                need = TARGET_SAMPLES - total
                if len(seg) > need:
                    seg = seg[:need]
                segments.append(seg)
                total += len(seg)
                progressed = True
                if total >= TARGET_SAMPLES:
                    break
        if not progressed:
            break
        i += 1

    return np.vstack(segments)

# ============================================================
# 7. تنفيذ جميع الحالات (TUNING فقط)
# ============================================================
def main():
    # 1) حدد الـ case_idx من argv أو من SLURM_ARRAY_TASK_ID
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        case_idx = int(sys.argv[1])
    else:
        case_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    dataset_category, labeling_strategy, labeling_ratio, imbalance_ratio = get_case_from_index(case_idx)
    print(f"\n[ CASE {case_idx} ] {dataset_category} | {labeling_strategy} | L={labeling_ratio} | Imb={imbalance_ratio}")
    """
    # 2)هذا عملت له تعليق لانه يختار فقط اول دات سيت وينحاز الى توزيع محدد 
    base_stream_name = DATASETS[dataset_category][0]
    n_samples = TUNING_SIZE[dataset_category]

    D_abrupt, seq_a = generate_stream(
        stream_type=base_stream_name,
        n_samples=n_samples,
        drift_type='abrupt',
        imbalance_ratio=imbalance_ratio,
        label_rate=labeling_ratio,
        label_strategy=labeling_strategy,
        seed=STREAM_SEED_TUNING
    )
    D_gradual, seq_g = generate_stream(
        stream_type=base_stream_name,
        n_samples=n_samples,
        drift_type='gradual',
        imbalance_ratio=imbalance_ratio,
        label_rate=labeling_ratio,
        label_strategy=labeling_strategy,
        seed=STREAM_SEED_TUNING
    )

    concept_len = 1000
    n_concepts = len(seq_a)
    D_segments = []
    for i in range(n_concepts):
        start = i * concept_len
        end = (i + 1) * concept_len
        if i % 2 == 0:
            D_segments.append(D_abrupt[start:end])
        else:
            D_segments.append(D_gradual[start:end])
    D = np.vstack(D_segments)
    """
    # ============================================================
# ✅ Build a mixed stream from all streams in the family
# ============================================================

# 👇 استدعاء الدالة الجديدة لبناء الـ stream
    D = build_family_stream(dataset_category, labeling_strategy, labeling_ratio, imbalance_ratio, STREAM_SEED_TUNING)


    # نستخدم /tmp مباشرة كمسار ثابت وسريع للتخزين المؤقت
    tmp_dir = "/tmp"
    D_path = os.path.join(tmp_dir, f"ossn_case_{case_idx}.npy")
    np.save(D_path, D)


    # 4) نفّذ التجارب
    best_gmean, best_params = -1.0, None
    
    BATCH_SIZE = 25  # عدد التجارب في كل دفعة
    batches = [HYPERPARAM_CANDIDATES[i:i+BATCH_SIZE] for i in range(0, len(HYPERPARAM_CANDIDATES), BATCH_SIZE)]

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(D_path,)) as ex:
        futures = [ex.submit(run_batch, batch) for batch in batches]
        for idx, fut in enumerate(as_completed(futures), 1):
            bg, bp = fut.result()
            if bg > best_gmean:
                best_gmean, best_params = bg, bp
            print(f"  Batch {idx}/{len(batches)} done | Best G-Mean: {best_gmean:.4f}")

    # 5) احفظ النتيجة في ملف خاص
    out_line = (f"{dataset_category},{labeling_strategy},{labeling_ratio},{imbalance_ratio},"
                f"{best_params[0]},{best_params[1]},{best_params[2]},{best_params[3]},{best_params[4]},{best_params[5]},"
                f"{best_gmean}\n")

    with open(f"tuning_result_case_{case_idx}.txt", "w") as f:
        f.write(out_line)

    print(f"[ CASE {case_idx} DONE ] Best G-Mean={best_gmean:.4f}")


if __name__ == "__main__":
    main()

