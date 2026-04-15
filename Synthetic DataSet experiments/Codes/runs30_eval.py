import os
import sys
import math
import numpy as np
import pandas as pd

from data_generation_osnn import generate_stream
from OSSN import OSNN  # type: ignore

from concurrent.futures import ProcessPoolExecutor, as_completed


# ================================
# 0) إعدادات أساسية
# ================================
# ملف التونينغ الموحّد الناتج من مرحلة التونينغ (دمج 160 ملف)
TUNING_CSV = "Tuning_All_Parameters.csv"

# الخرائط: الداتا سيت → العائلة (للبحث عن الهايبر)
FAMILY_OF = {
    "SINE1": "SINE",
    "SINE2": "SINE",
    "AGRAWAL1": "AGRAWAL",
    "AGRAWAL2": "AGRAWAL",
    "AGRAWAL3": "AGRAWAL",
    "AGRAWAL4": "AGRAWAL",
    "SEA1": "SEA",
    "SEA2": "SEA",
    "STAGGER1": "STAGGER",
    "STAGGER2": "STAGGER",
}

# قائمة كل الداتاسات (10 كما بالورقة)
DATASETS = [
    "SINE1", "SINE2",
    "AGRAWAL1", "AGRAWAL2", "AGRAWAL3", "AGRAWAL4",
    "SEA1", "SEA2",
    "STAGGER1", "STAGGER2"
]

# قيم التجارب داخل كل Run
DRIFTS = ["abrupt", "gradual"]
LABELING_STRATEGIES = ["uniform", "nonuniform"]
IMBALANCE_LIST = [0.01, 0.10, 0.30, 0.50]            # 4
LABEL_RATIOS = [0.05, 0.10, 0.20, 0.50, 1.00]        # 5  (نفس التونينغ)

# بذور الأساس (تتغير بين الـ Runs)
BASE_STREAM_SEED = 42
BASE_MODEL_SEED  = 101

# عدد الوركرات = (عدد الأنوية من SLURM) ÷ (عدد ثريدات كل عملية من OMP)
cpus_total = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
threads_per_worker = int(os.environ.get("OMP_NUM_THREADS", "1"))
N_WORKERS = max(1, cpus_total // threads_per_worker)

# ================================
# 1) تحميل ملف التونينغ وبناؤه كـ lookup
# ================================
def load_tuning_table(path):
    """
    يتوقع CSV بالأعمدة:
    dataset_category,labeling_strategy,labeling_ratio,imbalance_ratio,H,N,lam,alpha,beta,gamma,best_gmean
    (حيث dataset_category واحدة من: SINE/AGRAWAL/SEA/STAGGER)
    """
    df = pd.read_csv(path)
    # نبني dict للبحث:
    # key = (family, labeling_strategy, round(label_ratio,5), round(imbalance_ratio,5))
    # val = (H, N, lam, alpha, beta, gamma)
    lut = {}
    for _, r in df.iterrows():
        key = (
            str(r["Dataset"]).strip().upper(),
            str(r["LabelingStrategy"]).strip().lower(),
            round(float(r["LabelingRatio"]), 3),
            round(float(r["ImbalanceRatio"]), 3),
        )



        val = (
            int(r["H"]), int(r["N"]),
            float(r["lam"]), float(r["alpha"]),
            float(r["beta"]), float(r["gamma"])
        )
        lut[key] = val
    return lut

# ================================
# 2) المقاييس
# ================================
def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)

def compute_metrics(pred):
    """
    pred: مصفوفة من OSNN (نفس تنسيق التونينغ):
      y_pred = pred[:,1], y_true = pred[:,2]
    """
    y_true = pred[:, 2]
    y_pred = pred[:, 1]

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    acc  = _safe_div(TP + TN, TP + TN + FP + FN)
    prec = _safe_div(TP, TP + FP)
    rec  = _safe_div(TP, TP + FN)
    f1   = _safe_div(2 * prec * rec, prec + rec)
    spec = _safe_div(TN, TN + FP)
    gmean = math.sqrt(rec * spec)

    return acc, prec, rec, f1, spec, gmean

# ================================
# 3) جلب الهايبر باراميترز من LUT
# ================================
def get_hypers_for_case(lut, dataset_name, labeling_strategy, label_ratio, imbalance_ratio):
    family = FAMILY_OF[dataset_name].upper()
    key = (family, labeling_strategy.lower(),
       round(float(label_ratio), 3),
       round(float(imbalance_ratio), 3))

    if key not in lut:
        raise KeyError(f"Hyper-params not found for: {key}")
    return lut[key]  # (H, N, lam, alpha, beta, gamma)

def _worker_single_experiment(args):
    """
    ينفّذ تجربة واحدة (نفس ما كان داخل الحلقة الخماسية) ويرجع (task_idx, row_dict).
    لا يغيّر أي منطق؛ فقط يغلف التجربة كي نقدر نرسلها إلى ProcessPool.
    """
    (task_idx, ds, drift, lab_strategy, imb, lr, stream_seed, model_seed, lut) = args

    # 1) الهايبر
    H, N, lam, alpha, beta, gamma = get_hypers_for_case(
        lut, ds, lab_strategy, lr, imb
    )

    # 2) توليد الستريم
    D, _ = generate_stream(
        stream_type=ds,
        n_samples=None,
        drift_type=drift,
        imbalance_ratio=imb,
        label_rate=lr,
        label_strategy=lab_strategy,
        seed=stream_seed
    )

    # 3) تشغيل OSNN (بنفس الـ seed)
    np.random.seed(model_seed)
    pred = OSNN(D, N, H, lam, alpha, beta, gamma, fadingFactor=0.999, type=0)

    # 4) المقاييس (نفسها)
    acc, prec, rec, f1, spec, gmean = compute_metrics(pred)

    # 5) صفّ النتيجة (نفس الحقول بالضبط)
    row = {
        "Dataset": ds,
        "Drift Speed": drift,
        "Labeling Strategy": lab_strategy,
        "Imbalance": int(round(imb * 100)),
        "Label": int(round(lr * 100)),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Specificity": spec,
        "GMean": gmean,
        "H": H, "N": N, "lam": lam, "alpha": alpha, "beta": beta, "gamma": gamma,
        "stream_seed": stream_seed,
        "model_seed": model_seed
    }
    return task_idx, row



# ================================
# 4) الدالة الرئيسية لــ Run واحد
# ================================
def run_once(run_id: int):
    """
    run_id: رقم الـ Run من 1..30 (يُستخدم لحساب الـ seeds وتسمية الملفات).
    """
    print(f"[Run {run_id}] Loading tuned parameters from: {TUNING_CSV}")
    lut = load_tuning_table(TUNING_CSV)

    # نحسب البذور لهذا الـ Run
    stream_seed = BASE_STREAM_SEED + 1000 * run_id
    model_seed  = BASE_MODEL_SEED  + 2000 * run_id

    rows = []

    # حلقة التجارب = 10 datasets × 2 drifts × 2 strategies × 4 imbalance × 5 label ratios = 800
        # بدلاً من الحلقات: نبني قائمة المهام بنفس ترتيب التكرار الأصلي
    tasks = []
    task_idx = 0
    for ds in DATASETS:
        for drift in DRIFTS:
            for lab_strategy in LABELING_STRATEGIES:
                for imb in IMBALANCE_LIST:
                    for lr in LABEL_RATIOS:
                        tasks.append((
                            task_idx, ds, drift, lab_strategy, imb, lr,
                            stream_seed, model_seed, lut
                        ))
                        task_idx += 1

    # تنفيذ متوازي بنفس المنطق تمامًا
    rows_by_idx = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(_worker_single_experiment, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            idx, row = fut.result()
            rows_by_idx[idx] = row
            if i % 5 == 0:
                print(f"[Run {run_id}] Completed {i}/{len(tasks)} experiments")

    # نحافظ على نفس ترتيب الصفوف كما لو كانت الحلقات متسلسلة (لثبات الملفات)
    rows = [rows_by_idx[i] for i in range(len(tasks))]

    # =============================
    # 5) حفظ النتائج لكل Run: .xlsx + .csv
    # =============================
    out_df = pd.DataFrame(rows, columns=[
        "Dataset", "Drift Speed", "Labeling Strategy", "Imbalance", "Label",
        "Accuracy", "Precision", "Recall", "F1", "Specificity", "GMean",
        "H", "N", "lam", "alpha", "beta", "gamma",
        "stream_seed", "model_seed"
    ])

    xlsx_name = f"Run{run_id}.xlsx"
    csv_name  = f"Run{run_id}.csv"

    # Excel إن توفّرت المكتبة
    try:
        out_df.to_excel(xlsx_name, index=False)
        print(f"[Run {run_id}] Saved: {xlsx_name}")
    except Exception as e:
        print(f"[Run {run_id}] Could not save Excel ({e}). Skipping .xlsx.")

    # CSV دائماً
    out_df.to_csv(csv_name, index=False)
    print(f"[Run {run_id}] Saved: {csv_name}")

# ================================
# 5) نقطة الدخول
# ================================
if __name__ == "__main__":
    # نتوقع تمرير رقم الـ Run من 1..30
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        run_id = int(sys.argv[1])
    else:
        # افتراضياً 1 إذا لم يُمرّر شيء (مفيد للاختبار المحلي)
        run_id = 1

    if not (1 <= run_id <= 30):
        raise SystemExit("Please pass a run id between 1 and 30 (inclusive). Example: python runs30_eval.py 7")

    run_once(run_id)
