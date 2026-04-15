# data_generation_osnn.py
# Final paper-exact synthetic stream generator for OSNN experiments
# - Concept length = 4000
# - Drift types: 'abrupt' (instant) or 'gradual' (400-sample window, instance-level interleaving)
# - Per-concept optional imbalance (minority = class 1) WITHOUT breaking drift windows
# - Labeling: uniform (global), nonuniform (per-class-per-concept, KMeans=5)
# - Dims (OSSN ): SINE=4, SEA=3, STAGGER=7, AGRAWAL=36
# - Exact concept sequences and total instances per dataset as in paper

import numpy as np
from sklearn.cluster import KMeans

# -----------------------------
# Paper constants / meta-config
# -----------------------------
CONCEPT_LEN = 4000
GRADUAL_WINDOW = 400  # only for 'gradual'

# Dataset → (sequence of rules, total instances, target_dims)
PAPER_STREAMS = {
    # ----- SINE -----
    "SINE1":    (["r3", "r4", "r3"],                              12000, 4),
    "SINE2":    (["r1", "r2", "r3", "r4", "r1"],                  20000, 4),

    # ----- AGRAWAL -----
    "AGRAWAL1": (["r1", "r3", "r4", "r7", "r10"],                 20000, 36),
    "AGRAWAL2": (["r7", "r4", "r6", "r5", "r2", "r9"],            24000, 36),
    "AGRAWAL3": (["r4", "r2", "r1", "r3", "r4"],                  20000, 36),
    "AGRAWAL4": (["r1", "r3", "r6", "r5", "r4"],                  20000, 36),  # <<< UPDATED for paper match

    # ----- SEA -----
    "SEA1":     (["r4", "r3", "r1", "r2", "r4"],                  20000, 3),
    "SEA2":     (["r4", "r1", "r4", "r3", "r2"],                  20000, 3),

    # ----- STAGGER -----
    "STAGGER1": (["r1", "r2", "r3", "r2"],                        16000, 7),
    "STAGGER2": (["r2", "r3", "r1", "r2"],                        16000, 7),   # <<< UPDATED for paper match
}
# --------------------------
# Utility: RNG & normalization
# --------------------------
def _rng(seed): return np.random.default_rng(seed)

def _normalize_to_pm1(X):
    X = X.astype(float)
    minv = X.min(axis=0)
    maxv = X.max(axis=0)
    span = np.where(maxv > minv, maxv - minv, 1.0)
    X01 = (X - minv) / span
    return 2.0 * X01 - 1.0

# --------------------------
# Labeling helpers
# --------------------------
"""
def _label_uniform(X, y, label_rate, seed):
    n = len(y)
    flag = np.full(n, -1.0)
    n_lab = int(round(label_rate * n))
    if n_lab <= 0: return flag
    rg = _rng(seed)
    idx = rg.choice(n, n_lab, replace=False)
    flag[idx] = y[idx]
    return flag
 """   
"""  
#this version used for sensetivty test in matlab after the test of the beginning of the stream
def _label_uniform_v2(X, y, label_rate, seed):
    
    #Uniform labeling distributed sequentially over the stream (no shuffling).
    
    n = len(y)
    flags = np.full(n, -1.0)
    num_labeled = int(round(label_rate * n))
    if num_labeled <= 0:
        return flags

    step = max(1, n // num_labeled)
    idx = np.arange(0, step * num_labeled, step)
    idx = idx[:num_labeled]
    flags[idx] = y[idx]
    return flags
 """  

def _label_uniform_v3(X, y, concept_ids, n_concepts, label_rate, seed):
    """
    Uniform labeling per concept segment (time-distributed step-based).
    Matches V2 logic but applies labeling separately within each concept.
    """
    n = len(y)
    flags = np.full(n, -1.0)

    for c in range(n_concepts):
        idx_c = np.where(concept_ids == c)[0]
        n_c = len(idx_c)
        if n_c == 0:
            continue

        num_labeled_c = int(round(label_rate * n_c))
        if num_labeled_c <= 0:
            continue

        step = max(1, n_c // num_labeled_c)
        idx_local = np.arange(0, step * num_labeled_c, step)
        idx_local = idx_local[:num_labeled_c]
        idx_global = idx_c[idx_local]

        flags[idx_global] = y[idx_global]

    return flags



"""
def _label_per_concept_nonuniform(X, y, concept_ids, n_concepts, label_rate, seed):
    # KMeans(k=5) per (concept, class)
    n = len(y)
    flag = np.full(n, -1.0)
    rg = _rng(seed)
    for c in range(n_concepts):
        idx_c = np.where(concept_ids == c)[0]
        if len(idx_c) == 0: continue
        n_lab_c = int(round(label_rate * len(idx_c)))
        if n_lab_c <= 0: continue

        chosen = []
        for cls in [0, 1]:
            idx_cc = idx_c[y[idx_c] == cls]
            if len(idx_cc) == 0: continue
            X_cc = X[idx_cc]
            k = min(5, len(idx_cc))
            if k > 1:
                km = KMeans(n_clusters=k, random_state=seed, n_init=10)
                km.fit(X_cc)
                centers = km.cluster_centers_
                for cen in centers:
                    local = np.argmin(np.linalg.norm(X_cc - cen, axis=1))
                    chosen.append(idx_cc[local])
            else:
                chosen.append(idx_cc[0])

        chosen = np.unique(np.array(chosen, dtype=int))
        need = max(0, n_lab_c - len(chosen))
        if need > 0:
            remain = np.setdiff1d(idx_c, chosen, assume_unique=False)
            extra = rg.choice(remain, min(need, len(remain)), replace=False) if len(remain) > 0 else np.array([], int)
            chosen = np.concatenate([chosen, extra])
        flag[chosen] = y[chosen]
    return flag
"""
def _label_per_concept_nonuniform_v2(X, y, concept_ids, n_concepts, label_rate, seed):
    """
    Non-uniform labeling per (concept, class) using KMeans with proportional quota per cluster.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    flags = np.full(n, -1.0)
    k = 5  # Number of clusters per class as in OSSN paper

    for c in range(n_concepts):
        idx_c = np.where(concept_ids == c)[0]
        if len(idx_c) == 0:
            continue

        for cls in [0, 1]:
            idx_cls_in_concept = idx_c[y[idx_c] == cls]
            n_cls = len(idx_cls_in_concept)
            if n_cls == 0:
                continue

            num_labels = round(label_rate * n_cls)
            if num_labels == 0:
                continue

            X_cls = X[idx_cls_in_concept]
            k_actual = min(k, n_cls)

            if k_actual > 1:
                km = KMeans(n_clusters=k_actual, n_init=5, max_iter=300, random_state=seed)
                cluster_labels = km.fit_predict(X_cls)
            else:
                cluster_labels = np.zeros(n_cls, dtype=int)

            counts_per_cluster = np.array([np.sum(cluster_labels == cnum) for cnum in range(k_actual)])
            total_clustered = counts_per_cluster.sum()

            label_quota = np.round(num_labels * (counts_per_cluster / total_clustered)).astype(int)
            diff_labels = num_labels - label_quota.sum()

            if diff_labels > 0:
                sort_idx = np.argsort(-counts_per_cluster)  # descending
                for i in range(diff_labels):
                    label_quota[sort_idx[i % k_actual]] += 1
            elif diff_labels < 0:
                sort_idx = np.argsort(label_quota)  # ascending
                i = 0
                while diff_labels < 0 and i < k_actual:
                    if label_quota[sort_idx[i]] > 0:
                        label_quota[sort_idx[i]] -= 1
                        diff_labels += 1
                    i += 1

            selected = []
            for cnum in range(k_actual):
                cluster_idx = np.where(cluster_labels == cnum)[0]
                n_select = min(label_quota[cnum], len(cluster_idx))
                if n_select > 0:
                    sel = rng.choice(cluster_idx, size=n_select, replace=False)
                    selected.extend(sel)

            if selected:
                global_idx = idx_cls_in_concept[np.array(selected)]
                flags[global_idx] = y[global_idx]

    return flags

# --------------------------
# Imbalance per concept (optional)
# NOTE: The old imbalance version (below) is deprecated and kept only for reference.

# --------------------------
def _apply_ratio_keep_n(X, y, minority_ratio, seed):
    """Enforce class-1 proportion per concept while keeping total n fixed."""
    if minority_ratio is None:  # no imbalance (paper-default)
        return X, y
    rg = _rng(seed)
    n = len(y)
    target_min = int(round(minority_ratio * n))
    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]

    if len(idx1) > target_min:
        keep1 = rg.choice(idx1, target_min, replace=False)
        need0 = n - target_min
        if len(idx0) >= need0:
            keep0 = rg.choice(idx0, need0, replace=False)
        else:
            keep0 = np.concatenate([idx0, rg.choice(idx0, need0 - len(idx0), replace=True)])
    else:
        keep1 = idx1.copy()
        need1 = target_min - len(keep1)
        if need1 > 0 and len(idx1) > 0:
            keep1 = np.concatenate([keep1, rg.choice(idx1, need1, replace=True)])
        elif need1 > 0 and len(idx1) == 0:
            add = rg.choice(np.arange(n), need1, replace=False)
            keep1 = add
        need0 = n - len(keep1)
        if len(idx0) >= need0:
            keep0 = rg.choice(idx0, need0, replace=False)
        else:
            keep0 = np.concatenate([idx0, rg.choice(idx0, need0 - len(idx0), replace=True)])

    keep = np.concatenate([keep0, keep1])
    keep.sort()
    return X[keep], y[keep]

# --------------------------
# SINE (4D): 2 informative + 2 noise
# --------------------------
SINE_PHASES = {"r1": 0.0, "r2": np.pi/2, "r3": np.pi/6, "r4": np.pi/3}
def _sine_sample(rg, n):
    X2 = rg.random((n, 2))
    noise = rg.normal(0, 1, (n, 2))
    return np.concatenate([X2, noise], axis=1)

def _sine_rule(rule, X4):
    a = SINE_PHASES[rule]
    return (np.sin(2*np.pi*X4[:,0] + a) + np.sin(2*np.pi*X4[:,1]) > 0).astype(int)

# --------------------------
# SEA (3D): 2 informative + 1 noise
# --------------------------
SEA_THETA = {"r1": 0.7, "r2": 0.8, "r3": 0.75, "r4": 0.65}
def _sea_sample(rg, n): return rg.random((n, 3))
def _sea_rule(rule, X3): return (X3[:,0] + X3[:,1] > SEA_THETA[rule]).astype(int)

# --------------------------
# STAGGER (7D): size numeric (1) + color one-hot(3) + shape one-hot(3)
# --------------------------
def _stagger_cat_sample(rg, n):
    size  = rg.integers(0,3,n)
    color = rg.integers(0,3,n)
    shape = rg.integers(0,3,n)
    return np.column_stack([size, color, shape])

def _stagger_expand_to7(X_cat):
    size = X_cat[:,0].astype(float).reshape(-1,1)  # numeric channel
    color = X_cat[:,1].astype(int)
    shape = X_cat[:,2].astype(int)
    C = np.stack([(color==0),(color==1),(color==2)], axis=1).astype(float)
    S = np.stack([(shape==0),(shape==1),(shape==2)], axis=1).astype(float)
    return np.concatenate([size, C, S], axis=1)  # 1 + 3 + 3 = 7

def _stagger_rule(rule, X_cat):
    size, color, shape = X_cat[:,0].astype(int), X_cat[:,1].astype(int), X_cat[:,2].astype(int)
    if rule == "r1": return ((size==2) & (color==0)).astype(int)
    if rule == "r2": return ((color==1) | (shape==1)).astype(int)
    if rule == "r3": return ((size==0) & (shape==2)).astype(int)
    return ((size==2) | (shape==2)).astype(int)

# --------------------------
# AGRAWAL (36D): 6 numeric + car one-hot(20) + zipcode one-hot(10)
# Rules r1..r10 in a standard style used widely for Agrawal generator
# --------------------------
def _agrawal_base_sample(rg, n):
    salary     = rg.uniform(20000,150000,n)
    commission = rg.uniform(0,50000,n)
    age        = rg.uniform(18,80,n)
    education  = rg.integers(6,17,n)           # 6..16
    car        = rg.integers(1,21,n)           # 1..20
    zipcode    = rg.integers(1,11,n)           # 1..10
    house_val  = rg.uniform(50000,1_000_000,n)
    loan       = rg.uniform(0,500000,n)
    # Keep as separate arrays for rule, then expand below
    return np.column_stack([salary, commission, age, education, car, zipcode, house_val, loan])

def _agrawal_expand36(X8):
    salary, commission, age, education, car, zipcode, house_val, loan = \
        X8[:,0], X8[:,1], X8[:,2], X8[:,3], X8[:,4].astype(int), X8[:,5].astype(int), X8[:,6], X8[:,7]
    # numeric: salary, commission, age, education, house_val, loan  => 6
    num = np.column_stack([salary, commission, age, education, house_val, loan])
    # one-hot car (1..20)
    C = np.zeros((len(car), 20), float)
    C[np.arange(len(car)), np.clip(car-1, 0, 19)] = 1.0
    # one-hot zipcode (1..10)
    Z = np.zeros((len(zipcode), 10), float)
    Z[np.arange(len(zipcode)), np.clip(zipcode-1, 0, 9)] = 1.0
    return np.concatenate([num, C, Z], axis=1)  # 6 + 20 + 10 = 36

def _agrawal_rule(rule, X8):
    salary, commission, age, education, car, zipcode, house_value, loan = \
        X8[:,0], X8[:,1], X8[:,2], X8[:,3], X8[:,4], X8[:,5], X8[:,6], X8[:,7]
    # standard Agrawal functions (MOA version)
    if rule == "r1":
        y = ((salary >= 50000) | (commission >= 25000)).astype(int)
    elif rule == "r2":
        y = ((age < 40) & (salary < 75000)).astype(int)
    elif rule == "r3":
        y = ((education >= 12) | (house_value > 300000)).astype(int)
    elif rule == "r4":
        y = ((loan < 200000) & (salary > 80000)).astype(int)
    elif rule == "r5":
        y = ((commission > 20000) | (house_value > 400000)).astype(int)
    elif rule == "r6":
        y = ((salary < 50000) & (loan > 100000)).astype(int)
    elif rule == "r7":
        y = ((education < 10) | (age < 30)).astype(int)
    elif rule == "r8":
        y = (((salary + commission) > 100000) & (loan < 200000)).astype(int)
    elif rule == "r9":
        y = ((house_value / (loan + 1.0)) > 2.0).astype(int)
    elif rule == "r10":
        y = ((salary > 60000) & (age > 40) & (education > 12)).astype(int)
    else:
        y = (salary > 75000).astype(int)
    return y


# --------------------------
# Core generator per dataset (one concept)
# --------------------------
"""
#note This version create imbalance after generation by downsamping,oversamping
#The old imbalance version (below) is deprecated and kept only for reference.


def _gen_concept_block(stream_name, rule, imb_ratio, seed):
    rg = _rng(seed)
    if stream_name.startswith("SINE"):
        X = _sine_sample(rg, CONCEPT_LEN)
        y = _sine_rule(rule, X)
        X, y = _apply_ratio_keep_n(X, y, imb_ratio, seed+1000)

    elif stream_name.startswith("SEA"):
        X = _sea_sample(rg, CONCEPT_LEN)
        y = _sea_rule(rule, X)
        X, y = _apply_ratio_keep_n(X, y, imb_ratio, seed+1000)

    elif stream_name.startswith("STAGGER"):
        Xcat = _stagger_cat_sample(rg, CONCEPT_LEN)
        y = _stagger_rule(rule, Xcat)
        # imbalance on the concept
        if imb_ratio is not None:
            Xcat, y = _apply_ratio_keep_n(Xcat, y, imb_ratio, seed+1000)
        X = _stagger_expand_to7(Xcat)

    elif stream_name.startswith("AGRAWAL"):
        X8 = _agrawal_base_sample(rg, CONCEPT_LEN)
        y = _agrawal_rule(rule, X8)
        if imb_ratio is not None:
            X8, y = _apply_ratio_keep_n(X8, y, imb_ratio, seed+1000)
        X = _agrawal_expand36(X8)

    else:
        raise ValueError("Unknown stream_name.")
    return X, y
"""

# The new version (active) generates the imbalance directly during data generation.
# NOTE: Per-concept shuffling is applied because the new imbalance generation method
# creates class-specific sample blocks sequentially (0 then 1),
# unlike the original OSSN which mixed classes naturally before imbalance.
# Shuffling restores realistic temporal class distribution without altering rules or concepts.


def _gen_concept_block(stream_name, rule, imb_ratio, seed):
    """
    توليد عينات مفهوم واحد مع تطبيق نسبة عدم توازن أثناء التوليد نفسه
    بدون Downsampling/Oversampling بعد التوليد.
    """
    rg = _rng(seed)
    total_n = CONCEPT_LEN

    # ---------------------------
    # 1. تحديد عدد العينات من كل فئة
    # ---------------------------
    if imb_ratio is None:
        n_min = total_n // 2
    else:
        n_min = int(round(imb_ratio * total_n))
    n_maj = total_n - n_min

    # ---------------------------
    # 2. دالة مساعدة للتوليد المتكرر لضمان العدد المطلوب
    # ---------------------------
    def _sample_until_enough(X_fn, y_fn, rule, n_target, cls):
        collected = []
        max_attempts = 10000   # 👈 حد أقصى للمحاولات
        attempt = 0

        while len(collected) < n_target and attempt < max_attempts:
           X_batch = X_fn(rg, n_target * 2)
           y_batch = y_fn(rule, X_batch)
           X_selected = X_batch[y_batch == cls]
           collected.extend(X_selected)
           attempt += 1

        if len(collected) < n_target:
             raise RuntimeError(
               f"لم يتم العثور على العدد المطلوب من عينات الفئة {cls} "
               f"بعد {attempt} محاولة — ربما القاعدة rule={rule} لا تولّد هذه الفئة كفاية.")

        return np.array(collected[:n_target])


    # ---------------------------
    # 3. التوليد حسب نوع الداتاست
    # ---------------------------
    if stream_name.startswith("SINE"):
        X1 = _sample_until_enough(_sine_sample, _sine_rule, rule, n_min, 1)
        X0 = _sample_until_enough(_sine_sample, _sine_rule, rule, n_maj, 0)
        y1 = np.ones(len(X1), dtype=int)
        y0 = np.zeros(len(X0), dtype=int)
        X = np.vstack([X0, X1])
        y = np.concatenate([y0, y1])

         # Shuffling here not affecting (concepts , rules for all DS)
        idx = np.random.permutation(len(y))
        X = X[idx]
        y = y[idx]

        return X, y

    elif stream_name.startswith("SEA"):
        X1 = _sample_until_enough(_sea_sample, _sea_rule, rule, n_min, 1)
        X0 = _sample_until_enough(_sea_sample, _sea_rule, rule, n_maj, 0)
        y1 = np.ones(len(X1), dtype=int)
        y0 = np.zeros(len(X0), dtype=int)
        X = np.vstack([X0, X1])
        y = np.concatenate([y0, y1])

               # Shuffling here not affecting (concepts , rules for all DS)
        idx = np.random.permutation(len(y))
        X = X[idx]
        y = y[idx]

        return X, y

    elif stream_name.startswith("STAGGER"):
        # class 1
        collected1 = []
        while len(collected1) < n_min:
            Xcat = _stagger_cat_sample(rg, n_min * 2)
            ytmp = _stagger_rule(rule, Xcat)
            collected1.extend(Xcat[ytmp == 1])
            if len(collected1) == 0 and n_min > 0:
                raise RuntimeError("لم يتم العثور على عينات فئة 1 لـ STAGGER")
        Xcat1 = np.array(collected1[:n_min])

        # class 0
        collected0 = []
        while len(collected0) < n_maj:
            Xcat = _stagger_cat_sample(rg, n_maj * 2)
            ytmp = _stagger_rule(rule, Xcat)
            collected0.extend(Xcat[ytmp == 0])
        Xcat0 = np.array(collected0[:n_maj])

        X = _stagger_expand_to7(np.vstack([Xcat0, Xcat1]))
        y = np.concatenate([np.zeros(len(Xcat0), int), np.ones(len(Xcat1), int)])

               # Shuffling here not affecting (concepts , rules for all DS)
        idx = np.random.permutation(len(y))
        X = X[idx]
        y = y[idx]
        return X, y

    elif stream_name.startswith("AGRAWAL"):
        collected1 = []
        while len(collected1) < n_min:
            X8 = _agrawal_base_sample(rg, n_min * 2)
            ytmp = _agrawal_rule(rule, X8)
            collected1.extend(X8[ytmp == 1])
            if len(collected1) == 0 and n_min > 0:
                raise RuntimeError("لم يتم العثور على عينات فئة 1 لـ AGRAWAL")
        X8_1 = np.array(collected1[:n_min])

        collected0 = []
        while len(collected0) < n_maj:
            X8 = _agrawal_base_sample(rg, n_maj * 2)
            ytmp = _agrawal_rule(rule, X8)
            collected0.extend(X8[ytmp == 0])
        X8_0 = np.array(collected0[:n_maj])

        X = _agrawal_expand36(np.vstack([X8_0, X8_1]))
        y = np.concatenate([np.zeros(len(X8_0), int), np.ones(len(X8_1), int)])

               # Shuffling here not affecting (concepts , rules for all DS)
        idx = np.random.permutation(len(y))
        X = X[idx]
        y = y[idx]
        return X, y
        

    else:
        raise ValueError(f"Unknown stream_name: {stream_name}")


# --------------------------
# Gradual interleaving (instance-level, 400 window)
# --------------------------
def _apply_gradual_interleaving(concepts_X, concepts_y, stream_name):
    """Modify the boundary regions between consecutive concepts to interleave samples over a 400-sample window.
       Keeps total length unchanged for each concept (4000) and globally."""
    half = GRADUAL_WINDOW // 2  # 200
    n_concepts = len(concepts_X)
    for i in range(n_concepts-1):
        X_prev, y_prev = concepts_X[i], concepts_y[i]
        X_next, y_next = concepts_X[i+1], concepts_y[i+1]

        # Take tails/heads
        tail_X = X_prev[-half:].copy()
        tail_y = y_prev[-half:].copy()
        head_X = X_next[:half].copy()
        head_y = y_next[:half].copy()

        # Build interleaved window of length 400 with p(next) increasing 0→1
        L = GRADUAL_WINDOW
        t = np.linspace(0.0, 1.0, L)
        inter_X = []
        inter_y = []
        p_next_idx = 0
        p_prev_idx = 0

        # Make deques by simple indices
        tail_i = 0
        head_i = 0

        for k in range(L):
            p_next = t[k]
            take_next = (k / (L-1)) >= (1.0 - p_next)  # monotonic; equivalent to linearly increasing chance
            if take_next and head_i < half:
                inter_X.append(head_X[head_i]); inter_y.append(head_y[head_i]); head_i += 1
            elif tail_i < half:
                inter_X.append(tail_X[tail_i]); inter_y.append(tail_y[tail_i]); tail_i += 1
            elif head_i < half:
                inter_X.append(head_X[head_i]); inter_y.append(head_y[head_i]); head_i += 1

        inter_X = np.asarray(inter_X)
        inter_y = np.asarray(inter_y)

        # Put first 200 back to end of prev, next 200 to start of next
        concepts_X[i][-half:]   = inter_X[:half]
        concepts_y[i][-half:]   = inter_y[:half]
        concepts_X[i+1][:half]  = inter_X[half:]
        concepts_y[i+1][:half]  = inter_y[half:]

# --------------------------
# Public API
# --------------------------
def generate_stream(stream_type='SINE1',
                    n_samples=None,
                    drift_type='abrupt',            # 'abrupt' or 'gradual'
                    imbalance_ratio=None,           # None = paper default (no imposed imbalance); else e.g., 0.01, 0.10
                    label_rate=0.10,
                    label_strategy='uniform',       # 'uniform' or 'nonuniform'
                    seed=42):
    st = stream_type.upper()
    if st not in PAPER_STREAMS:
        raise ValueError(f"stream_type must be one of: {list(PAPER_STREAMS.keys())}")
    seq, total, target_dims = PAPER_STREAMS[st]
    if n_samples is None:
        n_samples = total
    #if n_samples != total:
        #raise ValueError(f"{st}: n_samples must be {total} to match the paper exactly.")

    n_concepts = len(seq)
    concepts_X, concepts_y = [], []
    for i, r in enumerate(seq):
        Xc, yc = _gen_concept_block(st, r, imbalance_ratio, seed + i)
        concepts_X.append(Xc)
        concepts_y.append(yc)

    # gradual interleaving if needed
    if drift_type.lower() == 'gradual':
        _apply_gradual_interleaving(concepts_X, concepts_y, st)

    # stack
    X = np.vstack(concepts_X)
    y = np.concatenate(concepts_y)

    # normalize full stream to [-1,1] (after building all concepts and interleaving)
    X = _normalize_to_pm1(X)

    # pack D = [features..., flag, label]
    D = np.hstack([X, np.zeros((len(y),1)), y.reshape(-1,1)])

    # labeling
    concept_ids = np.concatenate([np.full(CONCEPT_LEN, i, dtype=int) for i in range(n_concepts)])
    if label_strategy == 'uniform':
        # flags = _label_uniform(X, y, label_rate, seed)
        # flags = _label_uniform_v2(X, y, label_rate, seed)
        flags = _label_uniform_v3(X, y, concept_ids, n_concepts, label_rate, seed)



    elif label_strategy == 'nonuniform':
        # flags = _label_per_concept_nonuniform(X, y, concept_ids, n_concepts, label_rate, seed)
        flags = _label_per_concept_nonuniform_v2(X, y, concept_ids, n_concepts, label_rate, seed)
    else:
        raise ValueError("label_strategy must be 'uniform' or 'nonuniform'")
    D[:, -2] = flags

    # concept sequence names
    concept_seq = [f"{st}_{r}_c{i+1}" for i, r in enumerate(seq)]
    return D, concept_seq

# --------------------------
# Quick self-test (optional)
# --------------------------
if __name__ == "__main__":
    tests = [
        ("SINE1","abrupt"), ("SINE2","gradual"),
        ("AGRAWAL1","abrupt"), ("AGRAWAL2","gradual"),
        ("AGRAWAL3","gradual"), ("AGRAWAL4","abrupt"),
        ("SEA1","gradual"), ("SEA2","abrupt"),
        ("STAGGER1","gradual"), ("STAGGER2","abrupt"),
    ]  # <<< UPDATED for paper match

    for st, dr in tests:
        D, seq = generate_stream(stream_type=st, drift_type=dr,
                                 imbalance_ratio=0.10,    # keep or set None
                                 label_rate=0.10, label_strategy='nonuniform', seed=42)
        y = D[:,-1].astype(int)
        print(f"{st:9s}  drift={dr:7s}  N={len(D)}  dims={D.shape[1]-2}  concepts={len(seq)}  "
              f"class1={np.sum(y==1)} class0={np.sum(y==0)}  labeled={np.sum(D[:,-2]!=-1)}")
