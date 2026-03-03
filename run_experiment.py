"""
CIS 735 Assignment — Part II: SVM on Gait Dynamics in Neuro-Degenerative Disease

Task overview:
  - Custom SVM (no API): linear (no kernel) and RBF kernel
  - Scikit-learn SVM for comparison
  - 5 runs  →  table of accuracy, precision, recall, F-score
  - DET curve for clean data
  - 10% label-poisoning  →  same table + comparative DET curves

Dataset: https://archive.physionet.org/physiobank/database/gaitndd/
  Binary classification: Parkinson's disease vs. Healthy control
  Features extracted from stride-interval time series (.ts files)
"""

import os
import urllib.request
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from svm import SVM  # our from-scratch implementation

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# 1. Data downloading
# ────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://archive.physionet.org/physiobank/database/gaitndd/"

PARK_IDS = [f"park{i}" for i in range(1, 16)]
CTRL_IDS = [f"control{i}" for i in range(1, 17)]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return path
    url = BASE_URL + filename
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    return path


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    files = [sid + ".ts" for sid in PARK_IDS + CTRL_IDS]
    files.append("subject-description.txt")
    for f in files:
        download_file(f)
    print(f"  All files present in {DATA_DIR}/\n")


# ────────────────────────────────────────────────────────────────────────────
# 2. Feature extraction from .ts files
# ────────────────────────────────────────────────────────────────────────────

def load_ts(filepath):
    """Load a .ts file and return a numpy array (skip bad rows)."""
    rows = []
    with open(filepath) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 13:
                continue
            try:
                vals = [float(v) for v in parts]
                if any(v == 0.0 for v in vals[1:]):
                    continue
                rows.append(vals)
            except ValueError:
                continue
    return np.array(rows)


def extract_features(data):
    """
    From the 13-column time series, extract statistical features
    for columns 2-13 (stride / swing / stance / double-support intervals).
    Features per column: mean, std, coeff of variation, min, max, median,
    IQR, skewness, kurtosis  →  12 columns × 9 stats = 108 features.
    """
    feats = []
    for col in range(1, 13):
        x = data[:, col]
        mu = np.mean(x)
        sigma = np.std(x) + 1e-12
        feats.extend([
            mu,
            sigma,
            sigma / (abs(mu) + 1e-12),
            np.min(x),
            np.max(x),
            np.median(x),
            np.percentile(x, 75) - np.percentile(x, 25),
            float(np.mean(((x - mu) / sigma) ** 3)),
            float(np.mean(((x - mu) / sigma) ** 4) - 3.0),
        ])
    return np.array(feats)


def build_dataset():
    """Load all .ts files and build feature matrix + labels."""
    X_list, y_list = [], []
    for sid in PARK_IDS:
        fp = os.path.join(DATA_DIR, sid + ".ts")
        data = load_ts(fp)
        if data.shape[0] < 10:
            continue
        X_list.append(extract_features(data))
        y_list.append(1)   # Parkinson's = positive class

    for sid in CTRL_IDS:
        fp = os.path.join(DATA_DIR, sid + ".ts")
        data = load_ts(fp)
        if data.shape[0] < 10:
            continue
        X_list.append(extract_features(data))
        y_list.append(-1)  # Control = negative class

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float)
    return X, y


# ────────────────────────────────────────────────────────────────────────────
# 3. Helpers: split, scale, metrics
# ────────────────────────────────────────────────────────────────────────────

def split_data(X, y, test_ratio=0.3, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    split = int(len(y) * (1 - test_ratio))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


def standardize(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-12
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def compute_metrics(y_true, y_pred):
    """Return (accuracy, precision, recall, f1) for positive class = +1."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == -1))
    fn = np.sum((y_pred == -1) & (y_true == 1))
    tn = np.sum((y_pred == -1) & (y_true == -1))

    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1


def poison_labels(y, ratio=0.10, seed=0):
    """Flip `ratio` fraction of labels at random."""
    rng = np.random.RandomState(seed)
    y_poisoned = y.copy()
    n_flip = int(len(y) * ratio)
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y_poisoned[flip_idx] *= -1
    return y_poisoned


# ────────────────────────────────────────────────────────────────────────────
# 4. DET curve computation (from decision scores)
# ────────────────────────────────────────────────────────────────────────────

def det_curve(y_true, scores):
    """
    Compute DET curve: false positive rate vs. false negative rate
    over a range of thresholds.
    """
    thresholds = np.sort(np.unique(scores))[::-1]
    fprs, fnrs = [], []
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == -1)

    for t in thresholds:
        y_pred = np.where(scores >= t, 1, -1)
        fp = np.sum((y_pred == 1) & (y_true == -1))
        fn = np.sum((y_pred == -1) & (y_true == 1))
        fprs.append(fp / n_neg if n_neg > 0 else 0)
        fnrs.append(fn / n_pos if n_pos > 0 else 0)

    return np.array(fprs), np.array(fnrs)


def plot_det(fpr, fnr, label, ax, linestyle="-"):
    """Plot a single DET curve on probit-scaled axes."""
    eps = 1e-4
    fpr_c = np.clip(fpr, eps, 1 - eps)
    fnr_c = np.clip(fnr, eps, 1 - eps)
    ax.plot(norm.ppf(fpr_c), norm.ppf(fnr_c), label=label, linestyle=linestyle)


def setup_det_axes(ax, title="DET Curve"):
    ticks = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
    tick_locs = [norm.ppf(t) for t in ticks]
    tick_labels = [f"{t*100:.0f}%" for t in ticks]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_locs)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ────────────────────────────────────────────────────────────────────────────
# 5. Sklearn SVM wrapper (for comparison)
# ────────────────────────────────────────────────────────────────────────────

def run_sklearn_svm(X_tr, y_tr, X_te, y_te, kernel="rbf"):
    from sklearn.svm import SVC
    clf = SVC(C=1.0, kernel=kernel, gamma="scale")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    scores = clf.decision_function(X_te)
    return y_pred, scores


# ────────────────────────────────────────────────────────────────────────────
# 6. Single experiment run
# ────────────────────────────────────────────────────────────────────────────

def run_single(X, y, seed):
    """
    One run: split → scale → train three SVMs → return metrics & scores.
    Returns dict of {model_name: (metrics_tuple, scores)}.
    """
    X_tr, X_te, y_tr, y_te = split_data(X, y, test_ratio=0.3, seed=seed)
    X_tr_s, X_te_s = standardize(X_tr, X_te)

    results = {}

    # (a) Custom SVM — linear (no kernel)
    svm_lin = SVM(C=1.0, kernel="linear", tol=1e-3, max_iter=200)
    svm_lin.fit(X_tr_s, y_tr)
    pred_lin = svm_lin.predict(X_te_s)
    scores_lin = svm_lin.decision_function(X_te_s)
    results["Custom SVM (linear)"] = (compute_metrics(y_te, pred_lin), scores_lin, y_te)

    # (b) Custom SVM — RBF kernel
    svm_rbf = SVM(C=10.0, kernel="rbf", gamma="auto", tol=1e-3, max_iter=200)
    svm_rbf.fit(X_tr_s, y_tr)
    pred_rbf = svm_rbf.predict(X_te_s)
    scores_rbf = svm_rbf.decision_function(X_te_s)
    results["Custom SVM (RBF)"] = (compute_metrics(y_te, pred_rbf), scores_rbf, y_te)

    # (c) Sklearn SVM (API) — RBF kernel
    pred_sk, scores_sk = run_sklearn_svm(X_tr_s, y_tr, X_te_s, y_te, kernel="rbf")
    results["Sklearn SVM (RBF)"] = (compute_metrics(y_te, pred_sk), scores_sk, y_te)

    return results


# ────────────────────────────────────────────────────────────────────────────
# 7. Print a results table (5 runs)
# ────────────────────────────────────────────────────────────────────────────

def print_table(all_results, model_name, title):
    """Pretty-print a 5-row table for one model."""
    print(f"\n  {title}")
    print(f"  {'─'*62}")
    print(f"  {'Run':>4s}  {'Accuracy':>10s}  {'Precision':>10s}  {'Recall':>10s}  {'F-Score':>10s}")
    print(f"  {'─'*62}")
    accs, precs, recs, f1s = [], [], [], []
    for i, run_res in enumerate(all_results):
        acc, prec, rec, f1 = run_res[model_name][0]
        accs.append(acc); precs.append(prec)
        recs.append(rec); f1s.append(f1)
        print(f"  {i+1:>4d}  {acc:>10.4f}  {prec:>10.4f}  {rec:>10.4f}  {f1:>10.4f}")
    print(f"  {'─'*62}")
    print(f"  {'Mean':>4s}  {np.mean(accs):>10.4f}  {np.mean(precs):>10.4f}"
          f"  {np.mean(recs):>10.4f}  {np.mean(f1s):>10.4f}")
    print(f"  {'Std':>4s}  {np.std(accs):>10.4f}  {np.std(precs):>10.4f}"
          f"  {np.std(recs):>10.4f}  {np.std(f1s):>10.4f}")
    print()


# ════════════════════════════════════════════════════════════════════════════
# 8. Main Experiment
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CIS 735 — SVM Experiment on Gait Dynamics (Parkinson's vs Control)")
    print("=" * 70)

    # ── Download & build dataset ──
    print("\n[1] Downloading dataset from PhysioNet ...")
    download_dataset()

    print("[2] Building feature matrix ...")
    X, y = build_dataset()
    print(f"    Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
    print(f"    Parkinson's (+1): {np.sum(y == 1)}  |  Control (-1): {np.sum(y == -1)}\n")

    seeds = [42, 123, 7, 2024, 999]

    # ──────────────────────────────────────────────────────────────────
    # Part (1): 5 runs on clean data
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  PART (1): 5 Runs on Clean Data")
    print("=" * 70)

    clean_results = []
    for i, s in enumerate(seeds):
        print(f"  Running iteration {i+1}/5 (seed={s}) ...")
        clean_results.append(run_single(X, y, seed=s))

    for model in ["Custom SVM (linear)", "Custom SVM (RBF)", "Sklearn SVM (RBF)"]:
        print_table(clean_results, model, model)

    # DET curve for clean data (using all 5 runs pooled)
    fig, ax = plt.subplots(figsize=(7, 7))
    for model, ls in [("Custom SVM (linear)", "-"),
                      ("Custom SVM (RBF)", "--"),
                      ("Sklearn SVM (RBF)", ":")]:
        all_scores = np.concatenate([r[model][1] for r in clean_results])
        all_y = np.concatenate([r[model][2] for r in clean_results])
        fpr, fnr = det_curve(all_y, all_scores)
        plot_det(fpr, fnr, model, ax, linestyle=ls)
    setup_det_axes(ax, "DET Curve — Clean Data (5 Runs Pooled)")
    fig.tight_layout()
    det_clean_path = os.path.join(os.path.dirname(__file__), "det_clean.png")
    fig.savefig(det_clean_path, dpi=150)
    print(f"  DET curve (clean) saved → {det_clean_path}\n")
    plt.close(fig)

    # ──────────────────────────────────────────────────────────────────
    # Part (2): 10% label poisoning
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  PART (2): 10% Label Poisoning (Flipping Class Labels)")
    print("=" * 70)

    poisoned_results = []
    for i, s in enumerate(seeds):
        print(f"  Running iteration {i+1}/5 (seed={s}) ...")
        y_poisoned = poison_labels(y, ratio=0.10, seed=s)
        poisoned_results.append(run_single(X, y_poisoned, seed=s))

    for model in ["Custom SVM (linear)", "Custom SVM (RBF)", "Sklearn SVM (RBF)"]:
        print_table(poisoned_results, model, f"{model} [POISONED 10%]")

    # Comparative DET curves: clean vs poisoned
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (model, ls) in enumerate([("Custom SVM (linear)", "-"),
                                        ("Custom SVM (RBF)", "--"),
                                        ("Sklearn SVM (RBF)", ":")]):
        ax = axes[idx]

        # Clean
        sc_clean = np.concatenate([r[model][1] for r in clean_results])
        yt_clean = np.concatenate([r[model][2] for r in clean_results])
        fpr_c, fnr_c = det_curve(yt_clean, sc_clean)
        plot_det(fpr_c, fnr_c, "Clean", ax, linestyle="-")

        # Poisoned
        sc_poison = np.concatenate([r[model][1] for r in poisoned_results])
        yt_poison = np.concatenate([r[model][2] for r in poisoned_results])
        fpr_p, fnr_p = det_curve(yt_poison, sc_poison)
        plot_det(fpr_p, fnr_p, "Poisoned (10%)", ax, linestyle="--")

        setup_det_axes(ax, model)

    fig.suptitle("DET Curves — Clean vs 10% Poisoned", fontsize=14, y=1.02)
    fig.tight_layout()
    det_comp_path = os.path.join(os.path.dirname(__file__), "det_comparison.png")
    fig.savefig(det_comp_path, dpi=150, bbox_inches="tight")
    print(f"  DET comparison saved → {det_comp_path}\n")
    plt.close(fig)

    # ──────────────────────────────────────────────────────────────────
    # Summary comparison
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY: Mean F-Score Comparison")
    print("=" * 70)
    print(f"  {'Model':<25s} {'Clean':>10s} {'Poisoned':>10s} {'Δ':>10s}")
    print(f"  {'─'*58}")
    for model in ["Custom SVM (linear)", "Custom SVM (RBF)", "Sklearn SVM (RBF)"]:
        f1_clean = np.mean([r[model][0][3] for r in clean_results])
        f1_poison = np.mean([r[model][0][3] for r in poisoned_results])
        print(f"  {model:<25s} {f1_clean:>10.4f} {f1_poison:>10.4f} {f1_poison-f1_clean:>+10.4f}")
    print()
    print("  Experiment complete. Check det_clean.png and det_comparison.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
