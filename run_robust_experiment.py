"""
Robust SVM Experiment — Poison-Proof SVM via Dual-Task Learning

Compares three scenarios on the PhysioNet gait-dynamics dataset:
  1. Naive SVM trained on CLEAN data              (upper-bound baseline)
  2. Naive SVM trained on 10%-POISONED data       (shows degradation)
  3. Robust SVM trained on 10%-POISONED data      (proposed defense)

Each scenario is repeated 5 times (different random seeds).
Evaluation is ALWAYS on clean test labels.

DET figures written to this directory:
  - det_robust_comparison.png
  - det_poisoned_linear_naive_vs_robust.png
  - det_poisoned_rbf_naive_vs_robust.png
  - det_poisoned_naive_vs_robust_only.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from svm import SVM, RobustSVM
from run_experiment import (
    build_dataset, split_data, standardize,
    compute_metrics, poison_labels, det_curve, plot_det, setup_det_axes,
)

SEEDS = [42, 123, 7, 2024, 999]
POISON_RATIO = 0.10
HERE = os.path.dirname(__file__)


# ────────────────────────────────────────────────────────────────────────────
# Single experiment run
# ────────────────────────────────────────────────────────────────────────────

def run_single(X, y, seed):
    """
    One run: split → standardize → poison training labels only →
    train 6 models (2 kernels × 3 scenarios) → evaluate on clean test set.
    """
    X_tr, X_te, y_tr, y_te = split_data(X, y, test_ratio=0.3, seed=seed)
    X_tr_s, X_te_s = standardize(X_tr, X_te)

    y_tr_poisoned = poison_labels(y_tr, ratio=POISON_RATIO, seed=seed)
    n_flipped = int(np.sum(y_tr != y_tr_poisoned))

    results = {}

    # ── Scenario 1: Naive SVM on clean training data ──────────────
    svm = SVM(C=1.0, kernel="linear", tol=1e-3, max_iter=200)
    svm.fit(X_tr_s, y_tr)
    p = svm.predict(X_te_s)
    s = svm.decision_function(X_te_s)
    results["Naive Linear (clean)"] = (compute_metrics(y_te, p), s, y_te)

    svm = SVM(C=10.0, kernel="rbf", gamma="auto", tol=1e-3, max_iter=200)
    svm.fit(X_tr_s, y_tr)
    p = svm.predict(X_te_s)
    s = svm.decision_function(X_te_s)
    results["Naive RBF (clean)"] = (compute_metrics(y_te, p), s, y_te)

    # ── Scenario 2: Naive SVM on poisoned training data ───────────
    svm = SVM(C=1.0, kernel="linear", tol=1e-3, max_iter=200)
    svm.fit(X_tr_s, y_tr_poisoned)
    p = svm.predict(X_te_s)
    s = svm.decision_function(X_te_s)
    results["Naive Linear (poisoned)"] = (compute_metrics(y_te, p), s, y_te)

    svm = SVM(C=10.0, kernel="rbf", gamma="auto", tol=1e-3, max_iter=200)
    svm.fit(X_tr_s, y_tr_poisoned)
    p = svm.predict(X_te_s)
    s = svm.decision_function(X_te_s)
    results["Naive RBF (poisoned)"] = (compute_metrics(y_te, p), s, y_te)

    # ── Scenario 3: Robust SVM on poisoned training data ──────────
    rsvm = RobustSVM(C=1.0, kernel="linear", tol=1e-3, max_iter=200,
                     n_neighbors=5, n_refine_iters=3, lambda_balance=0.5,
                     trust_floor=0.1)
    rsvm.fit(X_tr_s, y_tr_poisoned)
    p = rsvm.predict(X_te_s)
    s = rsvm.decision_function(X_te_s)
    results["Robust Linear (poisoned)"] = (compute_metrics(y_te, p), s, y_te)

    rsvm = RobustSVM(C=10.0, kernel="rbf", gamma="auto", tol=1e-3, max_iter=200,
                     n_neighbors=5, n_refine_iters=3, lambda_balance=0.5,
                     trust_floor=0.1)
    rsvm.fit(X_tr_s, y_tr_poisoned)
    p = rsvm.predict(X_te_s)
    s = rsvm.decision_function(X_te_s)
    results["Robust RBF (poisoned)"] = (compute_metrics(y_te, p), s, y_te)

    return results, n_flipped


# ────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ────────────────────────────────────────────────────────────────────────────

def print_table(all_results, model_name, title):
    print(f"\n  {title}")
    print(f"  {'─' * 62}")
    print(f"  {'Run':>4s}  {'Accuracy':>10s}  {'Precision':>10s}  {'Recall':>10s}  {'F-Score':>10s}")
    print(f"  {'─' * 62}")
    accs, precs, recs, f1s = [], [], [], []
    for i, run_res in enumerate(all_results):
        acc, prec, rec, f1 = run_res[model_name][0]
        accs.append(acc); precs.append(prec)
        recs.append(rec); f1s.append(f1)
        print(f"  {i+1:>4d}  {acc:>10.4f}  {prec:>10.4f}  {rec:>10.4f}  {f1:>10.4f}")
    print(f"  {'─' * 62}")
    print(f"  {'Mean':>4s}  {np.mean(accs):>10.4f}  {np.mean(precs):>10.4f}"
          f"  {np.mean(recs):>10.4f}  {np.mean(f1s):>10.4f}")
    print(f"  {'Std':>4s}  {np.std(accs):>10.4f}  {np.std(precs):>10.4f}"
          f"  {np.std(recs):>10.4f}  {np.std(f1s):>10.4f}")
    print()


# ────────────────────────────────────────────────────────────────────────────
# DET curve generation
# ────────────────────────────────────────────────────────────────────────────

def _plot_det_single_panel(ax, all_results, models, title):
    """Plot DET curves for a list of (model_key, linestyle, legend_label)."""
    for model_key, ls, label in models:
        scores_all = np.concatenate([r[model_key][1] for r in all_results])
        yt_all = np.concatenate([r[model_key][2] for r in all_results])
        fpr, fnr = det_curve(yt_all, scores_all)
        plot_det(fpr, fnr, label, ax, linestyle=ls)
    setup_det_axes(ax, title)


def make_det_plots(all_results):
    """Generate DET figures: combined 2-panel + one figure per kernel (poisoned focus)."""

    # ── (1) Combined: both kernels, clean + poisoned-naive + poisoned-robust ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    kernel_groups = [
        ("Linear Kernel", [
            ("Naive Linear (clean)",    "-",  "Clean (naive SVM)"),
            ("Naive Linear (poisoned)", "--", "Poisoned train — naive SVM"),
            ("Robust Linear (poisoned)", ":", "Poisoned train — robust SVM"),
        ]),
        ("RBF Kernel", [
            ("Naive RBF (clean)",    "-",  "Clean (naive SVM)"),
            ("Naive RBF (poisoned)", "--", "Poisoned train — naive SVM"),
            ("Robust RBF (poisoned)", ":", "Poisoned train — robust SVM"),
        ]),
    ]

    for ax, (panel_title, models) in zip(axes, kernel_groups):
        _plot_det_single_panel(ax, all_results, models, f"DET — {panel_title}")

    fig.suptitle(
        "DET Curves (5 runs pooled): Clean vs Poisoned Training",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(HERE, "det_robust_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  DET figure saved → {path}")
    plt.close(fig)

    # ── (2) Per-kernel: poisoned train — original (naive) SVM vs robust SVM ──
    #     Optional clean baseline for reference.
    poisoned_linear = [
        ("Naive Linear (clean)", "-", "Reference: clean train (naive SVM)"),
        ("Naive Linear (poisoned)", "--", "Poisoned train — original SVM"),
        ("Robust Linear (poisoned)", "-.", "Poisoned train — robust SVM"),
    ]
    poisoned_rbf = [
        ("Naive RBF (clean)", "-", "Reference: clean train (naive SVM)"),
        ("Naive RBF (poisoned)", "--", "Poisoned train — original SVM"),
        ("Robust RBF (poisoned)", "-.", "Poisoned train — robust SVM"),
    ]

    fig_lin, ax_lin = plt.subplots(figsize=(7, 6))
    _plot_det_single_panel(ax_lin, all_results, poisoned_linear,
                           "DET — Linear kernel (test always clean)")
    fig_lin.suptitle(
        "Linear SVM: poisoned training — original vs robust",
        fontsize=12, y=0.98,
    )
    fig_lin.tight_layout()
    path_lin = os.path.join(HERE, "det_poisoned_linear_naive_vs_robust.png")
    fig_lin.savefig(path_lin, dpi=150, bbox_inches="tight")
    print(f"  DET figure saved → {path_lin}")
    plt.close(fig_lin)

    fig_rbf, ax_rbf = plt.subplots(figsize=(7, 6))
    _plot_det_single_panel(ax_rbf, all_results, poisoned_rbf,
                           "DET — RBF kernel (test always clean)")
    fig_rbf.suptitle(
        "RBF SVM: poisoned training — original vs robust",
        fontsize=12, y=0.98,
    )
    fig_rbf.tight_layout()
    path_rbf = os.path.join(HERE, "det_poisoned_rbf_naive_vs_robust.png")
    fig_rbf.savefig(path_rbf, dpi=150, bbox_inches="tight")
    print(f"  DET figure saved → {path_rbf}")
    plt.close(fig_rbf)

    # ── (3) Poisoned-only comparison (naive vs robust, no clean curve) ──
    fig_narrow, axes_n = plt.subplots(1, 2, figsize=(12, 5.5))
    _plot_det_single_panel(
        axes_n[0], all_results,
        [
            ("Naive Linear (poisoned)", "--", "Original SVM (poisoned train)"),
            ("Robust Linear (poisoned)", "-", "Robust SVM (poisoned train)"),
        ],
        "Linear — poisoned training only",
    )
    _plot_det_single_panel(
        axes_n[1], all_results,
        [
            ("Naive RBF (poisoned)", "--", "Original SVM (poisoned train)"),
            ("Robust RBF (poisoned)", "-", "Robust SVM (poisoned train)"),
        ],
        "RBF — poisoned training only",
    )
    fig_narrow.suptitle(
        "DET: original vs robust SVM (both trained on poisoned labels only)",
        fontsize=12, y=1.02,
    )
    fig_narrow.tight_layout()
    path_narrow = os.path.join(HERE, "det_poisoned_naive_vs_robust_only.png")
    fig_narrow.savefig(path_narrow, dpi=150, bbox_inches="tight")
    print(f"  DET figure saved → {path_narrow}")
    plt.close(fig_narrow)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n[1] Building feature matrix from PhysioNet gait data ...")
    X, y = build_dataset()
    print(f"    Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
    print(f"    Parkinson's (+1): {int(np.sum(y == 1))}  |  "
          f"Control (-1): {int(np.sum(y == -1))}\n")

    # ──────────────────────────────────────────────────────────────────
    # Run 5 experiments
    # ──────────────────────────────────────────────────────────────────
    all_results = []
    for i, seed in enumerate(SEEDS):
        print(f"  Run {i+1}/5 (seed={seed}) ...", end=" ")
        res, nf = run_single(X, y, seed)
        print(f"flipped {nf} training labels")
        all_results.append(res)

    ALL_MODELS = [
        "Naive Linear (clean)", "Naive RBF (clean)",
        "Naive Linear (poisoned)", "Naive RBF (poisoned)",
        "Robust Linear (poisoned)", "Robust RBF (poisoned)",
    ]

    # ──────────────────────────────────────────────────────────────────
    # Per-model 5-run tables
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS TABLES (5 runs each)")
    print("=" * 70)
    for model in ALL_MODELS:
        print_table(all_results, model, model)

    # ──────────────────────────────────────────────────────────────────
    # Summary comparison
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY: Mean Metrics (Accuracy / F-Score)")
    print("=" * 70)
    header = f"  {'Model':<30s} {'Accuracy':>10s} {'F-Score':>10s}"
    print(header)
    print(f"  {'─' * 52}")
    for model in ALL_MODELS:
        acc_mean = np.mean([r[model][0][0] for r in all_results])
        f1_mean = np.mean([r[model][0][3] for r in all_results])
        print(f"  {model:<30s} {acc_mean:>10.4f} {f1_mean:>10.4f}")

    print(f"\n  {'─' * 52}")
    print("  Δ F-Score  (Robust vs Naive, both on poisoned data):")
    for kern in ["Linear", "RBF"]:
        f1_naive = np.mean([r[f"Naive {kern} (poisoned)"][0][3] for r in all_results])
        f1_robust = np.mean([r[f"Robust {kern} (poisoned)"][0][3] for r in all_results])
        print(f"    {kern:8s}  Naive={f1_naive:.4f}  Robust={f1_robust:.4f}"
              f"  Δ={f1_robust - f1_naive:+.4f}")
    print()

    # ──────────────────────────────────────────────────────────────────
    # DET curves
    # ──────────────────────────────────────────────────────────────────
    print("  Generating DET curves ...")
    make_det_plots(all_results)
    print("\n" + "=" * 70)
    print("  Experiment complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
