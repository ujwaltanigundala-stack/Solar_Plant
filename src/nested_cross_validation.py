"""Nested Cross-Validation for Unbiased Model Evaluation.
Course: Machine Learning (25SC2107E) - Module 5: Evaluation & Validation
Applied to Solar AC_POWER Regularized Regression.
Reference: Session: Nested Cross-Validation

Contrasts non-nested hyperparameter tuning against Nested Cross-Validation:
  1. Inner CV loop selects optimal alpha hyperparameter across candidate grid.
  2. Outer CV loop estimates genuine generalization performance of the tuning procedure.
  3. External hold-out test split acts as unbiased benchmark reference.
  4. Quantifies selection optimism bias across data sample regimes.
  5. Demonstrates hyperparameter instability across outer test folds.
  6. Diagnostic plot exports to Output/plots/.
  7. Diagnostic text report generation to Output/Report/nested_cross_validation_report.txt.
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

NESTED_FEATURES = config.NUMERIC_FEATURES


def run_nested_cv_experiment(n_outer=5, n_inner=3, alphas=None):
    """Execute Nested Cross-Validation vs Non-Nested Grid Search on Solar Data."""
    if alphas is None:
        alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
    train_df = pd.read_csv(config.SPLIT_TRAIN_PATH)
    test_df = pd.read_csv(config.SPLIT_TEST_PATH)
    
    feats = [f for f in NESTED_FEATURES if f in train_df.columns]
    X_tr = train_df[feats].copy()
    y_tr = train_df[config.TARGET_COLUMN].to_numpy()
    
    X_te = test_df[feats].copy()
    y_te = test_df[config.TARGET_COLUMN].to_numpy()
    
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge())
    ])
    param_grid = {"ridge__alpha": alphas}
    
    # 1. Non-Nested Single-Level Grid Search (susceptible to selection bias)
    t0_nn = time.perf_counter()
    inner_cv_single = TimeSeriesSplit(n_splits=n_outer)
    non_nested_gs = GridSearchCV(
        pipe,
        param_grid,
        cv=inner_cv_single,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    non_nested_gs.fit(X_tr, y_tr)
    non_nested_rmse = -float(non_nested_gs.best_score_)
    best_single_alpha = float(non_nested_gs.best_params_["ridge__alpha"])
    time_nn = round(time.perf_counter() - t0_nn, 3)
    
    # Test set evaluation of non-nested model
    pred_test_single = non_nested_gs.predict(X_te)
    test_rmse = float(np.sqrt(mean_squared_error(y_te, pred_test_single)))
    
    # 2. Nested Cross-Validation (Outer loop evaluates tuning procedure)
    t0_ncv = time.perf_counter()
    outer_cv = TimeSeriesSplit(n_splits=n_outer)
    nested_outer_scores = []
    outer_selected_alphas = []
    
    for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_tr)):
        X_tr_fold = X_tr.iloc[tr_idx]
        y_tr_fold = y_tr[tr_idx]
        X_va_fold = X_tr.iloc[va_idx]
        y_va_fold = y_tr[va_idx]
        
        # Inner loop tunes alpha on fold training data
        inner_cv = TimeSeriesSplit(n_splits=n_inner)
        inner_gs = GridSearchCV(
            pipe,
            param_grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        inner_gs.fit(X_tr_fold, y_tr_fold)
        
        # Evaluate chosen inner model on outer unseen validation fold
        pred_outer = inner_gs.predict(X_va_fold)
        outer_rmse = float(np.sqrt(mean_squared_error(y_va_fold, pred_outer)))
        
        nested_outer_scores.append(outer_rmse)
        outer_selected_alphas.append(float(inner_gs.best_params_["ridge__alpha"]))
        
    time_ncv = round(time.perf_counter() - t0_ncv, 3)
    mean_nested_rmse = float(np.mean(nested_outer_scores))
    std_nested_rmse = float(np.std(nested_outer_scores))
    
    # Quantified Selection Optimism Bias
    # Non-nested score is typically optimistically lower than true nested outer error
    optimism_bias = mean_nested_rmse - non_nested_rmse
    
    return {
        "n_outer": n_outer,
        "n_inner": n_inner,
        "non_nested_rmse": round(non_nested_rmse, 3),
        "best_single_alpha": best_single_alpha,
        "test_rmse": round(test_rmse, 3),
        "nested_outer_scores": [round(s, 3) for s in nested_outer_scores],
        "mean_nested_rmse": round(mean_nested_rmse, 3),
        "std_nested_rmse": round(std_nested_rmse, 3),
        "outer_selected_alphas": outer_selected_alphas,
        "optimism_bias": round(optimism_bias, 3),
        "runtime_non_nested": time_nn,
        "runtime_nested": time_ncv
    }


def generate_nested_cv_diagrams(ncv_data):
    """Generate diagnostic plots for Nested Cross-Validation."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plots = {}
    
    # 1. Non-Nested vs Nested Outer vs Test Score Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Non-Nested Best CV\n(Optimistic)", "Nested Outer CV\n(Unbiased Mean)", "Holdout Test Split\n(Ground Truth)"]
    vals = [ncv_data["non_nested_rmse"], ncv_data["mean_nested_rmse"], ncv_data["test_rmse"]]
    errs = [0.0, ncv_data["std_nested_rmse"], 0.0]
    colors = ["#dc2626", "#059669", "#2563eb"]
    
    bars = ax.bar(categories, vals, yerr=errs, capsize=6, color=colors, width=0.45, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.3f} kW", ha="center", fontsize=10, fontweight="bold")
        
    ax.set_title(f"Nested CV vs Non-Nested Optimism Bias (+{ncv_data['optimism_bias']} kW)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation RMSE (kW)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    p1 = config.PLOTS_DIR / "nested_vs_nonnested_cv_scores.png"
    fig.savefig(p1, bbox_inches="tight", dpi=130)
    plt.close(fig)
    plots["scores_comparison"] = str(p1)
    
    # 2. Hyperparameter Instability Across Outer Folds
    fig, ax = plt.subplots(figsize=(8, 4.8))
    folds = [f"Outer Fold {i+1}" for i in range(len(ncv_data["outer_selected_alphas"]))]
    alphas = ncv_data["outer_selected_alphas"]
    
    ax.bar(folds, [np.log10(max(a, 1e-5)) for a in alphas], color="#8b5cf6", width=0.5)
    for i, a in enumerate(alphas):
        ax.text(i, np.log10(max(a, 1e-5)) + 0.1, f"alpha={a}", ha="center", fontsize=9, fontweight="bold")
        
    ax.set_title("Hyperparameter Instability Across Outer Temporal Folds", fontsize=12, fontweight="bold")
    ax.set_xlabel("Outer CV Split", fontsize=11)
    ax.set_ylabel("Log10(Optimal Alpha)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    p2 = config.PLOTS_DIR / "nested_cv_alpha_instability.png"
    fig.savefig(p2, bbox_inches="tight", dpi=130)
    plt.close(fig)
    plots["alpha_instability"] = str(p2)
    
    return plots


def save_nested_cv_report(ncv_data):
    """Save comprehensive nested cross-validation report."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rep_path = config.REPORTS_DIR / "nested_cross_validation_report.txt"
    
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  MODULE 5: NESTED CROSS-VALIDATION UNBIASED EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. EXPERIMENTAL DESIGN:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Outer Folds:              {ncv_data['n_outer']} (TimeSeriesSplit)\n")
        f.write(f"Inner Folds:              {ncv_data['n_inner']} (TimeSeriesSplit)\n")
        f.write("Model Evaluated:          Ridge Regression with L2 Penalty Tuning\n\n")
        
        f.write("2. QUANTIFIED SELECTION OPTIMISM BIAS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Non-Nested Best Score:    {ncv_data['non_nested_rmse']:.3f} kW (Tuned and evaluated on same folds)\n")
        f.write(f"Nested Outer Unbiased:    {ncv_data['mean_nested_rmse']:.3f} +/- {ncv_data['std_nested_rmse']:.3f} kW\n")
        f.write(f"Holdout Test Benchmark:   {ncv_data['test_rmse']:.3f} kW\n")
        f.write(f"Selection Optimism Bias:  +{ncv_data['optimism_bias']:.3f} kW\n\n")
        f.write("Theoretical Insight: Non-nested CV reports an overly optimistic error score because the\n")
        f.write("hyperparameter was chosen specifically to minimize error on the test folds.\n")
        f.write("Nested CV separates selection from evaluation, providing an unbiased error estimate\n")
        f.write("that closely tracks true holdout test performance.\n\n")
        
        f.write("3. HYPERPARAMETER INSTABILITY AUDIT:\n")
        f.write("-" * 50 + "\n")
        for i, (score, alpha) in enumerate(zip(ncv_data["nested_outer_scores"], ncv_data["outer_selected_alphas"])):
            f.write(f"Fold {i+1}: Chosen Alpha = {alpha:<8} | Outer Test RMSE = {score:.3f} kW\n")
            
    return str(rep_path)


def run_nested_cv_pipeline():
    """Execute complete nested CV pipeline."""
    print("  -> Running Nested Cross-Validation (5x3 Time-Series Folds)...")
    ncv_data = run_nested_cv_experiment(n_outer=5, n_inner=3)
    plots = generate_nested_cv_diagrams(ncv_data)
    rep_path = save_nested_cv_report(ncv_data)
    
    print(f"     Non-Nested: {ncv_data['non_nested_rmse']} kW -> Nested: {ncv_data['mean_nested_rmse']} kW (Test: {ncv_data['test_rmse']} kW)")
    print(f"     Selection Optimism Bias: +{ncv_data['optimism_bias']} kW")
    print(f"     Saved Report: {Path(rep_path).name}")
    print(f"     Generated Plots: {[Path(p).name for p in plots.values()]}")
    return ncv_data
