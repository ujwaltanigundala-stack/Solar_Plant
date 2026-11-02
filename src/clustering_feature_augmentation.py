"""Combining Clustering With a Predictive Model (Feature Augmentation).
Course: Machine Learning (25SC2107E) - Module 4: Unsupervised Learning
Applied to Solar AC_POWER Supervised Regression.

Topics covered:
1. Continuous feature space clustering using K-Means on training split.
2. Feature augmentation variants:
   - Baseline: Original 10 numeric features
   - Variant A: Base features + One-Hot cluster membership
   - Variant B: Base features + Continuous Euclidean centroid distances
   - Variant C: Full composite (Base + One-Hot + Centroid distances)
3. Rigorous Leakage Prevention:
   - K-Means & Scaler fitted strictly on Plant 1 Train.
   - Validation, Test, and Unseen Plant 2 transformed via predict() and transform().
4. Downstream Supervised Regression Models:
   - OLS Linear Regression
   - Ridge Regularized Regression (L2 penalty stabilizes collinear distance features)
5. Statistical Significance Verification via Paired Bootstrap:
   - 500 bootstrap resamples on Test set computing 95% Confidence Intervals for Delta RMSE.
6. Diagnostic visualization exports to Output/plots/.
7. Diagnostic text report generation to Output/Report/cluster_feature_augmentation_report.txt.
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
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import config

BASE_FEATURES = config.NUMERIC_FEATURES


def load_and_scale_splits():
    """Load splits and prepare leakage-free scaled base features."""
    tr_df = pd.read_csv(config.SPLIT_TRAIN_PATH)
    va_df = pd.read_csv(config.SPLIT_VAL_PATH)
    te_df = pd.read_csv(config.SPLIT_TEST_PATH)
    un_df = pd.read_csv(config.SPLIT_UNSEEN_PLANT2_PATH)
    
    feats = [f for f in BASE_FEATURES if f in tr_df.columns]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(tr_df[feats])
    X_va_s = scaler.transform(va_df[feats])
    X_te_s = scaler.transform(te_df[feats])
    X_un_s = scaler.transform(un_df[feats])
    
    y_tr = tr_df[config.TARGET_COLUMN].to_numpy()
    y_va = va_df[config.TARGET_COLUMN].to_numpy()
    y_te = te_df[config.TARGET_COLUMN].to_numpy()
    y_un = un_df[config.TARGET_COLUMN].to_numpy()
    
    return {
        "features": feats,
        "scaler": scaler,
        "X_train": X_tr_s, "y_train": y_tr,
        "X_val": X_va_s, "y_val": y_va,
        "X_test": X_te_s, "y_test": y_te,
        "X_unseen": X_un_s, "y_unseen": y_un
    }


def build_augmented_feature_sets(splits, k=4, random_state=42):
    """Fit K-Means on train only; generate one-hot and distance-to-centroid features."""
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
    km.fit(splits["X_train"])
    
    # 1. Cluster assignments
    lab_tr = km.labels_
    lab_va = km.predict(splits["X_val"])
    lab_te = km.predict(splits["X_test"])
    lab_un = km.predict(splits["X_unseen"])
    
    # One-hot encoding
    oh_tr = np.eye(k)[lab_tr]
    oh_va = np.eye(k)[lab_va]
    oh_te = np.eye(k)[lab_te]
    oh_un = np.eye(k)[lab_un]
    
    # 2. Continuous distances to centroids
    dist_tr = km.transform(splits["X_train"])
    dist_va = km.transform(splits["X_val"])
    dist_te = km.transform(splits["X_test"])
    dist_un = km.transform(splits["X_unseen"])
    
    # Build 4 Feature Variants
    variants = {
        "Base_Only": {
            "X_train": splits["X_train"],
            "X_val": splits["X_val"],
            "X_test": splits["X_test"],
            "X_unseen": splits["X_unseen"],
            "dim": splits["X_train"].shape[1]
        },
        "Base_Plus_OneHot": {
            "X_train": np.c_[splits["X_train"], oh_tr],
            "X_val": np.c_[splits["X_val"], oh_va],
            "X_test": np.c_[splits["X_test"], oh_te],
            "X_unseen": np.c_[splits["X_unseen"], oh_un],
            "dim": splits["X_train"].shape[1] + k
        },
        "Base_Plus_Distances": {
            "X_train": np.c_[splits["X_train"], dist_tr],
            "X_val": np.c_[splits["X_val"], dist_va],
            "X_test": np.c_[splits["X_test"], dist_te],
            "X_unseen": np.c_[splits["X_unseen"], dist_un],
            "dim": splits["X_train"].shape[1] + k
        },
        "Composite_All": {
            "X_train": np.c_[splits["X_train"], oh_tr, dist_tr],
            "X_val": np.c_[splits["X_val"], oh_va, dist_va],
            "X_test": np.c_[splits["X_test"], oh_te, dist_te],
            "X_unseen": np.c_[splits["X_unseen"], oh_un, dist_un],
            "dim": splits["X_train"].shape[1] + 2 * k
        }
    }
    
    return variants, km


def train_augmented_models(splits, variants):
    """Train OLS and Ridge across all feature variants."""
    y_tr = splits["y_train"]
    y_te = splits["y_test"]
    y_un = splits["y_unseen"]
    
    results = {}
    
    for v_name, v_data in variants.items():
        # 1. OLS
        ols = LinearRegression()
        ols.fit(v_data["X_train"], y_tr)
        pred_te_ols = ols.predict(v_data["X_test"])
        pred_un_ols = ols.predict(v_data["X_unseen"])
        
        # 2. Ridge (L2 penalty)
        ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        ridge.fit(v_data["X_train"], y_tr)
        pred_te_rdg = ridge.predict(v_data["X_test"])
        pred_un_rdg = ridge.predict(v_data["X_unseen"])
        
        results[v_name] = {
            "dim": v_data["dim"],
            "OLS": {
                "test_rmse": round(float(np.sqrt(mean_squared_error(y_te, pred_te_ols))), 3),
                "test_mae": round(float(mean_absolute_error(y_te, pred_te_ols)), 3),
                "test_r2": round(float(r2_score(y_te, pred_te_ols)), 4),
                "unseen_rmse": round(float(np.sqrt(mean_squared_error(y_un, pred_un_ols))), 2),
                "unseen_r2": round(float(r2_score(y_un, pred_un_ols)), 4),
                "preds_test": pred_te_ols
            },
            "Ridge": {
                "alpha": float(ridge.alpha_),
                "test_rmse": round(float(np.sqrt(mean_squared_error(y_te, pred_te_rdg))), 3),
                "test_mae": round(float(mean_absolute_error(y_te, pred_te_rdg)), 3),
                "test_r2": round(float(r2_score(y_te, pred_te_rdg)), 4),
                "unseen_rmse": round(float(np.sqrt(mean_squared_error(y_un, pred_un_rdg))), 2),
                "unseen_r2": round(float(r2_score(y_un, pred_un_rdg)), 4),
                "preds_test": pred_te_rdg
            }
        }
        
    return results


def run_paired_bootstrap(y_true, pred_base, pred_aug, n_bootstraps=500, random_state=42):
    """Compute empirical bootstrap distribution and 95% CI for Delta RMSE (Base - Augmented)."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    delta_rmse_list = []
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_sample = y_true[idx]
        pb_sample = pred_base[idx]
        pa_sample = pred_aug[idx]
        
        rmse_base = float(np.sqrt(mean_squared_error(yt_sample, pb_sample)))
        rmse_aug = float(np.sqrt(mean_squared_error(yt_sample, pa_sample)))
        # Positive delta indicates augmentation reduced RMSE (improved model)
        delta_rmse_list.append(rmse_base - rmse_aug)
        
    delta_arr = np.array(delta_rmse_list)
    ci_lower = float(np.percentile(delta_arr, 2.5))
    ci_upper = float(np.percentile(delta_arr, 97.5))
    mean_delta = float(np.mean(delta_arr))
    
    is_significant = (ci_lower > 0) or (ci_upper < 0)
    
    return {
        "n_bootstraps": n_bootstraps,
        "mean_delta_rmse": round(mean_delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "delta_arr": delta_arr,
        "is_significant": is_significant
    }


def generate_augmentation_diagrams(results, boot_res):
    """Generate performance bar charts and paired bootstrap distribution plot."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plots = {}
    
    # 1. Performance Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    variants = list(results.keys())
    ols_rmse = [results[v]["OLS"]["test_rmse"] for v in variants]
    rdg_rmse = [results[v]["Ridge"]["test_rmse"] for v in variants]
    
    x = np.arange(len(variants))
    width = 0.35
    b1 = ax.bar(x - width/2, ols_rmse, width, label="OLS Linear Regression", color="#2563eb")
    b2 = ax.bar(x + width/2, rdg_rmse, width, label="Ridge Regularized Regression", color="#10b981")
    
    for b in list(b1) + list(b2):
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 0.02, f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
        
    ax.set_title("Supervised Regression Test RMSE Across Cluster Feature Augmentations", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature Augmentation Strategy", fontsize=11)
    ax.set_ylabel("Test RMSE (kW)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", " ") for v in variants], fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    p1 = config.PLOTS_DIR / "cluster_augmented_regression_performance.png"
    fig.savefig(p1, bbox_inches="tight", dpi=130)
    plt.close(fig)
    plots["performance"] = str(p1)
    
    # 2. Paired Bootstrap Distribution Chart
    fig, ax = plt.subplots(figsize=(8, 4.8))
    deltas = boot_res["delta_arr"]
    sns.histplot(deltas, bins=30, kde=True, color="#0f766e", ax=ax)
    
    ax.axvline(x=0, color="#dc2626", linestyle="--", linewidth=1.8, label="Zero Null Line (Delta = 0)")
    ax.axvline(x=boot_res["ci_lower"], color="#f59e0b", linestyle=":", linewidth=2, label=f"95% CI Lower ({boot_res['ci_lower']:.4f})")
    ax.axvline(x=boot_res["ci_upper"], color="#f59e0b", linestyle=":", linewidth=2, label=f"95% CI Upper ({boot_res['ci_upper']:.4f})")
    
    ax.set_title("Paired Bootstrap Verification: Delta RMSE (Base vs Composite)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Delta RMSE [kW] (Positive = Augmentation Improves Model)", fontsize=11)
    ax.set_ylabel("Bootstrap Sample Frequency", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    
    p2 = config.PLOTS_DIR / "cluster_augmented_bootstrap_distribution.png"
    fig.savefig(p2, bbox_inches="tight", dpi=130)
    plt.close(fig)
    plots["bootstrap"] = str(p2)
    
    return plots


def save_augmentation_report(results, boot_res, k):
    """Save comprehensive feature augmentation report."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rep_path = config.REPORTS_DIR / "cluster_feature_augmentation_report.txt"
    
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  MODULE 4: CLUSTERING FEATURE AUGMENTATION FOR SUPERVISED REGRESSION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. METHODOLOGY & LEAKAGE PREVENTION:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Number of Clusters (K):      {k}\n")
        f.write("Scaler & K-Means:            Fitted strictly on Plant 1 Train split\n")
        f.write("Validation/Test Inference:   Transformed via fitted scaler, predict(), and transform()\n")
        f.write("Target AC_POWER:             Never exposed to unsupervised components\n\n")
        
        f.write("2. SUPERVISED TEST SET PERFORMANCE ACROSS FEATURE VARIANTS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Variant':<22} | {'Dim':<5} | {'OLS RMSE':<10} | {'OLS R2':<8} | {'Ridge RMSE':<11} | {'Ridge R2':<8}\n")
        f.write("-" * 50 + "\n")
        for v_name, res in results.items():
            f.write(
                f"{v_name:<22} | {res['dim']:<5} | "
                f"{res['OLS']['test_rmse']:<10.3f} | {res['OLS']['test_r2']:<8.4f} | "
                f"{res['Ridge']['test_rmse']:<11.3f} | {res['Ridge']['test_r2']:<8.4f}\n"
            )
        f.write("\n")
        
        f.write("3. CROSS-PLANT TRANSFER (PLANT 2 UNSEEN):\n")
        f.write("-" * 50 + "\n")
        for v_name, res in results.items():
            f.write(f"{v_name:<22}: Ridge Unseen RMSE = {res['Ridge']['unseen_rmse']:.2f} kW | R2 = {res['Ridge']['unseen_r2']:.4f}\n")
        f.write("\n")
        
        f.write("4. PAIRED BOOTSTRAP SIGNIFICANCE TEST (500 RESAMPLES):\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean Delta RMSE:   {boot_res['mean_delta_rmse']:+.4f} kW\n")
        f.write(f"95% Confidence Interval: [{boot_res['ci_lower']:+.4f}, {boot_res['ci_upper']:+.4f}] kW\n")
        f.write(f"Statistically Significant: {boot_res['is_significant']}\n\n")
        f.write("Interpretation: The paired bootstrap establishes whether observed gains from\n")
        f.write("cluster-centroid distance coordinates represent genuine physical structural improvement\n")
        f.write("over the baseline linear feature space.\n")
        
    return str(rep_path)


def run_cluster_augmentation_pipeline():
    """Execute complete clustering feature augmentation pipeline."""
    print("  -> Running Clustering Feature Augmentation Pipeline...")
    splits = load_and_scale_splits()
    variants, km = build_augmented_feature_sets(splits, k=4)
    results = train_augmented_models(splits, variants)
    
    # Bootstrap comparison between Base_Only and Composite_All
    y_te = splits["y_test"]
    pred_base = results["Base_Only"]["Ridge"]["preds_test"]
    pred_aug = results["Composite_All"]["Ridge"]["preds_test"]
    boot_res = run_paired_bootstrap(y_te, pred_base, pred_aug, n_bootstraps=500)
    
    plots = generate_augmentation_diagrams(results, boot_res)
    rep_path = save_augmentation_report(results, boot_res, k=4)
    
    base_rmse = results["Base_Only"]["Ridge"]["test_rmse"]
    aug_rmse = results["Composite_All"]["Ridge"]["test_rmse"]
    print(f"     Ridge Test RMSE: Base = {base_rmse:.3f} kW -> Augmented = {aug_rmse:.3f} kW")
    print(f"     Bootstrap 95% CI: [{boot_res['ci_lower']:+.4f}, {boot_res['ci_upper']:+.4f}] kW")
    print(f"     Saved Report: {Path(rep_path).name}")
    print(f"     Generated Plots: {[Path(p).name for p in plots.values()]}")
    return results
