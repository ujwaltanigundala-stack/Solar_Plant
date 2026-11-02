"""Course: Machine Learning (25SC2107E) - Module 4: Unsupervised Learning
Session 24: Hierarchical (Agglomerative) Clustering on Solar Power Generation Dataset.

Topics covered:
1. Feature extraction, imputation, and standardization strictly on Plant 1 Train (zero target leakage).
2. Subsampling (n=2,500) to manage O(n^2) pairwise distance matrix memory and compute complexity.
3. Four Linkage Methods compared: Ward, Complete, Average, Single.
4. Cophenetic Correlation Coefficient computation evaluating geometric metric fidelity.
5. Demonstrating Single-Linkage Chaining Effect vs Ward's balanced variance minimization at K=4.
6. Cutting the Ward tree across heights (K=2..8) and silhouette evaluation.
7. Adjusted Rand Index (ARI) cross-validation comparing Hierarchical (Ward) vs K-Means.
8. 2x2 Dendrogram visualization and Cophenetic correlation comparison chart.
9. Text report generation (hierarchical_clustering_report.txt).
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
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import config

CLUSTERING_FEATURES = [
    "DC_POWER",
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "TEMP_DIFF",
    "HOUR"
]


def get_hierarchical_data(train_df=None, n_sample=2500, random_state=42):
    """Extract and standardize continuous solar features with representative subsampling."""
    if train_df is None:
        train_df = pd.read_csv(config.SPLIT_TRAIN_PATH)
        
    features = [f for f in CLUSTERING_FEATURES if f in train_df.columns]
    X_raw = train_df[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    n_total = len(X_scaled)
    sample_size = min(n_sample, n_total)
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(n_total, size=sample_size, replace=False)
    
    X_sample = X_scaled[sample_idx]
    sample_df = train_df.iloc[sample_idx].copy().reset_index(drop=True)
    
    return train_df, features, X_sample, sample_df, scaler


def train_hierarchical_clustering(train_df=None, n_sample=2500, k_eval=4, random_state=42):
    """Fit hierarchical clustering across 4 linkage methods and evaluate cophenetic correlation."""
    train_df, features, X_sample, sample_df, scaler = get_hierarchical_data(
        train_df, n_sample=n_sample, random_state=random_state
    )
    
    condensed_dist = pdist(X_sample, metric="euclidean")
    linkage_methods = ["ward", "complete", "average", "single"]
    
    linkage_matrices = {}
    cophenetic_scores = {}
    
    for method in linkage_methods:
        t0 = time.perf_counter()
        Z = linkage(X_sample, method=method)
        elapsed = time.perf_counter() - t0
        linkage_matrices[method] = Z
        coph_corr, _ = cophenet(Z, condensed_dist)
        cophenetic_scores[method] = {
            "score": round(float(coph_corr), 4),
            "time_sec": round(elapsed, 4)
        }
        
    # Single-linkage chaining diagnostic vs Ward balance at K=4
    single_labels_4 = fcluster(linkage_matrices["single"], t=k_eval, criterion="maxclust")
    single_sizes = pd.Series(single_labels_4).value_counts().sort_index().to_dict()
    
    ward_labels_4 = fcluster(linkage_matrices["ward"], t=k_eval, criterion="maxclust")
    ward_sizes = pd.Series(ward_labels_4).value_counts().sort_index().to_dict()
    
    # Ward tree cutting across K=2..8 and silhouette scores
    cut_results = []
    for k in range(2, 9):
        labels_k = fcluster(linkage_matrices["ward"], t=k, criterion="maxclust")
        sil = float(silhouette_score(X_sample, labels_k))
        cut_results.append({"K": k, "Silhouette": round(sil, 4)})
    cut_df = pd.DataFrame(cut_results)
    
    # ARI Comparison: Hierarchical (Ward, K=4) vs K-Means (K=4)
    km = KMeans(n_clusters=k_eval, init="k-means++", n_init=10, random_state=random_state)
    km_labels = km.fit_predict(X_sample)
    ari = float(adjusted_rand_score(ward_labels_4, km_labels))
    
    return {
        "features": features,
        "n_samples": len(X_sample),
        "linkage_matrices": linkage_matrices,
        "cophenetic_scores": cophenetic_scores,
        "single_sizes": single_sizes,
        "ward_sizes": ward_sizes,
        "cut_df": cut_df,
        "ari_vs_kmeans": round(ari, 4),
        "k_eval": k_eval,
        "X_sample": X_sample,
        "sample_df": sample_df,
        "ward_labels": ward_labels_4
    }


def generate_hierarchical_diagrams(h_data):
    """Generate dendrograms, cophenetic comparison, and chaining diagnostics."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    linkage_matrices = h_data["linkage_matrices"]
    cophenetic_scores = h_data["cophenetic_scores"]
    cut_df = h_data["cut_df"]
    
    plots = {}
    
    # 1. 2x2 Comparative Dendrogram Grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    method_titles = {
        "ward": "Ward Linkage (Minimum Variance)",
        "complete": "Complete Linkage (Maximum Distance)",
        "average": "Average Linkage (UPGMA)",
        "single": "Single Linkage (Minimum Distance — Demonstrates Chaining)"
    }
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for (method, title), pos in zip(method_titles.items(), positions):
        ax = axes[pos[0], pos[1]]
        Z = linkage_matrices[method]
        coph = cophenetic_scores[method]["score"]
        dendrogram(
            Z,
            truncate_mode="lastp",
            p=25,
            leaf_rotation=90,
            leaf_font_size=9,
            show_contracted=True,
            ax=ax,
            color_threshold=0.7 * max(Z[:, 2])
        )
        ax.set_title(f"{title}\nCophenetic Corr = {coph}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Aggregated Leaf Clusters", fontsize=10)
        ax.set_ylabel("Linkage Distance Threshold", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        
    plt.tight_layout()
    p1 = config.PLOTS_DIR / "hierarchical_dendrograms_grid.png"
    fig.savefig(p1, bbox_inches="tight", dpi=130)
    plt.close(fig)
    plots["dendrograms"] = str(p1)
    
    # 2. Cophenetic Correlation Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.8))
    methods = list(cophenetic_scores.keys())
    scores = [cophenetic_scores[m]["score"] for m in methods]
    colors = ["#2563eb", "#059669", "#d97706", "#dc2626"]
    
    bars = ax.bar(methods, scores, color=colors, width=0.55)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.015, f"{s:.4f}", ha="center", fontsize=10, fontweight="bold")
        
    ax.set_ylim(0, 1.05)
    ax.set_title("Cophenetic Correlation Coefficient by Linkage Method", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hierarchical Linkage Method", fontsize=11)
    ax.set_ylabel("Cophenetic Correlation (Closeness to True Metric)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    p2 = config.PLOTS_DIR / "hierarchical_linkage_cophenetic.png"
    fig.savefig(p2, bbox_inches="tight", dpi=130)
    plt.close(fig)
    plots["cophenetic"] = str(p2)
    
    # 3. Chaining Effect Diagnostic vs Ward Balance (K=4)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    k_vals = list(range(1, 5))
    single_counts = [h_data["single_sizes"].get(k, 0) for k in k_vals]
    ward_counts = [h_data["ward_sizes"].get(k, 0) for k in k_vals]
    
    x = np.arange(len(k_vals))
    width = 0.35
    ax.bar(x - width/2, single_counts, width, label="Single Linkage (Chaining)", color="#dc2626")
    ax.bar(x + width/2, ward_counts, width, label="Ward Linkage (Balanced)", color="#2563eb")
    
    ax.set_title("Cluster Size Distribution at K=4: Single Chaining vs Ward Balance", fontsize=12, fontweight="bold")
    ax.set_xlabel("Discovered Cluster Index", fontsize=11)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {k}" for k in k_vals])
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    p3 = config.PLOTS_DIR / "hierarchical_chaining_diagnostic.png"
    fig.savefig(p3, bbox_inches="tight", dpi=130)
    plt.close(fig)
    plots["chaining"] = str(p3)
    
    return plots


def save_hierarchical_report(h_data):
    """Save comprehensive hierarchical clustering report."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rep_path = config.REPORTS_DIR / "hierarchical_clustering_report.txt"
    
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  MODULE 4: HIERARCHICAL (AGGLOMERATIVE) CLUSTERING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. EXPERIMENTAL SETUP & COMPLEXITY MANAGEMENT:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Sample Size:              {h_data['n_samples']:,} rows (subsampled from Plant 1 Train)\n")
        f.write(f"Pairwise Distances:       {h_data['n_samples'] * (h_data['n_samples'] - 1) // 2:,} elements\n")
        f.write(f"Adjusted Rand Index (ARI): Ward vs K-Means (K=4) = {h_data['ari_vs_kmeans']:.4f}\n\n")
        
        f.write("2. LINKAGE METHODS & COPHENETIC CORRELATION RANKINGS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Linkage Method':<18} | {'Cophenetic Correlation':<25} | {'Fit Time (s)':<14}\n")
        f.write("-" * 50 + "\n")
        sorted_coph = sorted(h_data["cophenetic_scores"].items(), key=lambda x: x[1]["score"], reverse=True)
        for method, info in sorted_coph:
            f.write(f"{method.title():<18} | {info['score']:<25.4f} | {info['time_sec']:<14.4f}\n")
        f.write("\n")
        
        f.write("3. SINGLE-LINKAGE CHAINING EFFECT AUDIT (K=4):\n")
        f.write("-" * 50 + "\n")
        f.write("Single Linkage Sizes: " + str(h_data["single_sizes"]) + "\n")
        f.write("Ward Linkage Sizes:   " + str(h_data["ward_sizes"]) + "\n\n")
        f.write("Mathematical Pathology: Single linkage repeatedly merges nearest solitary points\n")
        f.write("into a single massive cluster, leaving other clusters with trivial outliers (chaining effect).\n")
        f.write("Ward linkage minimizes within-cluster variance, generating balanced physical solar cohorts.\n\n")
        
        f.write("4. WARD TREE CUTTING HEIGHTS (K = 2..8):\n")
        f.write("-" * 50 + "\n")
        for _, row in h_data["cut_df"].iterrows():
            f.write(f"K = {int(row['K'])}: Mean Silhouette = {row['Silhouette']:.4f}\n")
            
    return str(rep_path)


def run_hierarchical_pipeline():
    """Execute complete hierarchical clustering pipeline."""
    print("  -> Running Hierarchical Clustering & Linkage Diagnostic...")
    h_data = train_hierarchical_clustering()
    h_plots = generate_hierarchical_diagrams(h_data)
    rep_path = save_hierarchical_report(h_data)
    print(f"     ARI (Ward vs K-Means): {h_data['ari_vs_kmeans']:.4f}")
    print(f"     Saved Report: {Path(rep_path).name}")
    print(f"     Generated Plots: {[Path(p).name for p in h_plots.values()]}")
    return h_data
