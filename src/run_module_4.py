"""Module 4 Master Execution Runner - Unsupervised Learning & Anomaly Detection.

Course: Machine Learning (25SC2107E) - Module 4
Applied to Solar Power Generation Prediction Project.

Executes all Module 4 pipelines:
  Phase 0: Cross-Validation Leakage Audit (cv_utils)
  Phase 1: Operational Solar Regimes via K-Means & Hierarchical Clustering
  Phase 2: Density Manifolds & Inverter Dropouts with DBSCAN
  Phase 3: PCA Multicollinearity Resolution & Standardization Pitfall
  Phase 4: Non-Linear Manifold Embeddings (t-SNE & UMAP)
  Phase 5: Multivariate Solar Anomaly Detection (Isolation Forest, One-Class SVM, Reconstruction)
  Phase 6: Supervised Feature Augmentation via Clustering (with Paired Bootstrap)

Usage:
  python src/run_module_4.py
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import config
from src.cv_utils import audit_cross_validation_leakage
from src.kmeans_clustering import run_kmeans_pipeline
from src.hierarchical_clustering import run_hierarchical_pipeline
from src.dbscan_clustering import run_dbscan_pipeline
from src.principal_component_analysis import run_pca_pipeline
from src.tsne_visualization import run_tsne_pipeline
from src.umap_visualization import run_umap_pipeline
from src.anomaly_detection import run_anomaly_pipeline
from src.clustering_feature_augmentation import run_cluster_augmentation_pipeline


def run_module_4():
    print("=" * 80)
    print("  SOLAR POWER GENERATION PREDICTION - MODULE 4")
    print("  Unsupervised Learning, Manifold Geometry & Anomaly Detection")
    print("=" * 80)
    start_time = time.perf_counter()

    train_df = pd.read_csv(config.SPLIT_TRAIN_PATH)

    # ---------------------------------------------------------
    # Phase 0: Cross-Validation Leakage Audit
    # ---------------------------------------------------------
    print("\n[Phase 0/6] Auditing Time-Series Validation Hygiene...")
    leak_audit = audit_cross_validation_leakage(train_df, n_splits=5)
    print(f"  * Shuffled K-Fold Leakage Overlap: {leak_audit['shuffled_kfold_overlap_pct']}% of samples")
    print(f"  * TimeSeriesSplit Overlap:         {leak_audit['time_series_split_overlap_pct']}% (Strict Zero Leakage)")

    # ---------------------------------------------------------
    # Phase 1: K-Means & Hierarchical Clustering
    # ---------------------------------------------------------
    print("\n[Phase 1/6] Running K-Means & Hierarchical Clustering Experiments...")
    km_data = run_kmeans_pipeline()
    h_data = run_hierarchical_pipeline()

    # ---------------------------------------------------------
    # Phase 2: DBSCAN Density Clustering
    # ---------------------------------------------------------
    print("\n[Phase 2/6] Running DBSCAN Density Clustering & Dropout Analysis...")
    db_data = run_dbscan_pipeline()

    # ---------------------------------------------------------
    # Phase 3: PCA & Dimensionality Reduction
    # ---------------------------------------------------------
    print("\n[Phase 3/6] Running Principal Component Analysis (PCA)...")
    pca_data = run_pca_pipeline()

    # ---------------------------------------------------------
    # Phase 4: Non-Linear Dimensionality Reduction (t-SNE & UMAP)
    # ---------------------------------------------------------
    print("\n[Phase 4/6] Running t-SNE & UMAP Manifold Embeddings...")
    tsne_data = run_tsne_pipeline()
    umap_data = run_umap_pipeline()

    # ---------------------------------------------------------
    # Phase 5: Multivariate Anomaly Detection
    # ---------------------------------------------------------
    print("\n[Phase 5/6] Running Multivariate Solar Anomaly Detection...")
    anom_data = run_anomaly_pipeline()

    # ---------------------------------------------------------
    # Phase 6: Supervised Feature Augmentation via Clustering
    # ---------------------------------------------------------
    print("\n[Phase 6/6] Running Clustering Feature Augmentation on Supervised Regression...")
    aug_data = run_cluster_augmentation_pipeline()

    elapsed = round(time.perf_counter() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"  MODULE 4 EXPERIMENTS COMPLETED SUCCESSFULLY in {elapsed}s!")
    print(f"  Output Plots Directory:   {config.PLOTS_DIR}")
    print(f"  Output Reports Directory: {config.REPORTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_module_4()
