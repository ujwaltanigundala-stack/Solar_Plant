"""Module 5 Master Execution Runner - Evaluation, Selection & Validation.

Course: Machine Learning (25SC2107E) - Module 5
Applied to Solar Power Generation Prediction Project.

Executes all Module 5 pipelines:
  Phase 7:  Cross-Validation Foundations & Time-Series Leakage (cross_validation.py)
  Phase 8:  Nested Cross-Validation & Selection Optimism Bias (nested_cross_validation.py)
  Phase 9A: Grid Search vs Random Search (grid_search_vs_random_search.py)
  Phase 9B: Bayesian Optimization via Optuna TPE (bayesian_optimization.py)
  Phase 9C: Learning Curves & Regularization Validation Curves (learning_and_validation_curves.py)
  Phase 9D: Systematic Model Diagnostics & Bias-Variance Decomposition (model_diagnostics.py)
  Phase 10: Regression Metrics, Residual Diagnostics & Outlier Stress (regression_metrics_and_robustness.py)
  Phase 11: Statistical Significance & Paired Bootstrap CIs (statistical_significance.py)

Usage:
  python src/run_module_5.py
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.cross_validation import run_cross_validation_pipeline
from src.nested_cross_validation import run_nested_cv_pipeline
from src.grid_search_vs_random_search import run_search_pipeline
from src.bayesian_optimization import run_bayesian_pipeline
from src.learning_and_validation_curves import run_curves_pipeline
from src.model_diagnostics import run_diagnostics_pipeline
from src.regression_metrics_and_robustness import run_regression_metrics_pipeline
from src.statistical_significance import run_significance_pipeline


def run_module_5():
    print("=" * 85)
    print("  SOLAR POWER GENERATION PREDICTION - MODULE 5 MASTER SUITE")
    print("  Model Evaluation, Selection, Calibration & Statistical Significance")
    print("=" * 85)
    total_start = time.perf_counter()

    # ---------------------------------------------------------
    # Phase 7: Cross-Validation & Time-Series Leakage
    # ---------------------------------------------------------
    print("\n" + "#" * 60)
    print("# [Phase 7/11] Running Cross-Validation & Temporal Leakage Suite")
    print("#" * 60)
    t0 = time.perf_counter()
    cv_data = run_cross_validation_pipeline()
    print(f"-> Phase 7 completed in {time.perf_counter() - t0:.2f}s")

    # ---------------------------------------------------------
    # Phase 8: Nested Cross-Validation
    # ---------------------------------------------------------
    print("\n" + "#" * 60)
    print("# [Phase 8/11] Running 5x3 Nested Cross-Validation Suite")
    print("#" * 60)
    t0 = time.perf_counter()
    nested_data = run_nested_cv_pipeline()
    print(f"-> Phase 8 completed in {time.perf_counter() - t0:.2f}s")

    # ---------------------------------------------------------
    # Phase 9A: Grid Search vs Random Search
    # ---------------------------------------------------------
    print("\n" + "#" * 60)
    print("# [Phase 9A/11] Running Grid Search vs Random Search Suite")
    print("#" * 60)
    t0 = time.perf_counter()
    search_data = run_search_pipeline()
    print(f"-> Phase 9A completed in {time.perf_counter() - t0:.2f}s")

    # ---------------------------------------------------------
    # Phase 9B: Bayesian Optimization (Optuna TPE)
    # ---------------------------------------------------------
    print("\n" + "#" * 60)
    print("# [Phase 9B/11] Running Optuna Bayesian Optimization Suite")
    print("#" * 60)
    t0 = time.perf_counter()
    bayes_data = run_bayesian_pipeline()
    print(f"-> Phase 9B completed in {time.perf_counter() - t0:.2f}s")

    # ---------------------------------------------------------
    # Phase 9C: Learning Curves & Validation Curves
    # ---------------------------------------------------------
    print("\n" + "#" * 60)
    print("# [Phase 9C/11] Running Learning & Validation Curves Suite")
    print("#" * 60)
    t0 = time.perf_counter()
    curves_data = run_curves_pipeline()
    print(f"-> Phase 9C completed in {time.perf_counter() - t0:.2f}s")

    # ---------------------------------------------------------
    # Phase 9D: Systematic Model Diagnostics & Bias-Variance
    # ---------------------------------------------------------
    print("\n" + "#" * 60)
    print("# [Phase 9D/11] Running Bias-Variance Model Diagnostics Suite")
    print("#" * 60)
    t0 = time.perf_counter()
    diag_data = run_diagnostics_pipeline()
    print(f"-> Phase 9D completed in {time.perf_counter() - t0:.2f}s")

    # ---------------------------------------------------------
    # Phase 10: Regression Metrics, Residual Diagnostics & Robustness
    # ---------------------------------------------------------
    print("\n" + "#" * 60)
    print("# [Phase 10/11] Running Regression Metrics & Outlier Stress Suite")
    print("#" * 60)
    t0 = time.perf_counter()
    metrics_data = run_regression_metrics_pipeline()
    print(f"-> Phase 10 completed in {time.perf_counter() - t0:.2f}s")

    # ---------------------------------------------------------
    # Phase 11: Statistical Significance & Paired Bootstrap CIs
    # ---------------------------------------------------------
    print("\n" + "#" * 60)
    print("# [Phase 11/11] Running Statistical Significance & Paired Bootstrap")
    print("#" * 60)
    t0 = time.perf_counter()
    sig_data = run_significance_pipeline()
    print(f"-> Phase 11 completed in {time.perf_counter() - t0:.2f}s")

    total_elapsed = round(time.perf_counter() - total_start, 2)
    print("\n" + "=" * 85)
    print(f"  MODULE 5 EXPERIMENTS COMPLETED SUCCESSFULLY in {total_elapsed}s!")
    print(f"  Output Plots Directory:   {config.PLOTS_DIR}")
    print(f"  Output Reports Directory: {config.REPORTS_DIR}")
    print("=" * 85)

    return {
        "cv_data": cv_data,
        "nested_data": nested_data,
        "search_data": search_data,
        "bayes_data": bayes_data,
        "curves_data": curves_data,
        "diag_data": diag_data,
        "metrics_data": metrics_data,
        "sig_data": sig_data,
        "total_elapsed_seconds": total_elapsed
    }


if __name__ == "__main__":
    run_module_5()
