"""Module 2 Execution Runner - Supervised Learning: Linear Models & Regularization.

Executes:
  1. Feature scaling comparison (Standard vs MinMax vs Robust)
  2. Ordinary Least Squares (OLS) Multiple Linear Regression (Normal Equation)
  3. Vectorized Batch Gradient Descent from scratch (NumPy)
  4. Single-Feature baselines (DC_POWER only and IRRADIATION only)
  5. Regularized Regression with 5-Fold CV (Ridge L2, Lasso L1, Elastic Net)
  6. Matrix condition number and analytical Ridge closed-form checks
  7. Cross-Plant generalization benchmark on Plant 2 Unseen data
  8. Diagnostic visual charts and automated text summary reports

Usage:
  python src/run_module_2.py
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import config
from src.feature_eng import compare_scalers, get_feature_matrices
from src.linear_regression import (
    generate_linear_diagrams,
    save_linear_regression_report,
    train_linear_models,
)
from src.regularization import (
    generate_regularization_diagrams,
    save_regularization_report,
    train_regularized_models,
)


def run_module_2():
    print("=" * 80)
    print("  SOLAR POWER GENERATION PREDICTION - MODULE 2")
    print("  Supervised Learning: Linear Models, Gradient Descent & Regularization")
    print("=" * 80)
    start_time = time.perf_counter()

    # Step 1: Load Persistent Splits
    print("\n[Step 1/5] Loading Persistent Dataset Splits...")
    train_df = pd.read_csv(config.SPLIT_TRAIN_PATH)
    val_df = pd.read_csv(config.SPLIT_VAL_PATH)
    test_df = pd.read_csv(config.SPLIT_TEST_PATH)
    unseen_df = pd.read_csv(config.SPLIT_UNSEEN_PLANT2_PATH)

    print(f"  -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,} | Unseen Plant 2: {len(unseen_df):,}")

    # Step 2: Feature Scaling Comparison
    print("\n[Step 2/5] Evaluating Feature Scaler Behaviors (Standard vs MinMax vs Robust)...")
    matrices = get_feature_matrices(train_df, test_df=test_df)
    scaler_benchmarks = compare_scalers(matrices["X_train"], matrices["y_train"], matrices["X_test"], matrices["y_test"])
    for sb in scaler_benchmarks:
        print(f"  * {sb['scaler'].title():<10} Scaler: Test RMSE = {sb['rmse']:.2f} kW | R² = {sb['r2']:.4f} | Condition Number = {sb['condition_number']:,.1f}")

    # Step 3: OLS Multiple Linear Regression & NumPy Gradient Descent
    print("\n[Step 3/5] Training OLS (Normal Equation) & NumPy Batch Gradient Descent...")
    linear_data = train_linear_models(train_df, val_df, test_df, unseen_df)
    linear_plots = generate_linear_diagrams(linear_data)
    rep_linear = save_linear_regression_report(linear_data)

    m_test = linear_data["test_metrics"]
    gd = linear_data["gd_results"]
    b_dc = linear_data["baseline_dc"]["metrics"]
    b_irr = linear_data["baseline_irr"]["metrics"]

    print(f"  -> OLS Normal Eq Test:  RMSE = {m_test['rmse']:.2f} kW | MAE = {m_test['mae']:.2f} kW | R² = {m_test['r2']:.4f} | Daylight MAPE = {m_test['daylight_mape']}%")
    print(f"  -> Batch GD Test:       RMSE = {linear_data['gd_test_metrics']['rmse']:.2f} kW | R² = {linear_data['gd_test_metrics']['r2']:.4f} (Converged in {gd['elapsed_sec']}s)")
    print(f"  -> DC-Only Baseline:    RMSE = {b_dc['rmse']:.2f} kW | R² = {b_dc['r2']:.4f}")
    print(f"  -> Irradiation-Only:    RMSE = {b_irr['rmse']:.2f} kW | R² = {b_irr['r2']:.4f}")
    print(f"  -> Generated Plots:     {list(linear_plots.keys())}")
    print(f"  -> Generated Report:    {Path(rep_linear).name}")

    # Step 4: Regularized Regression (Ridge, Lasso, Elastic Net) with CV
    print("\n[Step 4/5] Training Regularized Models (Ridge, Lasso, Elastic Net) with 5-Fold CV...")
    reg_data = train_regularized_models(train_df, val_df, test_df, unseen_df)
    reg_plots = generate_regularization_diagrams(reg_data)
    rep_reg = save_regularization_report(reg_data)

    print(f"  -> Optimal Alphas: Ridge = {reg_data['best_alpha_ridge']} | Lasso = {reg_data['best_alpha_lasso']} | ElasticNet = {reg_data['best_alpha_elastic']}")
    print(f"  -> Matrix Condition Number: Unregularized = {reg_data['condition_number_unreg']:,.1f} -> Ridge = {reg_data['condition_number_ridge']:,.1f}")

    print("\n  Regularized Models Performance Summary (Plant 1 Test Set):")
    for name, r in reg_data["results"].items():
        tm = r["test_metrics"]
        print(f"     * {name:<11}: Test RMSE = {tm['rmse']:.2f} kW | R² = {tm['r2']:.4f} | Retained Features = {r['retained_features_count']}/{len(reg_data['features'])}")

    print("\n  Cross-Plant Generalization Benchmark (Plant 2 Unseen Data):")
    for name, r in reg_data["results"].items():
        um = r["unseen_metrics"]
        print(f"     * {name:<11}: Unseen RMSE = {um['rmse']:.2f} kW | Unseen MAE = {um['mae']:.2f} kW | Unseen R² = {um['r2']:.4f}")

    print(f"  -> Generated Plots:     {list(reg_plots.keys())}")
    print(f"  -> Generated Report:    {Path(rep_reg).name}")

    # Step 5: Wrap up
    elapsed = round(time.perf_counter() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"  MODULE 2 PIPELINE COMPLETED SUCCESSFULLY in {elapsed}s!")
    print(f"  Output Plots Directory:   {config.PLOTS_DIR}")
    print(f"  Output Reports Directory: {config.REPORTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_module_2()
