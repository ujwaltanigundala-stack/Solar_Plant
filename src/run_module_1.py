"""Module 1 Execution Runner - Solar Power ML Lifecycle & Ingestion Pipeline.

Executes:
  1. Data ingestion & multi-source sensor integration (Generation + Weather)
  2. Data cleaning, bounds validation & IQR outlier detection
  3. Feature engineering (solar physical & temporal features)
  4. Time-aware chronological splitting (Plant 1) & Unseen Hold-out split (Plant 2)
  5. Zero-Rule Baseline benchmark computation
  6. Comprehensive EDA visualizations and summary reports

Usage:
  python src/run_module_1.py
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_utils import (
    calculate_baseline_metrics,
    clean_plant_data,
    generate_and_save_splits,
    load_cleaned_plant,
    load_raw_plant,
    save_baseline_report,
    save_cleaning_report,
    save_split_report,
)
from src.EDA import generate_all_plots, generate_eda_summary


def run_module_1():
    print("=" * 80)
    print("  SOLAR POWER GENERATION PREDICTION - MODULE 1")
    print("  Data Lifecycle, Ingestion, Integration, Cleaning, Splits & Baselines")
    print("=" * 80)
    start_time = time.perf_counter()

    # Step 1: Multi-source Ingestion & Deduplication
    print("\n[Step 1/5] Ingesting & Integrating Generation + Weather Data...")
    raw_p1 = load_raw_plant(plant_id=1)
    clean_p1, dropped_p1 = clean_plant_data(raw_p1, save_path=config.CLEANED_PLANT1_PATH)
    print(f"  -> Plant 1: Ingested {len(raw_p1):,} rows | Cleaned {len(clean_p1):,} rows ({dropped_p1} dropped/deduped)")

    raw_p2 = load_raw_plant(plant_id=2)
    clean_p2, dropped_p2 = clean_plant_data(raw_p2, save_path=config.CLEANED_PLANT2_PATH)
    print(f"  -> Plant 2: Ingested {len(raw_p2):,} rows | Cleaned {len(clean_p2):,} rows ({dropped_p2} dropped/deduped)")

    # Save Cleaning Report
    rep_clean = save_cleaning_report(raw_p1, clean_p1, dropped_p1, raw_p2, clean_p2, dropped_p2)
    print(f"  -> Generated Cleaning Report: {Path(rep_clean).name}")

    # Step 2: Time-Aware Persistent Splits & Unseen Data Split
    print("\n[Step 2/5] Creating Time-Aware Chronological Splits & Unseen Benchmark...")
    split_info = generate_and_save_splits()
    rep_split = save_split_report(split_info)
    print(f"  -> Plant 1 Train:      {split_info['train_len']:,} rows (70%) [{split_info['train_dates'][0]} to {split_info['train_dates'][1]}]")
    print(f"  -> Plant 1 Validation: {split_info['val_len']:,} rows (15%) [{split_info['val_dates'][0]} to {split_info['val_dates'][1]}]")
    print(f"  -> Plant 1 Test:       {split_info['test_len']:,} rows (15%) [{split_info['test_dates'][0]} to {split_info['test_dates'][1]}]")
    print(f"  -> Plant 2 Unseen:     {split_info['plant2_unseen_len']:,} rows (Held-out Generalization Benchmark)")
    print(f"  -> Saved CSV splits to: {config.SPLITS_DIR}")
    print(f"  -> Generated Split Report: {Path(rep_split).name}")

    # Step 3: Zero-Rule Baseline Benchmarks
    print("\n[Step 3/5] Calculating Zero-Rule Baseline Benchmarks on Test Split...")
    import pandas as pd
    train_df = pd.read_csv(config.SPLIT_TRAIN_PATH)
    test_df = pd.read_csv(config.SPLIT_TEST_PATH)
    baseline_metrics = calculate_baseline_metrics(train_df, test_df)
    rep_base = save_baseline_report(baseline_metrics)

    m = baseline_metrics["global_mean_baseline"]
    d = baseline_metrics["diurnal_solar_baseline"]
    print(f"  -> Global Mean Baseline:  MAE={m['mae']:.2f} kW | RMSE={m['rmse']:.2f} kW | R2={m['r2']:.4f}")
    print(f"  -> Diurnal Solar Baseline: MAE={d['mae']:.2f} kW | RMSE={d['rmse']:.2f} kW | R2={d['r2']:.4f}")
    print(f"  -> Generated Baseline Report: {Path(rep_base).name}")

    # Step 4: Exploratory Data Analysis (EDA) Charts
    print("\n[Step 4/5] Generating Diagnostic & Exploratory Visualizations...")
    plots = generate_all_plots(clean_p1, clean_p2)
    for name, p in plots.items():
        print(f"  * Generated plot: {Path(p).name}")

    # Step 5: Comprehensive Statistical EDA Summary
    print("\n[Step 5/5] Generating Comprehensive EDA Summary...")
    rep_eda = generate_eda_summary(clean_p1, clean_p2)
    print(f"  -> Generated EDA Summary: {Path(rep_eda).name}")

    elapsed = round(time.perf_counter() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"  MODULE 1 PIPELINE COMPLETED SUCCESSFULLY in {elapsed}s!")
    print(f"  Output Plots Directory:   {config.PLOTS_DIR}")
    print(f"  Output Reports Directory: {config.REPORTS_DIR}")
    print(f"  Persistent Splits:        {config.SPLITS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_module_1()
