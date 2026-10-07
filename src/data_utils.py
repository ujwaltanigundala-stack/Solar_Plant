"""Data utilities for Solar Power Generation Prediction Project.

Handles data ingestion, plant-level sensor merging, IQR outlier detection,
data cleaning, feature engineering, time-aware dataset splitting, baseline benchmarks,
and automated report generation.
Follows the pattern from the Placement Prediction reference project.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config


# ======================================================================================
# 1. DATA INGESTION & INTEGRATION
# ======================================================================================

def load_raw_plant(plant_id=1):
    """Load and merge raw generation and weather sensor CSVs for Plant 1 or Plant 2.
    
    Standardizes differing timestamp formats across plants and merges on DATE_TIME.
    """
    if plant_id == 1:
        gen_path = config.PLANT1_GEN_PATH
        weather_path = config.PLANT1_WEATHER_PATH
        gen_dt_format = "%d-%m-%Y %H:%M"
        weather_dt_format = "%Y-%m-%d %H:%M:%S"
    elif plant_id == 2:
        gen_path = config.PLANT2_GEN_PATH
        weather_path = config.PLANT2_WEATHER_PATH
        gen_dt_format = "%Y-%m-%d %H:%M:%S"
        weather_dt_format = "%Y-%m-%d %H:%M:%S"
    else:
        raise ValueError(f"Unsupported plant_id: {plant_id}. Must be 1 or 2.")

    gen_df = pd.read_csv(gen_path)
    weather_df = pd.read_csv(weather_path)

    # Standardize datetime parsing
    gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"], format=gen_dt_format, errors="coerce")
    weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"], format=weather_dt_format, errors="coerce")

    # Rename sensor key in weather to avoid collision with inverter source key
    weather_clean = weather_df.rename(columns={"SOURCE_KEY": "SOURCE_KEY_SENSOR"})
    gen_clean = gen_df.rename(columns={"SOURCE_KEY": "SOURCE_KEY_INVERTER"})

    # Drop redundant PLANT_ID from weather before merge
    if "PLANT_ID" in weather_clean.columns:
        weather_clean = weather_clean.drop(columns=["PLANT_ID"])

    # Inner merge on standardized DATE_TIME
    merged = pd.merge(gen_clean, weather_clean, on="DATE_TIME", how="inner")
    merged = merged.sort_values(by=["DATE_TIME", "SOURCE_KEY_INVERTER"]).reset_index(drop=True)
    return merged


# ======================================================================================
# 2. OUTLIER DETECTION (1.5 * IQR METHOD)
# ======================================================================================

def detect_iqr_outliers(df, cols=None):
    """Detect and summarize outliers using the standard 1.5 * IQR method."""
    if cols is None:
        cols = [c for c in ["DC_POWER", "AC_POWER", "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"] if c in df.columns]

    report = {}
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        report[col] = {
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_count": int(len(outliers)),
            "outlier_percent": round(float(len(outliers) / len(series) * 100), 2) if len(series) > 0 else 0.0,
            "min_val": round(float(series.min()), 4),
            "max_val": round(float(series.max()), 4),
        }
    return report


# ======================================================================================
# 3. DATA CLEANING & FEATURE ENGINEERING
# ======================================================================================

def engineer_features(df):
    """Derive domain-relevant temporal and physical interaction features."""
    data = df.copy()

    # Temporal feature extraction
    dt = pd.to_datetime(data["DATE_TIME"])
    data["HOUR"] = dt.dt.hour
    data["MINUTE"] = dt.dt.minute
    data["DAY"] = dt.dt.day
    data["DAY_OF_WEEK"] = dt.dt.dayofweek  # 0=Monday, 6=Sunday
    data["TIME_DECIMAL"] = data["HOUR"] + data["MINUTE"] / 60.0

    # Solar physics features
    # Thermal gradient: module temperature rises above ambient under direct irradiation
    if "MODULE_TEMPERATURE" in data.columns and "AMBIENT_TEMPERATURE" in data.columns:
        data["TEMP_DIFF"] = (data["MODULE_TEMPERATURE"] - data["AMBIENT_TEMPERATURE"]).round(4)

    # Daylight indicator (direct sunlight presence)
    if "IRRADIATION" in data.columns:
        data["IS_DAYLIGHT"] = (data["IRRADIATION"] > 0.0001).astype(int)

    return data


def clean_plant_data(df, save_path=None):
    """Clean solar data: bounds enforcement, non-negative clipping, deduplication, feature engineering."""
    data = df.copy()
    initial_rows = len(data)

    # Drop null timestamps or invalid rows
    data = data.dropna(subset=["DATE_TIME", "DC_POWER", "AC_POWER"])

    # Physical domain constraints (power and irradiation cannot be negative)
    non_negative_cols = ["DC_POWER", "AC_POWER", "IRRADIATION"]
    for col in non_negative_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").clip(lower=0.0)

    # Deduplicate exact inverter readings at identical timestamps
    dedup_cols = ["DATE_TIME", "SOURCE_KEY_INVERTER"]
    if all(c in data.columns for c in dedup_cols):
        data = data.drop_duplicates(subset=dedup_cols, keep="first")

    # Engineer temporal and physical features
    data = engineer_features(data)

    dropped = initial_rows - len(data)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path, index=False)

    return data, dropped


def load_cleaned_plant(plant_id=1):
    """Load cleaned dataset from disk or clean raw data if not yet saved."""
    cleaned_path = config.CLEANED_PLANT1_PATH if plant_id == 1 else config.CLEANED_PLANT2_PATH
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
        return df
    else:
        raw_df = load_raw_plant(plant_id=plant_id)
        cleaned_df, _ = clean_plant_data(raw_df, save_path=cleaned_path)
        return cleaned_df


# ======================================================================================
# 4. DATASET SPLITTING (TIME-AWARE TEMPORAL SPLIT & UNSEEN GENERALIZATION)
# ======================================================================================

def create_time_aware_splits(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Split time-series solar dataset chronologically to avoid future data leakage.
    
    Train: First 70% of chronological timeline
    Validation: Next 15% of chronological timeline
    Test: Final 15% of chronological timeline
    """
    data = df.sort_values(by="DATE_TIME").reset_index(drop=True)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = data.iloc[:train_end].copy().reset_index(drop=True)
    val = data.iloc[train_end:val_end].copy().reset_index(drop=True)
    test = data.iloc[val_end:].copy().reset_index(drop=True)

    return train, val, test


def generate_and_save_splits():
    """Create persistent train, validation, test splits for Plant 1, and unseen dataset for Plant 2.
    
    Plant 1 is split chronologically (70/15/15) for model training and tuning.
    Plant 2 is saved as unseen hold-out data to evaluate cross-plant generalization.
    """
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Clean Plant 1 & Plant 2
    raw_p1 = load_raw_plant(plant_id=1)
    clean_p1, dropped_p1 = clean_plant_data(raw_p1, save_path=config.CLEANED_PLANT1_PATH)

    raw_p2 = load_raw_plant(plant_id=2)
    clean_p2, dropped_p2 = clean_plant_data(raw_p2, save_path=config.CLEANED_PLANT2_PATH)

    # 2. Time-aware split for Plant 1
    train_df, val_df, test_df = create_time_aware_splits(clean_p1, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    # 3. Save splits
    train_df.to_csv(config.SPLIT_TRAIN_PATH, index=False)
    val_df.to_csv(config.SPLIT_VAL_PATH, index=False)
    test_df.to_csv(config.SPLIT_TEST_PATH, index=False)
    clean_p2.to_csv(config.SPLIT_UNSEEN_PLANT2_PATH, index=False)

    return {
        "plant1_total": len(clean_p1),
        "train_len": len(train_df),
        "val_len": len(val_df),
        "test_len": len(test_df),
        "plant2_unseen_len": len(clean_p2),
        "train_dates": (train_df["DATE_TIME"].min(), train_df["DATE_TIME"].max()),
        "val_dates": (val_df["DATE_TIME"].min(), val_df["DATE_TIME"].max()),
        "test_dates": (test_df["DATE_TIME"].min(), test_df["DATE_TIME"].max()),
        "plant2_dates": (clean_p2["DATE_TIME"].min(), clean_p2["DATE_TIME"].max()),
    }


# ======================================================================================
# 5. BASELINE BENCHMARKS (ZERO-RULE REGRESSION)
# ======================================================================================

def calculate_baseline_metrics(train_df, test_df, target_col=config.TARGET_COLUMN):
    """Compute Zero-Rule Regression Baseline benchmarks for comparison with ML models.
    
    1. Global Mean Baseline: Always predicts the training set mean AC power.
    2. Diurnal Solar Baseline: Predicts 0 during nighttime (IRRADIATION == 0) and
       daytime mean during daytime (IRRADIATION > 0).
    """
    y_train = train_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()

    # 1. Global Mean Predictor
    mean_val = float(np.mean(y_train))
    y_pred_mean = np.full_like(y_test, fill_value=mean_val)

    mae_mean = float(mean_absolute_error(y_test, y_pred_mean))
    mse_mean = float(mean_squared_error(y_test, y_pred_mean))
    rmse_mean = float(np.sqrt(mse_mean))
    r2_mean = float(r2_score(y_test, y_pred_mean))

    # 2. Diurnal Solar Baseline
    daylight_mask_train = train_df["IRRADIATION"] > 0
    daylight_mean = float(np.mean(y_train[daylight_mask_train])) if daylight_mask_train.any() else mean_val

    y_pred_diurnal = np.where(test_df["IRRADIATION"] > 0, daylight_mean, 0.0)

    mae_diurnal = float(mean_absolute_error(y_test, y_pred_diurnal))
    mse_diurnal = float(mean_squared_error(y_test, y_pred_diurnal))
    rmse_diurnal = float(np.sqrt(mse_diurnal))
    r2_diurnal = float(r2_score(y_test, y_pred_diurnal))

    return {
        "global_mean_baseline": {
            "strategy": "Global Mean AC_POWER Predictor",
            "train_mean_ac": round(mean_val, 4),
            "mae": round(mae_mean, 4),
            "rmse": round(rmse_mean, 4),
            "r2": round(r2_mean, 4),
            "description": "Always guessing the global training mean AC power provides the basic regression benchmark."
        },
        "diurnal_solar_baseline": {
            "strategy": "Diurnal Solar Baseline (Night=0, Day=Daylight Mean)",
            "daylight_mean_ac": round(daylight_mean, 4),
            "mae": round(mae_diurnal, 4),
            "rmse": round(rmse_diurnal, 4),
            "r2": round(r2_diurnal, 4),
            "description": "Incorporating the diurnal solar cycle provides a stronger domain-aware baseline benchmark."
        }
    }


# ======================================================================================
# 6. AUTOMATED TEXT REPORTS
# ======================================================================================

def save_cleaning_report(p1_raw, p1_clean, p1_dropped, p2_raw, p2_clean, p2_dropped):
    """Generate comprehensive data cleaning and outlier report."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.CLEANING_REPORT_PATH

    p1_outliers = detect_iqr_outliers(p1_clean)
    p2_outliers = detect_iqr_outliers(p2_clean)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  SOLAR POWER GENERATION - DATA CLEANING & OUTLIER REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. INGESTION & DEDUPLICATION SUMMARY:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Plant 1 Raw Merged Rows:   {len(p1_raw):,}\n")
        f.write(f"Plant 1 Cleaned Rows:      {len(p1_clean):,}\n")
        f.write(f"Plant 1 Rows Dropped/Dedup: {p1_dropped}\n\n")
        f.write(f"Plant 2 Raw Merged Rows:   {len(p2_raw):,}\n")
        f.write(f"Plant 2 Cleaned Rows:      {len(p2_clean):,}\n")
        f.write(f"Plant 2 Rows Dropped/Dedup: {p2_dropped}\n\n")

        f.write("2. PLANT 1 IQR OUTLIER DETECTION (1.5 * IQR Rule):\n")
        f.write("-" * 50 + "\n")
        for col, s in p1_outliers.items():
            f.write(f"Feature: {col:<22} | Q1={s['q1']:<8} Q3={s['q3']:<8} IQR={s['iqr']:<8}\n")
            f.write(f"  Lower Bound: {s['lower_bound']:<8} | Upper Bound: {s['upper_bound']:<8}\n")
            f.write(f"  Outliers:    {s['outlier_count']:,} ({s['outlier_percent']}%) | Range: [{s['min_val']} to {s['max_val']}]\n\n")

        f.write("3. PLANT 2 IQR OUTLIER DETECTION (1.5 * IQR Rule):\n")
        f.write("-" * 50 + "\n")
        for col, s in p2_outliers.items():
            f.write(f"Feature: {col:<22} | Q1={s['q1']:<8} Q3={s['q3']:<8} IQR={s['iqr']:<8}\n")
            f.write(f"  Lower Bound: {s['lower_bound']:<8} | Upper Bound: {s['upper_bound']:<8}\n")
            f.write(f"  Outliers:    {s['outlier_count']:,} ({s['outlier_percent']}%) | Range: [{s['min_val']} to {s['max_val']}]\n\n")

        f.write("4. DOMAIN-SPECIFIC CLEANING INSIGHT:\n")
        f.write("-" * 50 + "\n")
        f.write("* In solar datasets, nighttime zero-values for DC_POWER, AC_POWER, and IRRADIATION are physically\n")
        f.write("  valid and natural, not missing or corrupted data. They represent nighttime inactivity.\n")
        f.write("* Non-negative clipping ensures physical consistency (power and solar flux cannot be negative).\n")
        f.write("* Notice the difference in DC_POWER magnitude between Plant 1 and Plant 2. In Plant 1, DC_POWER is\n")
        f.write("  approx 10x AC_POWER, whereas in Plant 2, DC_POWER is approx 1x AC_POWER due to sensor unit scaling.\n")

    return str(report_path)


def save_split_report(split_info):
    """Save dataset split summary report."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.SPLIT_REPORT_PATH

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  SOLAR POWER GENERATION - DATASET SPLIT REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. SPLITTING METHODOLOGY:\n")
        f.write("-" * 50 + "\n")
        f.write("Method: Time-Aware Chronological Splitting (No Random Shuffle Leakage)\n")
        f.write("Plant 1 Split: 70% Train, 15% Validation, 15% Test\n")
        f.write("Plant 2 Split: 100% Held-Out Unseen Data (Cross-Plant Generalization Benchmark)\n\n")

        f.write("2. SPLIT SIZES & CHRONOLOGICAL RANGES:\n")
        f.write("-" * 50 + "\n")
        p1_tot = split_info["plant1_total"]
        f.write(f"Plant 1 Total Clean Records: {p1_tot:,}\n")
        f.write(f"  * Train Set:      {split_info['train_len']:,} rows ({split_info['train_len']/p1_tot*100:.1f}%) | {split_info['train_dates'][0]} to {split_info['train_dates'][1]}\n")
        f.write(f"  * Validation Set: {split_info['val_len']:,} rows ({split_info['val_len']/p1_tot*100:.1f}%) | {split_info['val_dates'][0]} to {split_info['val_dates'][1]}\n")
        f.write(f"  * Test Set:       {split_info['test_len']:,} rows ({split_info['test_len']/p1_tot*100:.1f}%) | {split_info['test_dates'][0]} to {split_info['test_dates'][1]}\n\n")

        p2_tot = split_info["plant2_unseen_len"]
        f.write(f"Plant 2 Total Clean Records: {p2_tot:,}\n")
        f.write(f"  * Unseen Test:    {p2_tot:,} rows (100.0%) | {split_info['plant2_dates'][0]} to {split_info['plant2_dates'][1]}\n\n")

        f.write("3. PERSISTENT SPLIT FILES SAVED:\n")
        f.write("-" * 50 + "\n")
        f.write(f"- Train Split:       {config.SPLIT_TRAIN_PATH}\n")
        f.write(f"- Validation Split:  {config.SPLIT_VAL_PATH}\n")
        f.write(f"- Test Split:        {config.SPLIT_TEST_PATH}\n")
        f.write(f"- Plant 2 Unseen:    {config.SPLIT_UNSEEN_PLANT2_PATH}\n")

    return str(report_path)


def save_baseline_report(baseline_metrics):
    """Save baseline performance metrics report."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.BASELINE_REPORT_PATH

    m = baseline_metrics["global_mean_baseline"]
    d = baseline_metrics["diurnal_solar_baseline"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  SOLAR POWER GENERATION - ZERO-RULE BASELINE BENCHMARK REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("TARGET REGRESSION VARIABLE: AC_POWER (kW)\n")
        f.write("-" * 50 + "\n\n")

        f.write("1. GLOBAL MEAN BASELINE:\n")
        f.write(f"   Strategy:         {m['strategy']}\n")
        f.write(f"   Train Mean Value: {m['train_mean_ac']:.4f} kW\n")
        f.write(f"   Test MAE:         {m['mae']:.4f} kW\n")
        f.write(f"   Test RMSE:        {m['rmse']:.4f} kW\n")
        f.write(f"   Test R2:          {m['r2']:.4f}\n")
        f.write(f"   Description:      {m['description']}\n\n")

        f.write("2. DIURNAL SOLAR BASELINE (DOMAIN-AWARE ZERO-RULE):\n")
        f.write(f"   Strategy:         {d['strategy']}\n")
        f.write(f"   Daylight Mean:    {d['daylight_mean_ac']:.4f} kW\n")
        f.write(f"   Test MAE:         {d['mae']:.4f} kW\n")
        f.write(f"   Test RMSE:        {d['rmse']:.4f} kW\n")
        f.write(f"   Test R2:          {d['r2']:.4f}\n")
        f.write(f"   Description:      {d['description']}\n\n")

        f.write("3. CONCLUSION & BENCHMARK THRESHOLDS FOR SUPERVISED MODELS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Any valid supervised machine learning model (Linear, Ridge, Lasso, Trees, Ensembles)\n")
        f.write(f"must significantly outperform the Diurnal Baseline (RMSE < {d['rmse']:.2f} kW, R2 > {d['r2']:.4f}).\n")

    return str(report_path)
