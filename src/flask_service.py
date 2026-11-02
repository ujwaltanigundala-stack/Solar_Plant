"""Service layer bridging Flask routes to existing ML modules.

Handles model caching, data loading, and result formatting.
Does NOT contain any ML algorithms — delegates entirely to existing modules.
The fitted scaler from training is saved and reused for prediction (no recreation).
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

import config
from src.data_utils import load_cleaned_plant, detect_iqr_outliers
from src.feature_eng import get_feature_matrices, scale_features, compare_scalers
from src.linear_regression import train_linear_models
from src.regularization import train_regularized_models
from src.EDA import generate_all_plots

# Cache directory for trained models
MODELS_DIR = config.OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache (populated once per app lifetime)
_cache = {}


# =============================================================================
# MODEL TRAINING & CACHING
# =============================================================================

def _get_or_train_linear():
    """Get linear model results, training only if not yet cached."""
    cache_path = MODELS_DIR / "linear_results.joblib"
    if "linear" in _cache:
        return _cache["linear"]
    if cache_path.exists():
        data = joblib.load(cache_path)
        _cache["linear"] = data
        return data

    # Train using existing function (loads splits internally when called with no args)
    data = train_linear_models()

    # Extract serializable parts (exclude large numpy test arrays)
    save_data = {
        "model": data["model"],
        "scaler": data["scaler"],  # exact scaler fitted during training
        "features": data["features"],
        "intercept": data["intercept"],
        "coefficients": data["coefficients"],
        "train_metrics": data["train_metrics"],
        "val_metrics": data["val_metrics"],
        "test_metrics": data["test_metrics"],
        "unseen_metrics": data["unseen_metrics"],
        "gd_results": {
            "intercept": data["gd_results"]["intercept"],
            "cost_history": data["gd_results"]["cost_history"],
            "initial_cost": data["gd_results"]["initial_cost"],
            "final_cost": data["gd_results"]["final_cost"],
            "epochs": data["gd_results"]["epochs"],
            "learning_rate": data["gd_results"]["learning_rate"],
            "elapsed_sec": data["gd_results"]["elapsed_sec"],
        },
        "gd_test_metrics": data["gd_test_metrics"],
        "baseline_dc": {
            "metrics": data["baseline_dc"]["metrics"],
            "slope": data["baseline_dc"]["slope"],
            "intercept": data["baseline_dc"]["intercept"],
        },
        "baseline_irr": {
            "metrics": data["baseline_irr"]["metrics"],
            "slope": data["baseline_irr"]["slope"],
            "intercept": data["baseline_irr"]["intercept"],
        },
    }
    joblib.dump(save_data, cache_path)
    _cache["linear"] = save_data
    return save_data


def _get_or_train_regularized():
    """Get regularized model results, training only if not yet cached."""
    cache_path = MODELS_DIR / "regularized_results.joblib"
    if "regularized" in _cache:
        return _cache["regularized"]
    if cache_path.exists():
        data = joblib.load(cache_path)
        _cache["regularized"] = data
        return data

    # Train using existing function
    data = train_regularized_models()

    save_data = {
        "scaler": data["scaler"],  # exact scaler fitted during training
        "features": data["features"],
        "best_alpha_ridge": data["best_alpha_ridge"],
        "best_alpha_lasso": data["best_alpha_lasso"],
        "best_alpha_elastic": data["best_alpha_elastic"],
        "best_l1_ratio_elastic": data["best_l1_ratio_elastic"],
        "condition_number_unreg": data["condition_number_unreg"],
        "condition_number_ridge": data["condition_number_ridge"],
        "results": {},
    }

    for name, r in data["results"].items():
        save_data["results"][name] = {
            "model": r["model"],
            "penalty": r["penalty"],
            "alpha": r["alpha"],
            "l1_ratio": r["l1_ratio"],
            "description": r["description"],
            "intercept": r["intercept"],
            "coefficients": r["coefficients"],
            "zero_coefficients_count": r["zero_coefficients_count"],
            "retained_features_count": r["retained_features_count"],
            "train_metrics": r["train_metrics"],
            "val_metrics": r["val_metrics"],
            "test_metrics": r["test_metrics"],
            "unseen_metrics": r["unseen_metrics"],
        }

    joblib.dump(save_data, cache_path)
    _cache["regularized"] = save_data
    return save_data


# =============================================================================
# DASHBOARD
# =============================================================================

def get_dashboard_stats():
    """Gather summary statistics for the dashboard from actual data."""
    p1 = load_cleaned_plant(1)
    p2 = load_cleaned_plant(2)

    return {
        "plant1_samples": len(p1),
        "plant2_samples": len(p2),
        "num_features": len(config.NUMERIC_FEATURES),
        "feature_names": list(config.NUMERIC_FEATURES),
        "target_variable": config.TARGET_COLUMN,
        "num_models": 4,
        "model_names": ["OLS Linear Regression", "Ridge Regression",
                        "Lasso Regression", "Elastic Net"],
        "plant1_date_range": (str(p1["DATE_TIME"].min()),
                              str(p1["DATE_TIME"].max())),
        "plant2_date_range": (str(p2["DATE_TIME"].min()),
                              str(p2["DATE_TIME"].max())),
        "plant1_inverters": int(p1["SOURCE_KEY_INVERTER"].nunique()),
        "plant2_inverters": int(p2["SOURCE_KEY_INVERTER"].nunique()),
    }


# =============================================================================
# DATASET
# =============================================================================

def get_dataset_info(plant_id):
    """Get dataset information for a given plant from actual data."""
    df = load_cleaned_plant(plant_id)
    numeric_cols = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    all_display = numeric_cols + [config.TARGET_COLUMN]

    stats = df[all_display].describe().round(4)
    stats_dict = {}
    for col in stats.columns:
        stats_dict[col] = {
            idx: float(val) for idx, val in stats[col].items()
        }

    sample = df[all_display].head(10).round(4).to_dict(orient="records")

    return {
        "plant_id": plant_id,
        "rows": len(df),
        "columns": len(df.columns),
        "feature_names": numeric_cols,
        "target": config.TARGET_COLUMN,
        "all_columns": list(df.columns),
        "date_range": (str(df["DATE_TIME"].min()), str(df["DATE_TIME"].max())),
        "inverters": int(df["SOURCE_KEY_INVERTER"].nunique()),
        "stats": stats_dict,
        "stat_indices": list(stats.index),
        "sample": sample,
        "missing_values": {
            col: int(df[col].isnull().sum()) for col in all_display
        },
    }


# =============================================================================
# EDA
# =============================================================================

def get_eda_plots():
    """Get existing EDA plot paths, generating if needed."""
    expected = {
        "ac_power_distribution": {
            "file": "ac_power_distribution.png",
            "title": "AC Power Distribution",
            "desc": "Distribution of AC output power for Plant 1, overall and filtered to daylight hours."
        },
        "dc_vs_ac_power": {
            "file": "dc_vs_ac_power.png",
            "title": "DC Power vs AC Power",
            "desc": "Inverter conversion curve showing DC-to-AC relationship, colored by module temperature."
        },
        "irradiation_vs_ac_power": {
            "file": "irradiation_vs_ac_power.png",
            "title": "Irradiation vs AC Power",
            "desc": "Relationship between solar irradiation intensity and AC output power."
        },
        "diurnal_cycle_hourly": {
            "file": "diurnal_cycle_hourly.png",
            "title": "Diurnal Solar Generation Cycle",
            "desc": "Hourly profile of AC power and irradiation across 24 hours, showing the solar generation window."
        },
        "temperature_relationships": {
            "file": "temperature_relationships.png",
            "title": "Temperature Relationships",
            "desc": "Ambient vs module temperature scatter plot, colored by irradiation intensity."
        },
        "feature_correlation_heatmap": {
            "file": "feature_correlation_heatmap.png",
            "title": "Feature Correlation Heatmap",
            "desc": "Pearson correlation matrix of all numeric features with AC_POWER target."
        },
        "plant1_vs_plant2_comparison": {
            "file": "plant1_vs_plant2_comparison.png",
            "title": "Plant 1 vs Plant 2 Comparison",
            "desc": "Cross-plant distribution comparison highlighting the ~10x DC_POWER scale divergence."
        },
    }

    # Check if plots exist, generate if missing
    all_exist = all(
        (config.PLOTS_DIR / info["file"]).exists()
        for info in expected.values()
    )
    if not all_exist:
        generate_all_plots()

    result = {}
    for key, info in expected.items():
        if (config.PLOTS_DIR / info["file"]).exists():
            result[key] = {
                "url": info["file"],
                "title": info["title"],
                "desc": info["desc"],
            }
    return result


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def get_feature_engineering_info():
    """Get feature engineering pipeline details from the existing implementation."""
    train_df = pd.read_csv(config.SPLIT_TRAIN_PATH)
    test_df = pd.read_csv(config.SPLIT_TEST_PATH)
    matrices = get_feature_matrices(train_df, test_df=test_df)

    scaler_results = compare_scalers(
        matrices["X_train"], matrices["y_train"],
        matrices["X_test"], matrices["y_test"]
    )

    return {
        "numeric_features": list(config.NUMERIC_FEATURES),
        "core_physical": list(config.CORE_PHYSICAL_FEATURES),
        "temporal_features": list(config.TEMPORAL_FEATURES),
        "engineered_features": ["TEMP_DIFF", "IS_DAYLIGHT"],
        "target": config.TARGET_COLUMN,
        "scaler_comparison": scaler_results,
        "feature_count": len(config.NUMERIC_FEATURES),
    }


# =============================================================================
# INDIVIDUAL MODEL RESULTS
# =============================================================================

def get_model_results(model_name):
    """Get results for a specific model.

    Args:
        model_name: 'linear', 'ridge', 'lasso', 'elasticnet'
    """
    if model_name == "linear":
        data = _get_or_train_linear()
        return {
            "name": "OLS Linear Regression",
            "short_name": "OLS",
            "description": (
                "Ordinary Least Squares via Normal Equation. "
                "Minimizes the sum of squared residuals without regularization."
            ),
            "penalty": "None (λ = 0)",
            "alpha": 0.0,
            "intercept": data["intercept"],
            "coefficients": data["coefficients"],
            "features": data["features"],
            "train_metrics": data["train_metrics"],
            "val_metrics": data["val_metrics"],
            "test_metrics": data["test_metrics"],
            "unseen_metrics": data["unseen_metrics"],
            "gd_results": data.get("gd_results", {}),
            "gd_test_metrics": data.get("gd_test_metrics", {}),
            "baseline_dc": data.get("baseline_dc", {}),
            "baseline_irr": data.get("baseline_irr", {}),
            "plots": _get_linear_plots(),
            "is_linear": True,
        }
    else:
        reg_data = _get_or_train_regularized()
        key_map = {"ridge": "Ridge", "lasso": "Lasso", "elasticnet": "ElasticNet"}
        display_map = {
            "ridge": "Ridge Regression",
            "lasso": "Lasso Regression",
            "elasticnet": "Elastic Net Regression",
        }
        key = key_map.get(model_name)
        if key is None or key not in reg_data["results"]:
            raise ValueError(f"Unknown model: {model_name}")
        r = reg_data["results"][key]

        return {
            "name": display_map[model_name],
            "short_name": key,
            "description": r["description"],
            "penalty": r["penalty"],
            "alpha": r["alpha"],
            "l1_ratio": r.get("l1_ratio", 0.0),
            "intercept": r["intercept"],
            "coefficients": [
                {"feature": f, "standardized_weight": w}
                for f, w in r["coefficients"].items()
            ],
            "features": reg_data["features"],
            "retained_features_count": r["retained_features_count"],
            "zero_coefficients_count": r["zero_coefficients_count"],
            "train_metrics": r["train_metrics"],
            "val_metrics": r["val_metrics"],
            "test_metrics": r["test_metrics"],
            "unseen_metrics": r["unseen_metrics"],
            "best_alpha": reg_data.get(f"best_alpha_{model_name}", r["alpha"]),
            "condition_number_unreg": reg_data.get("condition_number_unreg"),
            "condition_number_ridge": reg_data.get("condition_number_ridge"),
            "plots": _get_regularization_plots(),
            "is_linear": False,
        }


def _get_linear_plots():
    """Get linear model diagnostic plot paths (only those that exist)."""
    mapping = {
        "actual_vs_predicted": "linear_actual_vs_predicted.png",
        "residuals": "linear_residuals_plot.png",
        "gd_convergence": "gradient_descent_cost_curve.png",
    }
    return {
        name: fname for name, fname in mapping.items()
        if (config.PLOTS_DIR / fname).exists()
    }


def _get_regularization_plots():
    """Get regularization diagnostic plot paths (only those that exist)."""
    mapping = {
        "coefficients_comparison": "regularization_coefficients_comparison.png",
        "alpha_path": "regularization_alpha_path.png",
        "cross_plant_transfer": "cross_plant_transfer_comparison.png",
    }
    return {
        name: fname for name, fname in mapping.items()
        if (config.PLOTS_DIR / fname).exists()
    }


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def get_model_comparison():
    """Get comparison data for all implemented models."""
    linear = _get_or_train_linear()
    reg = _get_or_train_regularized()

    comparison = []

    # OLS from linear module
    comparison.append({
        "name": "OLS Linear Regression",
        "short_name": "OLS",
        "alpha": 0.0,
        "penalty": "None",
        "plant1_train": linear["train_metrics"],
        "plant1_val": linear["val_metrics"],
        "plant1_test": linear["test_metrics"],
        "plant2_unseen": linear["unseen_metrics"],
        "retained_features": len(linear["features"]),
        "total_features": len(linear["features"]),
    })

    # Regularized models
    display = {
        "Ridge": "Ridge Regression",
        "Lasso": "Lasso Regression",
        "ElasticNet": "Elastic Net Regression",
    }
    for name in ["Ridge", "Lasso", "ElasticNet"]:
        r = reg["results"][name]
        comparison.append({
            "name": display[name],
            "short_name": name,
            "alpha": r["alpha"],
            "penalty": r["penalty"],
            "plant1_train": r["train_metrics"],
            "plant1_val": r["val_metrics"],
            "plant1_test": r["test_metrics"],
            "plant2_unseen": r["unseen_metrics"],
            "retained_features": r["retained_features_count"],
            "total_features": len(reg["features"]),
        })

    # Find best Plant 1 Test RMSE (separately from best Plant 2)
    best_p1_idx = min(range(len(comparison)),
                      key=lambda i: comparison[i]["plant1_test"]["rmse"])
    best_p2_idx = min(range(len(comparison)),
                      key=lambda i: comparison[i]["plant2_unseen"]["rmse"])

    for i, c in enumerate(comparison):
        c["is_best_plant1"] = (i == best_p1_idx)
        c["is_best_plant2"] = (i == best_p2_idx)

    return {
        "models": comparison,
        "plots": _get_regularization_plots(),
    }


# =============================================================================
# GENERALIZATION
# =============================================================================

def get_generalization_data():
    """Get Plant 1 → Plant 2 generalization analysis data."""
    comp = get_model_comparison()

    # Compute performance gaps
    for m in comp["models"]:
        p1_rmse = m["plant1_test"]["rmse"]
        p2_rmse = m["plant2_unseen"]["rmse"]
        m["rmse_gap"] = round(p2_rmse - p1_rmse, 4)
        p1_r2 = m["plant1_test"]["r2"]
        p2_r2 = m["plant2_unseen"]["r2"]
        m["r2_gap"] = round(p2_r2 - p1_r2, 4)

    return {
        "models": comp["models"],
        "plots": {
            **_get_regularization_plots(),
            **_get_linear_plots(),
        },
        "features": list(config.NUMERIC_FEATURES),
        "target": config.TARGET_COLUMN,
    }


# =============================================================================
# PREDICTION
# =============================================================================

def predict_ac_power(feature_values, model_name="linear"):
    """Make a prediction using the specified model.

    Args:
        feature_values: dict with keys matching config.NUMERIC_FEATURES
        model_name: 'linear', 'ridge', 'lasso', 'elasticnet'

    Returns:
        dict with prediction value and model info

    Raises:
        ValueError: if any required feature is missing or non-numeric
    """
    # Determine model and scaler
    if model_name == "linear":
        data = _get_or_train_linear()
        model = data["model"]
        scaler = data["scaler"]
        features = data["features"]
    elif model_name in ("ridge", "lasso", "elasticnet"):
        reg_data = _get_or_train_regularized()
        key_map = {"ridge": "Ridge", "lasso": "Lasso", "elasticnet": "ElasticNet"}
        key = key_map[model_name]
        model = reg_data["results"][key]["model"]
        scaler = reg_data["scaler"]  # exact scaler from training
        features = reg_data["features"]
    else:
        raise ValueError(f"Unknown model: '{model_name}'. "
                         f"Choose from: linear, ridge, lasso, elasticnet")

    # Strict validation: every feature must be present and numeric
    missing = [f for f in features if f not in feature_values]
    if missing:
        raise ValueError(
            f"Missing required feature(s): {', '.join(missing)}. "
            f"All {len(features)} features are required for prediction."
        )

    validated = {}
    for f in features:
        val = feature_values[f]
        try:
            validated[f] = float(val)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid numeric value for '{f}': '{val}'. "
                f"Must be a valid number."
            )

    # Build feature vector in the exact column order used during training
    feature_vector = [validated[f] for f in features]

    # Create DataFrame with correct column names (required by sklearn scaler)
    X_input = pd.DataFrame([feature_vector], columns=features)

    # Scale using the EXACT fitted scaler from training
    X_scaled = pd.DataFrame(
        scaler.transform(X_input),
        columns=features,
    )

    # Predict
    prediction = float(model.predict(X_scaled)[0])

    display_names = {
        "linear": "OLS Linear Regression",
        "ridge": "Ridge Regression",
        "lasso": "Lasso Regression",
        "elasticnet": "Elastic Net Regression",
    }

    return {
        "prediction": round(prediction, 2),
        "model_name": display_names[model_name],
        "model_key": model_name,
        "features_used": features,
        "input_values": validated,
    }


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def clear_model_cache():
    """Clear cached models to force retraining on next request."""
    _cache.clear()
    for f in MODELS_DIR.glob("*.joblib"):
        f.unlink()
    return True


# =============================================================================
# CLASSROOM DASHBOARD ADAPTATION HELPERS
# =============================================================================

def get_presets():
    """Return pre-configured realistic telemetry presets for interactive prediction."""
    return {
        "peak_noon": {
            "name": "Peak Solar Noon",
            "badge": "Maximum Solar",
            "desc": "Clear sky zenith at 12:30 PM: maximal solar irradiance with elevated module temperatures.",
            "values": {
                "DC_POWER": 11500.0,
                "IRRADIATION": 0.98,
                "AMBIENT_TEMPERATURE": 33.5,
                "MODULE_TEMPERATURE": 55.2,
                "HOUR": 12,
                "MINUTE": 30,
                "DAY": 15,
                "DAY_OF_WEEK": 2,
                "TEMP_DIFF": 21.7,
                "IS_DAYLIGHT": 1,
            },
        },
        "overcast_morning": {
            "name": "Overcast Morning",
            "badge": "Diffuse Light",
            "desc": "Morning ramp at 9:15 AM under overcast skies with diffuse sunlight and moderate heating.",
            "values": {
                "DC_POWER": 3200.0,
                "IRRADIATION": 0.28,
                "AMBIENT_TEMPERATURE": 25.1,
                "MODULE_TEMPERATURE": 31.4,
                "HOUR": 9,
                "MINUTE": 15,
                "DAY": 20,
                "DAY_OF_WEEK": 4,
                "TEMP_DIFF": 6.3,
                "IS_DAYLIGHT": 1,
            },
        },
        "sunset_ramp": {
            "name": "Sunset Ramp-Down",
            "badge": "Declining Solar",
            "desc": "Dusk transition at 6:15 PM with decaying irradiation and cooling panel surfaces.",
            "values": {
                "DC_POWER": 650.0,
                "IRRADIATION": 0.05,
                "AMBIENT_TEMPERATURE": 28.2,
                "MODULE_TEMPERATURE": 29.5,
                "HOUR": 18,
                "MINUTE": 15,
                "DAY": 5,
                "DAY_OF_WEEK": 5,
                "TEMP_DIFF": 1.3,
                "IS_DAYLIGHT": 1,
            },
        },
        "nocturnal_idle": {
            "name": "Nocturnal Idle",
            "badge": "Zero Irradiance",
            "desc": "Nighttime standby at 2:00 AM: zero irradiance, ambient thermal equilibrium, zero inverter generation.",
            "values": {
                "DC_POWER": 0.0,
                "IRRADIATION": 0.0,
                "AMBIENT_TEMPERATURE": 21.0,
                "MODULE_TEMPERATURE": 19.8,
                "HOUR": 2,
                "MINUTE": 0,
                "DAY": 12,
                "DAY_OF_WEEK": 1,
                "TEMP_DIFF": -1.2,
                "IS_DAYLIGHT": 0,
            },
        },
    }


def get_solar_eda_stats(df=None, plant_id=1):
    """Compute parametric and non-parametric exploratory stats matching classroom EDA dashboard."""
    if df is None:
        df = load_cleaned_plant(plant_id)

    num_cols = [c for c in config.NUMERIC_FEATURES if c in df.columns] + [config.TARGET_COLUMN]
    desc = df[num_cols].describe().round(3)
    skews = df[num_cols].skew().round(3)

    numeric_table = []
    for col in num_cols:
        numeric_table.append({
            "column": col,
            "mean": float(desc[col]["mean"]),
            "std": float(desc[col]["std"]),
            "min": float(desc[col]["min"]),
            "50%": float(desc[col]["50%"]),
            "max": float(desc[col]["max"]),
            "skew": float(skews.get(col, 0.0)),
        })

    corrs = df[num_cols].corr()[config.TARGET_COLUMN].sort_values(ascending=False).round(4).to_dict()

    iqr_outliers = {}
    for col in num_cols:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        outliers_count = int(((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum())
        iqr_outliers[col] = outliers_count

    return {
        "overview": {
            "rows": len(df),
            "cols": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "anomaly_rate": round(float(iqr_outliers.get("DC_POWER", 0) / len(df) * 100), 2),
            "data_types": [{"column": col, "dtype": str(df[col].dtype)} for col in df.columns],
            "missing_table": [{"column": col, "count": int(df[col].isna().sum()), "percent": round(float(df[col].isna().mean() * 100), 2)} for col in df.columns if df[col].isna().sum() > 0],
            "columns": list(df.columns),
            "head_rows": df[num_cols].head(10).round(3).to_dict(orient="records"),
            "tail_rows": df[num_cols].tail(10).round(3).to_dict(orient="records"),
        },
        "univariate": {
            "numeric_table": numeric_table,
            "categorical_freq": {
                "PLANT_ID": {str(k): int(v) for k, v in df["PLANT_ID"].value_counts().items()} if "PLANT_ID" in df.columns else {},
                "IS_DAYLIGHT": {("Daylight (1)" if k == 1 else "Night (0)"): int(v) for k, v in df["IS_DAYLIGHT"].value_counts().items()} if "IS_DAYLIGHT" in df.columns else {},
            },
        },
        "bivariate": {
            "outliers": iqr_outliers,
        },
        "correlation": {
            "top_correlations": corrs,
        },
    }


def get_regularization_benchmark():
    """Build regression comparison matrix matching classroom Session 19 format."""
    lin = _get_or_train_linear()
    reg = _get_or_train_regularized()

    comparison_table = []
    comparison_table.append({
        "Model": "OLS Linear Regression",
        "Penalty": "None (lambda = 0)",
        "Best Alpha (lambda)": 0.0,
        "L1 Ratio (alpha)": 0.0,
        "Train R2": round(lin["train_metrics"]["r2"], 4),
        "Test R2": round(lin["test_metrics"]["r2"], 4),
        "Test RMSE": round(lin["test_metrics"]["rmse"], 3),
        "Zero Coefs": 0,
        "Active Coefs": len(lin["features"]),
        "Overfitting Gap": round(lin["train_metrics"]["r2"] - lin["test_metrics"]["r2"], 4),
    })

    display_map = {"Ridge": "Ridge Regression (L2)", "Lasso": "Lasso Regression (L1)", "ElasticNet": "Elastic Net"}
    for name in ["Ridge", "Lasso", "ElasticNet"]:
        r = reg["results"][name]
        comparison_table.append({
            "Model": display_map[name],
            "Penalty": r["penalty"],
            "Best Alpha (lambda)": r["alpha"],
            "L1 Ratio (alpha)": r.get("l1_ratio", 0.0),
            "Train R2": round(r["train_metrics"]["r2"], 4),
            "Test R2": round(r["test_metrics"]["r2"], 4),
            "Test RMSE": round(r["test_metrics"]["rmse"], 3),
            "Zero Coefs": r["zero_coefficients_count"],
            "Active Coefs": r["retained_features_count"],
            "Overfitting Gap": round(r["train_metrics"]["r2"] - r["test_metrics"]["r2"], 4),
        })

    best_hyperparameters = {
        "ridge_alpha": reg.get("best_alpha_ridge", 0.0001),
        "lasso_alpha": reg.get("best_alpha_lasso", 1.0),
        "elastic_alpha": reg.get("best_alpha_elastic", 0.001),
        "elastic_l1_ratio": reg.get("best_l1_ratio_elastic", 0.7),
    }

    return {
        "comparison_table": comparison_table,
        "best_hyperparameters": best_hyperparameters,
        "features": lin["features"],
        "linear_coefficients": lin["coefficients"],
        "regularized_results": reg["results"],
    }

