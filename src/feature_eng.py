"""Feature engineering and scaling utilities for Solar Power Generation.

Course: Machine Learning (25SC2107E) - Module 2
Applies StandardScaler, MinMaxScaler, and RobustScaler.
Critically enforces strict training-serving boundary:
All scaler statistics are fitted ONLY on training data to prevent data leakage.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import config


def get_feature_matrices(train_df, val_df=None, test_df=None, unseen_df=None, feature_cols=None):
    """Extract feature matrices X and target vectors y for all splits."""
    feats = feature_cols or config.NUMERIC_FEATURES
    available_feats = [c for c in feats if c in train_df.columns]

    X_train = train_df[available_feats].copy()
    y_train = train_df[config.TARGET_COLUMN].to_numpy()

    splits = {
        "features": available_feats,
        "X_train": X_train,
        "y_train": y_train,
    }

    if val_df is not None:
        splits["X_val"] = val_df[available_feats].copy()
        splits["y_val"] = val_df[config.TARGET_COLUMN].to_numpy()

    if test_df is not None:
        splits["X_test"] = test_df[available_feats].copy()
        splits["y_test"] = test_df[config.TARGET_COLUMN].to_numpy()

    if unseen_df is not None:
        splits["X_unseen"] = unseen_df[available_feats].copy()
        splits["y_unseen"] = unseen_df[config.TARGET_COLUMN].to_numpy()

    return splits


def scale_features(X_train, X_val=None, X_test=None, X_unseen=None, scaler_type="standard"):
    """Scale feature matrices using statistics learned strictly from training data.
    
    Supported types:
      - 'standard': z = (x - mean) / std (essential for Ridge/Lasso/Gradient Descent)
      - 'minmax':   x' = (x - min) / (max - min) (bounds values into [0, 1])
      - 'robust':   x' = (x - median) / IQR (resilient to extreme sensor spikes)
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'standard', 'minmax', or 'robust'.")

    # Fit ONLY on training data
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    result = {
        "scaler": scaler,
        "scaler_type": scaler_type,
        "X_train_scaled": X_train_scaled,
    }

    if X_val is not None:
        result["X_val_scaled"] = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

    if X_test is not None:
        result["X_test_scaled"] = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

    if X_unseen is not None:
        result["X_unseen_scaled"] = pd.DataFrame(
            scaler.transform(X_unseen),
            columns=X_unseen.columns,
            index=X_unseen.index
        )

    return result


def compare_scalers(X_train, y_train, X_test, y_test):
    """Empirical comparison of linear regression performance across different scaling methods."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    results = []
    for s_type in ["standard", "minmax", "robust"]:
        scaled = scale_features(X_train, X_test=X_test, scaler_type=s_type)
        model = LinearRegression()
        model.fit(scaled["X_train_scaled"], y_train)
        preds = model.predict(scaled["X_test_scaled"])

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        # Check condition number of X^T X
        X_mat = scaled["X_train_scaled"].to_numpy()
        cond_num = float(np.linalg.cond(X_mat.T @ X_mat))

        results.append({
            "scaler": s_type,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "condition_number": round(cond_num, 2),
            "intercept": round(float(model.intercept_), 4)
        })

    return results
