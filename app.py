"""Solar Power Generation ML Web Application.

Transferred from the Placement Prediction classroom architecture and adapted
specifically to multivariate solar power forecasting, regularized regression,
operational regime clustering, and cross-plant domain transfer.
"""

from pathlib import Path
import sys
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, session, url_for, jsonify)
from werkzeug.utils import secure_filename
import pandas as pd

import config
from src import flask_service
from src.data_utils import load_cleaned_plant

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
app.config["SECRET_KEY"] = "solar-power-telemetry-classroom-key-2026"
app.config["UPLOAD_FOLDER"] = PROJECT_ROOT / "uploads"
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def read_report_lines(filename):
    """Read diagnostic text report lines safely from Output/Report/."""
    file_path = config.REPORTS_DIR / filename
    if file_path.exists():
        return file_path.read_text(encoding="utf-8").splitlines()
    alt_path = config.PLOTS_DIR / filename
    if alt_path.exists():
        return alt_path.read_text(encoding="utf-8").splitlines()
    return []


def current_dataset_path():
    """Return Path to currently active dataset for session, or None for default Plant 1."""
    uploaded_path = session.get("dataset_path")
    if uploaded_path and Path(uploaded_path).is_file():
        return Path(uploaded_path)
    return None


def get_current_dataframe():
    """Load current active DataFrame based on session selection."""
    uploaded = current_dataset_path()
    if uploaded:
        try:
            return pd.read_csv(uploaded)
        except Exception:
            pass

    plant_id = session.get("plant_id", 1)
    return load_cleaned_plant(plant_id)


# =============================================================================
# STATIC PLOTS ROUTE (MATCHING CLASSROOM SERVE_PLOT)
# =============================================================================

@app.route("/plots/<path:filename>")
def serve_plot(filename):
    """Serve dynamically generated analysis and model plots from Output/plots directory."""
    return send_from_directory(config.PLOTS_DIR, filename)


# =============================================================================
# WEB ROUTES
# =============================================================================

@app.route("/")
def home_page():
    """First landing page displaying the large solar plant hero section and CTA."""
    stats = flask_service.get_dashboard_stats()
    return render_template(
        "home.html",
        stats=stats,
        dataset_name=session.get("dataset_name", "Plant 1 Solar Generation Facility"),
    )


@app.route("/load", methods=["GET", "POST"])
def load_page():
    """Dataset loader matching classroom load.html with Plant 1/2 switcher and CSV upload."""
    if request.method == "POST":
        dataset = request.files.get("dataset")
        if not dataset or not dataset.filename:
            flash("Choose a CSV file before uploading.", "error")
        elif not dataset.filename.lower().endswith(".csv"):
            flash("Only CSV files are supported.", "error")
        else:
            filename = f"{uuid4().hex}_{secure_filename(dataset.filename)}"
            saved_path = app.config["UPLOAD_FOLDER"] / filename
            dataset.save(saved_path)
            try:
                test_df = pd.read_csv(saved_path)
                if test_df.empty:
                    raise ValueError("Uploaded CSV is empty.")
            except Exception as error:
                saved_path.unlink(missing_ok=True)
                flash(f"The CSV could not be read: {error}", "error")
            else:
                session["dataset_path"] = str(saved_path)
                session["dataset_name"] = dataset.filename
                session.pop("plant_id", None)
                flash("Custom solar telemetry dataset uploaded successfully.", "success")
                return redirect(url_for("load_page"))

    # Check query param switcher (Plant 1 vs Plant 2)
    plant_param = request.args.get("plant")
    if plant_param in ["1", "2"]:
        session["plant_id"] = int(plant_param)
        session.pop("dataset_path", None)
        session["dataset_name"] = f"Plant {plant_param} Cleaned Solar Dataset"

    plant_id = session.get("plant_id", 1)
    df = get_current_dataframe()
    inverters_count = int(df["SOURCE_KEY_INVERTER"].nunique()) if "SOURCE_KEY_INVERTER" in df.columns else 22

    df_preview = df.head(15).copy()
    for col in df_preview.select_dtypes(include=["float64", "float32"]).columns:
        df_preview[col] = df_preview[col].round(3)

    return render_template(
        "load.html",
        dataset_name=session.get("dataset_name", f"Plant {plant_id} Solar Telemetry Dataset"),
        selected_plant=str(plant_id),
        shape=df.shape,
        columns=df.columns.tolist(),
        inverters_count=inverters_count,
        preview=df_preview.to_dict(orient="records"),
    )


@app.route("/eda")
@app.route("/eda/<section>")
def eda_page(section="overview"):
    """Exploratory Data Analysis dashboard matching classroom eda.html."""
    plant_id = session.get("plant_id", 1)
    df = get_current_dataframe()

    eda_stats = flask_service.get_solar_eda_stats(df=df, plant_id=plant_id)
    eda_plots = flask_service.get_eda_plots()

    plot_list = []
    for info in eda_plots.values():
        plot_list.append({
            "url": info["url"],
            "title": info["title"],
            "desc": info["desc"],
        })

    return render_template(
        "eda.html",
        dataset_name=session.get("dataset_name", f"Plant {plant_id} Cleaned Solar Dataset"),
        section=section,
        overview=eda_stats["overview"],
        univariate=eda_stats["univariate"],
        bivariate=eda_stats["bivariate"],
        correlation=eda_stats["correlation"],
        plots=plot_list,
    )


@app.route("/feature-engg", methods=["GET", "POST"])
def feature_engg_page():
    """Feature engineering, regularization benchmark, and live 10-feature AC power predictor."""
    fe_info = flask_service.get_feature_engineering_info()
    linear_data = flask_service.get_model_results("linear")
    reg_benchmark = flask_service.get_regularization_benchmark()
    presets = flask_service.get_presets()

    # Form defaults (Peak solar noon preset)
    pred_inputs = dict(presets["peak_noon"]["values"])
    chosen_model = "linear"
    prediction_result = None
    error_message = None

    req_form = request.form if request.method == "POST" else {}
    active_folder = request.args.get("folder", req_form.get("active_folder", "predictor-output"))

    if request.method == "POST":
        chosen_model = req_form.get("model_type", "linear")
        for feat in config.NUMERIC_FEATURES:
            if feat in req_form:
                pred_inputs[feat] = req_form.get(feat)

        try:
            prediction_result = flask_service.predict_ac_power(pred_inputs, model_name=chosen_model)
        except Exception as e:
            error_message = str(e)

        active_folder = "predictor-output"

    return render_template(
        "feature_engg.html",
        dataset_name=session.get("dataset_name", "Plant 1 Solar Generation Facility"),
        cleaned_shape=(68774, 18),
        duplicate_count=0,
        feature_columns=list(config.NUMERIC_FEATURES),
        scaler_outputs=fe_info["scaler_comparison"],
        linear_model_data=linear_data,
        reg_benchmark=reg_benchmark,
        presets=presets,
        pred_inputs=pred_inputs,
        chosen_model=chosen_model,
        prediction_result=prediction_result,
        error_message=error_message,
        active_folder=active_folder,
    )


@app.route("/linear-regression", methods=["GET", "POST"])
def linear_regression_page():
    """Shortcut matching classroom linear-regression redirect."""
    return redirect(url_for("feature_engg_page", folder="linear-output"))


@app.route("/regularization", methods=["GET", "POST"])
def regularization_page():
    """Shortcut matching classroom regularization redirect."""
    return redirect(url_for("feature_engg_page", folder="regularization-output"))


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    """Dedicated prediction shortcut directly opening the predictor folder."""
    return redirect(url_for("feature_engg_page", folder="predictor-output"))


@app.route("/unsupervised", methods=["GET", "POST"])
@app.route("/clustering", methods=["GET", "POST"])
def unsupervised_learning_page():
    """Unsupervised learning matching classroom clustering.html."""
    active_folder = request.args.get("folder", "kmeans-output")

    return render_template(
        "clustering.html",
        dataset_name=session.get("dataset_name", "Plant 1 Solar Telemetry Dataset"),
        cleaned_shape=(68774, 18),
        duplicate_count=0,
        feature_columns=list(config.NUMERIC_FEATURES),
        active_folder=active_folder,
        km_report_lines=read_report_lines("kmeans_regime_report.txt"),
        hier_report_lines=read_report_lines("hierarchical_clustering_report.txt"),
        dbscan_report_lines=read_report_lines("dbscan_clustering_report.txt"),
        pca_report_lines=read_report_lines("pca_analysis_report.txt"),
        anomaly_report_lines=read_report_lines("solar_anomaly_detection_report.txt"),
        cluster_aug_report_lines=read_report_lines("cluster_feature_augmentation_report.txt"),
    )


@app.route("/generalization")
def generalization_page():
    """Cross-plant transfer analysis and domain shift (Plant 1 -> Plant 2)."""
    gen_data = flask_service.get_generalization_data()
    return render_template(
        "generalization.html",
        gen=gen_data,
        dataset_name="Plant 1 to Plant 2 Transfer Benchmark",
    )


@app.route("/evaluation", methods=["GET"])
def evaluation_page():
    """Model evaluation and validation suite matching classroom evaluation.html."""
    active_folder = request.args.get("folder", "leakage-output")

    reports = {
        "leakage": read_report_lines("time_series_cv_report.txt"),
        "nested_cv": read_report_lines("nested_cross_validation_report.txt"),
        "search": read_report_lines("grid_vs_random_search_report.txt"),
        "learning_curves": read_report_lines("learning_validation_curves_report.txt"),
        "regression": read_report_lines("regression_metrics_and_robustness_report.txt"),
        "significance": read_report_lines("statistical_significance_report.txt"),
    }

    return render_template(
        "evaluation.html",
        dataset_name=session.get("dataset_name", "Solar Telemetry Splits"),
        cleaned_shape=(68774, 18),
        duplicate_count=0,
        active_folder=active_folder,
        reports=reports,
    )


# =============================================================================
# JSON REST APIS
# =============================================================================

@app.route("/api/health", methods=["GET"])
def api_health():
    """Operational health check and cache status."""
    return jsonify({
        "status": "healthy",
        "service": "Solar Power Prediction and Generalization",
        "models_count": 4,
        "target": config.TARGET_COLUMN,
        "features": list(config.NUMERIC_FEATURES),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Strict JSON API prediction endpoint accepting {model: '...', features: {...}}."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Content-Type must be 'application/json'"}), 415

    payload = request.get_json(silent=True) or {}
    model_name = payload.get("model", "linear")
    feature_values = payload.get("features")

    if not feature_values or not isinstance(feature_values, dict):
        return jsonify({"status": "error", "message": "Missing required 'features' dictionary."}), 400

    try:
        res = flask_service.predict_ac_power(feature_values, model_name=model_name)
        return jsonify({"status": "success", "data": res}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# =============================================================================
# APPLICATION ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Starting Solar Power Prediction and Generalization Flask Server")
    print("Serving on http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=True)
