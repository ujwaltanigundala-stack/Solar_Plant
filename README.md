# Solar Power Generation Telemetry & Predictive Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20Optimization-blueviolet.svg)](https://optuna.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

An end-to-end Machine Learning pipeline and interactive Flask web analytics dashboard designed for multivariate solar power forecasting, operational regime clustering, regularized regression, and cross-plant domain transfer.

---

## 📌 Project Overview

This project provides an end-to-end ML engineering system for solar photovoltaic power plant telemetry. Combining power generation telemetry (inverter DC/AC output, daily/total yield) with ambient and module weather sensor readings, the platform addresses real-world machine learning challenges:

- **Data Ingestion & Quality Auditing:** Multi-sensor alignment, chronological time-aware train/val/test splits, IQR outlier treatment, and physical bounds validation.
- **Supervised Regression & Regularization:** Closed-form OLS, vectorized Batch Gradient Descent from scratch, Ridge ($L_2$), Lasso ($L_1$), and ElasticNet regression with cross-validated hyperparameter shrinkage.
- **Unsupervised Regime Discovery & Anomaly Detection:** Operational regime segmentation using K-Means (Lloyd vs. MiniBatch), Agglomerative Hierarchical Clustering, DBSCAN density manifolds, PCA biplots, non-linear manifold embeddings (t-SNE & UMAP), and multi-detector anomaly concordance (Isolation Forest, One-Class SVM).
- **Rigorous Evaluation & Model Diagnostics:** Temporal CV vs. K-Fold leakage comparisons, Nested Cross-Validation for selection optimism auditing, Optuna Bayesian Optimization (TPE), bias-variance decomposition, residual diagnostic suites, and paired bootstrap hypothesis testing.
- **Cross-Plant Generalization:** Out-of-domain evaluation testing models trained on Plant 1 against completely unseen telemetry from Plant 2.
- **Interactive Flask Dashboard:** Responsive web interface with interactive prediction, exploratory data views, operational clustering, model evaluation, and report viewer.

---

## 📁 Repository Structure

```text
├── Data/
│   ├── Raw Data/                         # Original telemetry & sensor CSV files
│   │   ├── Plant_1_Generation_Data.csv
│   │   ├── Plant_1_Weather_Sensor_Data.csv
│   │   ├── Plant_2_Generation_Data.csv
│   │   └── Plant_2_Weather_Sensor_Data.csv
│   └── processed Data/                   # Cleaned datasets and chronological splits
│       ├── cleaned_plant1_data.csv
│       ├── cleaned_plant2_data.csv
│       └── splits/                       # Train, validation, test & unseen holdouts
├── Output/
│   ├── models/                           # Serialized scikit-learn & joblib models
│   ├── plots/                            # High-resolution diagnostic plots & charts
│   └── Report/                           # Comprehensive diagnostic text reports
├── src/                                  # Modular core ML library
│   ├── __init__.py
│   ├── config.py                         # Central paths, features, and random states
│   ├── data_utils.py                     # Cleaning, feature engineering & split pipelines
│   ├── cv_utils.py                       # Leakage-free splitters & CV estimators
│   ├── EDA.py                            # Exploratory data analysis & statistical summaries
│   ├── feature_eng.py                    # Domain-specific solar feature transformers
│   ├── linear_regression.py              # Analytical OLS & NumPy Batch Gradient Descent
│   ├── regularization.py                 # Ridge, Lasso, ElasticNet & shrinkage paths
│   ├── kmeans_clustering.py              # Lloyd vs MiniBatch K-Means & elbow analysis
│   ├── hierarchical_clustering.py        # Agglomerative clustering & cophenetic linkage
│   ├── dbscan_clustering.py              # Density-based clustering & k-dist elbow
│   ├── principal_component_analysis.py   # PCA loadings, scree plot & standardization
│   ├── tsne_visualization.py             # t-SNE non-linear 2D projections
│   ├── umap_visualization.py             # UMAP manifold embeddings & cross-plant transfer
│   ├── anomaly_detection.py              # Isolation Forest, OCSVM & reconstruction
│   ├── clustering_feature_augmentation.py # Semi-supervised feature augmentation
│   ├── cross_validation.py               # Time-series vs K-Fold cross validation
│   ├── nested_cross_validation.py        # Nested CV & selection bias estimation
│   ├── grid_search_vs_random_search.py   # Hyperparameter space search comparison
│   ├── bayesian_optimization.py          # Optuna Tree-structured Parzen Estimator
│   ├── learning_and_validation_curves.py # Bias-variance learning & validation curves
│   ├── model_diagnostics.py              # Residual diagnostic plots & Gauss-Markov tests
│   ├── regression_metrics_and_robustness.py # Outlier stress testing & metrics
│   ├── statistical_significance.py       # Paired t-tests, Wilcoxon & bootstrap CIs
│   ├── flask_service.py                  # Service layer bridging ML logic to Web UI
│   ├── run_module_1.py                   # Master runner for Ingestion, Cleaning & EDA
│   ├── run_module_2.py                   # Master runner for Linear Models & Regularization
│   ├── run_module_4.py                   # Master runner for Unsupervised Learning & Anomaly
│   └── run_module_5.py                   # Master runner for Evaluation, CV & Diagnostics
├── static/                               # UI assets: styles, responsive CSS, images
├── templates/                            # Jinja2 HTML templates for Flask UI
│   ├── base.html                         # Shared navigation & layout shell
│   ├── home.html                         # Landing page with hero section & KPIs
│   ├── load.html                         # Dataset loader & CSV upload switcher
│   ├── eda.html                          # Interactive EDA telemetry visualizer
│   ├── clustering.html                   # Unsupervised clustering & regime inspector
│   ├── feature_engg.html                 # Feature engineering & correlation view
│   ├── evaluation.html                   # Model evaluation, CV & residuals view
│   └── generalization.html               # Cross-plant transferability benchmark
├── uploads/                              # Temporary directory for runtime user uploads
├── app.py                                # Main Flask web application entrypoint
├── compose.yaml                          # Docker Compose multi-container deployment
├── Dockerfile                            # Production-ready container specification
├── requirements.txt                      # Project Python dependencies
└── .gitignore                            # Clean Git exclusion rules
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: Version 3.10, 3.11, or 3.12 recommended
- **Git**

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 💻 Running the Web Application

### Option A: Local Flask Development Server

```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Option B: Docker Compose

```bash
docker compose up --build
```
Access the application at `http://localhost:5000`.

---

## 🔬 Running the Machine Learning Pipelines

Each module can be executed independently to regenerate data splits, retrain models, produce diagnostic plots, and compile detailed reports:

| Module | Runner Command | Description |
|---|---|---|
| **Module 1** | `python src/run_module_1.py` | Data ingestion, multi-sensor merging, IQR cleaning, chronological splits, baseline evaluation, and exploratory data analysis. |
| **Module 2** | `python src/run_module_2.py` | Feature scaling, OLS regression, Batch Gradient Descent from scratch, Ridge/Lasso/ElasticNet hyperparameter tuning, and cross-plant transfer. |
| **Module 4** | `python src/run_module_4.py` | K-Means, Agglomerative Hierarchical clustering, DBSCAN, PCA biplots, t-SNE & UMAP embeddings, anomaly detection, and cluster-augmented regression. |
| **Module 5** | `python src/run_module_5.py` | Time-series cross validation, Nested CV, Optuna Bayesian Optimization, learning/validation curves, residual diagnostics, and paired bootstrap significance tests. |

All generated artifacts are saved automatically to:
- `Output/plots/`: High-resolution figures (.png)
- `Output/models/`: Serialized models (.joblib)
- `Output/Report/`: Analytical summaries (.txt, .csv)

---

## 📊 Core Features & Methodology

### 1. Multivariate Telemetry & Physical Feature Engineering
- **Target Variable:** `AC_POWER` (Alternating Current Power Output in kW).
- **Core Predictors:** `DC_POWER`, `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`.
- **Engineered Signals:** `TEMP_DIFF` ($\Delta T = T_{module} - T_{ambient}$), `IS_DAYLIGHT`, and temporal features (`HOUR`, `MINUTE`, `DAY`, `DAY_OF_WEEK`).

### 2. Regression & Regularization Suite
- **OLS Closed-Form:** Normal Equation $\hat{\beta} = (X^T X)^{-1} X^T y$ with matrix condition number monitoring ($\kappa(X)$).
- **NumPy Batch Gradient Descent:** Fully vectorized implementation with cost history tracking across learning rates.
- **Regularized Estimators:** Cross-validated Ridge ($L_2$), Lasso ($L_1$), and ElasticNet with full coefficient shrinkage path visualizations.

### 3. Clustering & Dimensionality Reduction
- **Regime Discovery:** Identifies clear operational regimes (Night standby, Low-light ramp-up, Midday peak generation, Thermal throttling).
- **Clustering Algorithms:** Lloyd's K-Means, MiniBatch K-Means, Hierarchical (Ward, Complete, Average with cophenetic correlation), and DBSCAN density analysis.
- **Manifold Learning:** PCA variance scree and loading biplots, t-SNE perplexity sweeps, and UMAP embeddings.

### 4. Statistical Verification & Diagnostics
- **Residual Testing:** Gauss-Markov assumption tests (homoscedasticity, normality of residuals via Q-Q and Shapiro/Kolmogorov-Smirnov).
- **Hypothesis Testing:** Paired t-tests, Wilcoxon signed-rank tests, and 1,000-sample bootstrap confidence intervals to prove model improvements are statistically significant.

---

## 🛠️ Tech Stack

- **Backend & Serving:** Python, Flask, Jinja2, Werkzeug, Gunicorn
- **Machine Learning & Statistics:** Scikit-Learn, SciPy, NumPy, Pandas, Optuna, UMAP-learn
- **Visualization:** Matplotlib, Seaborn
- **Containerization:** Docker, Docker Compose
