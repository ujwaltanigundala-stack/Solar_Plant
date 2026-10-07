"""Central configuration for Solar Power Generation Prediction Project.

Follows the reference architecture of the Placement Prediction classroom project,
tailored specifically to multivariate solar plant generation and weather sensor data.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RAW_DATA_DIR = DATA_DIR / "Raw Data"
PROCESSED_DATA_DIR = DATA_DIR / "processed Data"
SPLITS_DIR = DATA_DIR / "processed Data" / "splits"
OUTPUT_DIR = BASE_DIR / "Output"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "Report"

# Ensure required directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Raw dataset files
PLANT1_GEN_PATH = RAW_DATA_DIR / "Plant_1_Generation_Data.csv"
PLANT1_WEATHER_PATH = RAW_DATA_DIR / "Plant_1_Weather_Sensor_Data.csv"
PLANT2_GEN_PATH = RAW_DATA_DIR / "Plant_2_Generation_Data.csv"
PLANT2_WEATHER_PATH = RAW_DATA_DIR / "Plant_2_Weather_Sensor_Data.csv"

# Processed dataset paths
CLEANED_PLANT1_PATH = PROCESSED_DATA_DIR / "cleaned_plant1_data.csv"
CLEANED_PLANT2_PATH = PROCESSED_DATA_DIR / "cleaned_plant2_data.csv"
SPLIT_TRAIN_PATH = SPLITS_DIR / "plant1_train.csv"
SPLIT_VAL_PATH = SPLITS_DIR / "plant1_val.csv"
SPLIT_TEST_PATH = SPLITS_DIR / "plant1_test.csv"
SPLIT_UNSEEN_PLANT2_PATH = SPLITS_DIR / "plant2_unseen.csv"

# Target Definition (Supervised Multivariate Regression)
TARGET_COLUMN = "AC_POWER"
TARGET_REGRESSION = TARGET_COLUMN

# Key Feature Definitions
# Note: Categorical/Identifiers like PLANT_ID, SOURCE_KEY_INVERTER are tracked for group analysis
# but excluded from direct cross-plant generalization features to prevent overfitting to specific inverter IDs.
NUMERIC_FEATURES = [
    "DC_POWER",
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "HOUR",
    "MINUTE",
    "DAY",
    "DAY_OF_WEEK",
    "TEMP_DIFF",
    "IS_DAYLIGHT"
]

CORE_PHYSICAL_FEATURES = [
    "DC_POWER",
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE"
]

TEMPORAL_FEATURES = [
    "HOUR",
    "MINUTE",
    "DAY",
    "DAY_OF_WEEK"
]

ID_COLS = ["DATE_TIME", "PLANT_ID", "SOURCE_KEY_INVERTER", "SOURCE_KEY_SENSOR"]

# Reproducibility
RANDOM_STATE = 42

# Report paths
CLEANING_REPORT_PATH = REPORTS_DIR / "cleaning_report.txt"
SPLIT_REPORT_PATH = REPORTS_DIR / "split_report.txt"
BASELINE_REPORT_PATH = REPORTS_DIR / "baseline_report.txt"
EDA_REPORT_PATH = REPORTS_DIR / "EDAsummary.txt"
