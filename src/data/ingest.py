"""STEP 1 — Data Ingestion"""

import pandas as pd
import ray
import logging
import os
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote
def load_data(raw_path: str) -> pd.DataFrame:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Data not found: {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


@ray.remote
def validate_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    required = (
        config["data"]["numeric_features"]
        + config["data"]["categorical_features"]
        + [config["data"]["target_column"]]
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    missing_vals = df.isnull().sum()
    missing_vals = missing_vals[missing_vals > 0]
    if len(missing_vals):
        logger.warning(f"Missing values:\n{missing_vals}")
    else:
        logger.info("No missing values")

    dupes = df.duplicated().sum()
    logger.info(f"Duplicate rows: {dupes}")
    logger.info(f"Class distribution:\n{df[config['data']['target_column']].value_counts()}")

    for col, bounds in config["data"]["outlier_thresholds"].items():
        if col in df.columns:
            low  = (df[col] < bounds["min"]).sum()
            high = (df[col] > bounds["max"]).sum()
            if low + high > 0:
                logger.warning(f"Outliers in '{col}': {low} low, {high} high")
    return df


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    df = ray.get(validate_data.remote(
        ray.get(load_data.remote(config["data"]["raw_path"])), config
    ))
    print(df.head())
