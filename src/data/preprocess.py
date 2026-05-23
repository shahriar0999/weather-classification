"""STEP 2 — Preprocessing"""

import pandas as pd
import numpy as np
import ray
import pickle
import os
import yaml
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs("models", exist_ok=True)


@ray.remote
def preprocess(df: pd.DataFrame, config: dict) -> dict:
    cfg = config["data"]

    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing values
    for col in cfg["numeric_features"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in cfg["categorical_features"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Clip outliers
    for col, bounds in cfg["outlier_thresholds"].items():
        if col in df.columns:
            df[col] = df[col].clip(lower=bounds["min"], upper=bounds["max"])

    # Encode categorical features
    label_encoders = {}
    for col in cfg["categorical_features"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    with open(config["model"]["encoder_path"], "wb") as f:
        pickle.dump(label_encoders, f)

    # Encode target
    target_le = LabelEncoder()
    df[cfg["target_column"]] = target_le.fit_transform(df[cfg["target_column"]])
    with open(config["model"]["target_encoder_path"], "wb") as f:
        pickle.dump(target_le, f)

    # Build feature matrix
    all_features = cfg["numeric_features"] + cfg["categorical_features"]
    X = df[all_features].values
    y = df[cfg["target_column"]].values

    # Scale numeric columns only
    scaler = StandardScaler()
    n = len(cfg["numeric_features"])
    X[:, :n] = scaler.fit_transform(X[:, :n])
    with open(config["model"]["scaler_path"], "wb") as f:
        pickle.dump(scaler, f)

    # Train / Val / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"],
        random_state=cfg["random_state"], stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=cfg["val_size"],
        random_state=cfg["random_state"], stratify=y_train
    )

    logger.info(f"Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}")

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    data = {
        "X_train": X_train, "X_val": X_val,   "X_test":  X_test,
        "y_train": y_train, "y_val": y_val,   "y_test":  y_test,
        "feature_names":  all_features,
        "target_classes": list(target_le.classes_)
    }
    with open("data/processed/data.pkl", "wb") as f:
        pickle.dump(data, f)

    # Save reference data for monitoring
    ref = pd.DataFrame(X_train, columns=all_features)
    ref[cfg["target_column"]] = y_train
    ref.to_csv(config["monitoring"]["reference_data_path"], index=False)

    return data


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    df = pd.read_csv(config["data"]["raw_path"])
    data = ray.get(preprocess.remote(df, config))
    print(f"Train: {data['X_train'].shape} | Classes: {data['target_classes']}")
