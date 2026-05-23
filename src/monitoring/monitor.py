"""STEP 6 — Monitoring"""

import json
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs("logs/monitoring", exist_ok=True)


class WeatherMonitor:

    def __init__(self, config: dict):
        self.config = config
        self.mcfg = config["monitoring"]
        self.ref_df = pd.read_csv(self.mcfg["reference_data_path"])
        self.features = (
            config["data"]["numeric_features"] + config["data"]["categorical_features"]
        )
        with open(config["model"]["save_path"], "rb") as f:
            self.model = pickle.load(f)

    def check_drift(self, current_df: pd.DataFrame) -> dict:
        """Simple statistical drift check (mean shift per feature)."""
        ref = self.ref_df[self.features]
        curr = current_df[self.features]
        drifted = []
        for col in self.features:
            ref_mean = ref[col].mean()
            curr_mean = curr[col].mean()
            ref_std = ref[col].std() + 1e-8
            shift = abs(curr_mean - ref_mean) / ref_std
            if shift > 0.5:
                drifted.append({"feature": col, "shift": round(shift, 4)})

        drift_score = len(drifted) / len(self.features)
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_score": round(drift_score, 4),
            "drift_detected": drift_score > self.mcfg["drift_threshold"],
            "drifted_features": drifted,
        }
        if result["drift_detected"]:
            logger.warning(f"⚠️  DRIFT DETECTED! Score: {drift_score:.4f}")
        else:
            logger.info(f"✅ No drift. Score: {drift_score:.4f}")
        return result

    def check_performance(self, current_df: pd.DataFrame, y_true) -> dict:
        acc = accuracy_score(
            y_true, self.model.predict(current_df[self.features].values)
        )
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "accuracy": round(acc, 4),
            "alert": acc < self.mcfg["accuracy_threshold"],
        }
        if result["alert"]:
            logger.warning(f"⚠️  ACCURACY DROP! {acc:.4f}")
        else:
            logger.info(f"✅ Accuracy OK: {acc:.4f}")
        return result

    def run(self, current_df: pd.DataFrame, y_true=None) -> dict:
        report = {"drift": self.check_drift(current_df)}
        if y_true is not None:
            report["performance"] = self.check_performance(current_df, y_true)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = f"{self.mcfg['report_path']}/monitor_{ts}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved → {path}")
        return report


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    with open("data/processed/data.pkl", "rb") as f:
        data = pickle.load(f)

    monitor = WeatherMonitor(config)
    current_df = pd.DataFrame(
        data["X_test"],
        columns=config["data"]["numeric_features"]
        + config["data"]["categorical_features"],
    )
    print(json.dumps(monitor.run(current_df, data["y_test"]), indent=2))
