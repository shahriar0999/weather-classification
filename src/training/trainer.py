"""STEP 3 — Model Training"""

import pickle
import yaml
import logging
import os
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs("models", exist_ok=True)


def train_model(data: dict, config: dict):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="xgboost-weather-v1"):
        params = config["model"]["params"]
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(
            data["X_train"],
            data["y_train"],
            eval_set=[(data["X_val"], data["y_val"])],
            verbose=50,
        )

        y_pred = model.predict(data["X_test"])
        acc = accuracy_score(data["y_test"], y_pred)
        f1 = f1_score(data["y_test"], y_pred, average="weighted")
        val_acc = accuracy_score(data["y_val"], model.predict(data["X_val"]))

        logger.info(f"Val Acc : {val_acc:.4f}")
        logger.info(f"Test Acc: {acc:.4f}")
        logger.info(f"F1      : {f1:.4f}")
        logger.info(
            "\n"
            + classification_report(
                data["y_test"], y_pred, target_names=data["target_classes"]
            )
        )

        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)

        with open(config["model"]["save_path"], "wb") as f:
            pickle.dump(model, f)
        mlflow.xgboost.log_model(model, "model")

    return model, {"val_accuracy": val_acc, "test_accuracy": acc, "test_f1": f1}


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    with open("data/processed/data.pkl", "rb") as f:
        data = pickle.load(f)
    model, metrics = train_model(data, config)
    print(f"Metrics: {metrics}")
