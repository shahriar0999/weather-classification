"""STEP 4 — Hyperparameter Tuning"""

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pickle
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _train_fn(trial_config, X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        n_estimators     = trial_config["n_estimators"],
        max_depth        = trial_config["max_depth"],
        learning_rate    = trial_config["learning_rate"],
        subsample        = trial_config["subsample"],
        colsample_bytree = trial_config["colsample_bytree"],
        use_label_encoder= False,
        eval_metric      = "mlogloss",
        random_state     = 42
    )
    model.fit(X_train, y_train)
    tune.report({"accuracy": accuracy_score(y_val, model.predict(X_val))})


def run_tuning(data: dict, config: dict) -> dict:
    ray.init(ignore_reinit_error=True)
    cfg = config["tuning"]
    sp  = cfg["search_space"]

    tuner = tune.Tuner(
        tune.with_parameters(
            _train_fn,
            X_train=data["X_train"], y_train=data["y_train"],
            X_val=data["X_val"],     y_val=data["y_val"]
        ),
        param_space={
            "n_estimators":     tune.choice(sp["n_estimators"]),
            "max_depth":        tune.choice(sp["max_depth"]),
            "learning_rate":    tune.loguniform(*sp["learning_rate"]),
            "subsample":        tune.uniform(*sp["subsample"]),
            "colsample_bytree": tune.uniform(*sp["colsample_bytree"]),
        },
        tune_config=tune.TuneConfig(
            scheduler   = ASHAScheduler(metric="accuracy", mode="max"),
            num_samples = cfg["num_samples"],
            metric      = cfg["metric"],
            mode        = cfg["mode"]
        )
    )

    results = tuner.fit()
    best    = results.get_best_result(cfg["metric"], cfg["mode"])
    logger.info(f"Best config   : {best.config}")
    logger.info(f"Best accuracy : {best.metrics['accuracy']:.4f}")
    return best.config


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    with open("data/processed/data.pkl", "rb") as f:
        data = pickle.load(f)
    print(run_tuning(data, config))
