"""STEP 7 — Full Pipeline DAG"""

import logging
import os
import pickle
import sys

import mlflow  # ✅ moved to top-level imports
import ray
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.ingest import load_data, validate_data
from src.data.preprocess import preprocess
from src.training.trainer import train_model
from src.tuning.tune import run_tuning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(config_path="config/config.yaml", skip_tuning=False):
    ray.init(ignore_reinit_error=True)

    # ✅ Set MLflow URI immediately after Ray init before it can override
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:///tmp/mlruns"))

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # STEP 1 — Ingest
    logger.info("=" * 50)
    logger.info("STEP 1 — Data Ingestion")
    logger.info("=" * 50)
    df = ray.get(
        validate_data.remote(
            ray.get(load_data.remote(config["data"]["raw_path"])), config
        )
    )

    # STEP 2 — Preprocess
    logger.info("=" * 50)
    logger.info("STEP 2 — Preprocessing")
    logger.info("=" * 50)
    data = ray.get(preprocess.remote(df, config))

    # STEP 3 — Tune (optional)
    if not skip_tuning:
        logger.info("=" * 50)
        logger.info("STEP 3 — Hyperparameter Tuning")
        logger.info("=" * 50)
        best = run_tuning(data, config)
        config["model"]["params"].update(best)
    else:
        logger.info("STEP 3 — Skipping tuning")

    # STEP 4 — Train
    logger.info("=" * 50)
    logger.info("STEP 4 — Training")
    logger.info("=" * 50)
    model, metrics = train_model(data, config)
    logger.info(f"Metrics: {metrics}")

    # Accuracy gate
    if metrics["test_accuracy"] < config["monitoring"]["accuracy_threshold"]:
        logger.error(
            f"❌ Accuracy {metrics['test_accuracy']:.4f} below threshold. Aborting."
        )
        return

    # STEP 5 — Deploy (lazy import fixes serialization error)
    logger.info("=" * 50)
    logger.info("STEP 5 — Deploying with Ray Serve")
    logger.info("=" * 50)
    from src.serving.serve import deploy

    deploy(config)

    logger.info("=" * 50)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info(f"Test Accuracy : {metrics['test_accuracy']:.4f}")
    logger.info(f"API           → http://0.0.0.0:{config['serving']['port']}")
    logger.info(f"Docs          → http://0.0.0.0:{config['serving']['port']}/docs")
    logger.info(f"MLflow        → {config['mlflow']['tracking_uri']}")
    logger.info(f"Ray Dashboard → http://0.0.0.0:8265")
    logger.info("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()
    run_pipeline(config_path=args.config, skip_tuning=args.skip_tuning)
