# Weather Classification

A machine learning pipeline that classifies weather conditions from meteorological data. The project trains an XGBoost model and serves it as a live REST API on Anyscale, with the entire process automated through GitHub Actions.


## Overview

The pipeline takes raw weather data, preprocesses it, trains a classifier, and deploys it as a persistent API. Every time you push a change to the main branch, the model retrains automatically and the live API is updated without any manual steps.

The model classifies weather into four categories: Cloudy, Rainy, Snowy, and Sunny, achieving around 92% accuracy on the test set.


## Project Structure

```
weather-classification/
├── .github/
│   └── workflows/
│       └── ci_cd.yml                  # GitHub Actions CI/CD pipeline
├── config/
│   └── config.yaml                    # All pipeline configuration
├── data/                              # Raw and processed data
├── logs/
│   └── monitoring/                    # Monitoring reports
├── models/
│   ├── weather_model.pkl              # Trained XGBoost model
│   ├── scaler.pkl                     # Feature scaler
│   ├── label_encoders.pkl             # Categorical encoders
│   └── target_encoder.pkl             # Target label encoder
├── mlruns/                            # MLflow experiment tracking data
├── notebooks/                         # Exploratory notebooks
├── src/
│   ├── data/
│   │   ├── ingest.py                  # Data loading and validation
│   │   └── preprocess.py             # Feature engineering and scaling
│   ├── monitoring/
│   │   └── monitor.py                # Drift and accuracy monitoring
│   ├── pipeline/
│   │   └── workflow.py               # Full pipeline orchestration
│   ├── serving/
│   │   └── serve.py                  # Ray Serve API deployment
│   ├── training/
│   │   └── trainer.py                # Model training and MLflow logging
│   └── tuning/
│       └── tune.py                   # Hyperparameter tuning with Ray Tune
├── tests/
│   └── test_pipeline.py              # Unit and integration tests
├── Dockerfile                         # Container definition
├── requirements.txt
└── run_serve.py                       # Local serving entry point
```


## Requirements

- Python 3.10
- The dependencies listed in requirements.txt
- An Anyscale account for deployment
- A GitHub repository with Actions enabled


## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/weather-classification.git
cd weather-classification
pip install -r requirements.txt
```


## Configuration

All pipeline settings are controlled through config/config.yaml. The key sections are:

- data: file paths, train/val/test split ratios, feature lists, and outlier thresholds
- model: XGBoost hyperparameters and artifact save paths
- mlflow: experiment tracking settings
- serving: API host, port, and replica count
- monitoring: accuracy thresholds and drift detection settings

To retrain with different hyperparameters, edit the params block under model in config.yaml and push to main. The CI/CD pipeline will automatically retrain and redeploy.


## Running Locally

To run the full training pipeline locally:

```bash
python src/pipeline/workflow.py --skip-tuning
```

To run with hyperparameter tuning:

```bash
python src/pipeline/workflow.py
```


## CI/CD Pipeline

The GitHub Actions workflow runs automatically on every push to main or develop, and on pull requests to main. It runs four jobs in sequence:

**lint** checks code formatting using Black. All source files must pass before the pipeline proceeds.

**test** runs the full test suite with pytest and generates a coverage report.

**train** installs dependencies, creates the necessary Ray directories, runs the training pipeline, and uploads the trained model as a build artifact.

**deploy** downloads the trained model artifact and deploys it to Anyscale as a persistent service. This job only runs on pushes to main.

To trigger a full retrain and redeploy, simply push any change to the main branch.


## Deployment

The API is deployed on Anyscale using Ray Serve. The deployment is managed entirely by the CI/CD pipeline and requires one secret to be set in your GitHub repository.

Go to your repository settings, navigate to Secrets and variables under Actions, and create a new secret:

```
Name:  ANYSCALE_CLI_TOKEN
Value: your token from https://console.anyscale.com/v2/api-keys
```

Once set, every push to main will automatically deploy the latest model to your Anyscale service.


## API Usage

The live API requires an Authorization header with your Anyscale API token.

**Health check:**

```bash
curl -H "Authorization: Bearer your-token" \
  https://your-service-url/health
```

**Single prediction:**

```bash
curl -X POST https://your-service-url/predict \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "Temperature": 25.0,
    "Humidity": 70.0,
    "Wind_Speed": 15.0,
    "Precipitation": 40.0,
    "Cloud_Cover": "partly cloudy",
    "Atmospheric_Pressure": 1013.0,
    "UV_Index": 5,
    "Season": "Winter",
    "Visibility_km": 10.0,
    "Location": "inland"
  }'
```

**Example response:**

```json
{
  "weather_type": "Rainy",
  "confidence": 0.923,
  "probabilities": {
    "Cloudy": 0.021,
    "Rainy": 0.923,
    "Snowy": 0.031,
    "Sunny": 0.025
  },
  "latency_ms": 12.5
}
```

**Batch prediction:**

```bash
curl -X POST https://your-service-url/batch \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "Temperature": 25.0,
        "Humidity": 70.0,
        "Wind_Speed": 15.0,
        "Precipitation": 40.0,
        "Cloud_Cover": "partly cloudy",
        "Atmospheric_Pressure": 1013.0,
        "UV_Index": 5,
        "Season": "Winter",
        "Visibility_km": 10.0,
        "Location": "inland"
      }
    ]
  }'
```

**Interactive API docs** are available at your service URL with /docs appended. This provides a browser-based interface to test all endpoints without writing any code.


## Model Performance

The XGBoost model is trained on an 80/10/10 train/validation/test split with the following results:

```
              precision    recall  f1-score   support

      Cloudy       0.88      0.91      0.90       660
       Rainy       0.90      0.92      0.91       660
       Snowy       0.95      0.92      0.93       660
       Sunny       0.93      0.92      0.93       660

    accuracy                           0.92      2640
   macro avg       0.92      0.92      0.92      2640
weighted avg       0.92      0.92      0.92      2640
```


## Experiment Tracking

Training metrics and model artifacts are logged to MLflow. When running locally, the tracking data is stored at /tmp/mlruns by default. You can view the MLflow UI by running:

```bash
mlflow ui --backend-store-uri /tmp/mlruns
```

Then open http://localhost:5000 in your browser.


## Running Tests

```bash
pytest tests/ --cov=src --cov-report=xml -v
```


## Troubleshooting

**Permission denied on /home/ray during training**

The CI/CD workflow creates and opens permissions on /home/ray before training runs. If you see this error locally, run:

```bash
sudo mkdir -p /home/ray && sudo chmod 777 /home/ray
```

**MLflow writing to wrong directory**

Set the MLFLOW_TRACKING_URI environment variable before running the pipeline:

```bash
export MLFLOW_TRACKING_URI=file:///tmp/mlruns
python src/pipeline/workflow.py --skip-tuning
```

**Anyscale deployment unhealthy**

Check the application logs in the Anyscale console under Services, then select your service version and open the Logs tab. Common causes are missing Python dependencies or model files not being included in the working directory.