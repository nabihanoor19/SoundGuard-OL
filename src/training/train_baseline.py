
from pathlib import Path
import json

import dagshub
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


def main():

    # Connect MLflow to DagsHub
    dagshub.init(
        repo_owner="nabihanoorkhalique",
        repo_name="SoundGuard-OL",
        mlflow=True
    )

    mlflow.set_experiment("SoundGuard-OL")

    features = np.load("data/processed/features.npy")
    labels = np.load("data/processed/labels.npy")
    folds = np.load("data/processed/folds.npy")

    X = np.concatenate(
        [features.mean(axis=2), features.std(axis=2)],
        axis=1
    )

    train_idx = folds != 5
    test_idx = folds == 5

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    print("Training baseline model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test))
    }

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(model, "models/baseline_model.joblib")

    with open("reports/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Log experiment to MLflow
    with mlflow.start_run(run_name="Baseline-Logistic-Regression") as run:

        mlflow.log_params({
            "model": "Logistic Regression",
            "max_iter": 1000,
            "random_state": 42,
            "test_fold": 5,
            "n_mels": 64,
            "sample_rate": 22050
        })

        mlflow.log_metrics(metrics)

        mlflow.log_artifact("reports/baseline_metrics.json")

        # Log model and capture its actual URI
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="baseline_model"
        )

        # Register model using the URI returned by MLflow
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name="SoundGuard-Baseline"
        )

        # Move registered model to Staging
        client = MlflowClient()

        client.transition_model_version_stage(
            name="SoundGuard-Baseline",
            version=registered_model.version,
            stage="Staging",
            archive_existing_versions=True
        )

        print("\nMLFLOW RUN COMPLETED")
        print("Run ID:", run.info.run_id)
        print("Registered model:", "SoundGuard-Baseline")
        print("Model version:", registered_model.version)
        print("Model stage: Staging")

    print("\nBASELINE METRICS:")
    print(json.dumps(metrics, indent=4))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
