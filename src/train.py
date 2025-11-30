# src/train.py
import os
from pathlib import Path
import pandas as pd
import yaml
import mlflow
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_OUT = PROJECT_ROOT / "model.pkl"

def load_params():
    with open(PROJECT_ROOT / "params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    mlflow_uri = params["mlflow"]["host"]
    mlflow.set_tracking_uri(mlflow_uri)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    # Загружаем данные
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    # fix seed для воспроизводимости
    random_state = params["split"]["random_state"]
    np.random.seed(random_state)

    # Подготовка модели LogisticRegression
    from sklearn.linear_model import LogisticRegression
    model_params = params["model"]
    model = LogisticRegression(C=model_params["C"], max_iter=model_params["max_iter"], random_state=random_state)

    with mlflow.start_run():
        mlflow.log_param("model", model_params.get("name", "LogisticRegression"))
        mlflow.log_param("C", model_params["C"])
        mlflow.log_param("max_iter", model_params["max_iter"])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", float(acc))

        # Сохраняем модель на диск и логируем артефакт
        joblib.dump(model, MODEL_OUT)
        mlflow.log_artifact(str(MODEL_OUT), artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"Run saved with id: {run_id}, accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
