# src/train.py
import os
from pathlib import Path
import pandas as pd
import yaml
import mlflow
import joblib
import numpy as np

# === MLflow settings ===
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_OUT = PROJECT_ROOT / "model.pkl"

def load_params():
    with open(PROJECT_ROOT / "params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()

    split_params = params.get("split") or params.get("prepare") or {}
    test_size = split_params.get("test_size", 0.2)
    random_state = split_params.get("random_state", 42)

    # определяем target column
    target_col = None
    if isinstance(params.get("prepare"), dict):
        target_col = params["prepare"].get("target_col")
    if not target_col:
        target_col = params.get("target_col", "target")

    # MLflow URI
    mlflow_uri = params.get("mlflow", {}).get("host", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    # загружаем данные
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    if target_col not in train.columns:
        raise KeyError(f"Target column '{target_col}' не найден. Train columns: {list(train.columns)}")

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    np.random.seed(random_state)

    # модель
    from sklearn.linear_model import LogisticRegression
    model_params = params.get("model", {})
    C = model_params.get("C", 1.0)
    max_iter = model_params.get("max_iter", 200)
    model_name = model_params.get("name", "LogisticRegression")
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)

    with mlflow.start_run() as run:
        # логируем параметры
        mlflow.log_param("model", model_name)
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)

        # тренируем
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", float(acc))

        # сохраняем модель на диск
        joblib.dump(model, MODEL_OUT)

        # логируем модель как артефакт после сохранения
        # создаём папку артефактов, чтобы точно существовала
        artifact_dir = Path(PROJECT_ROOT / "mlruns" / run.info.run_id / "artifacts" / "model")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.log_artifact(str(MODEL_OUT), artifact_path="model")

        print(f"Run saved with id: {run.info.run_id}, accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
