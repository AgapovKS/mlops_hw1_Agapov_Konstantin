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

# Mlflow settings
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = PROJECT_ROOT / "mlruns" # Папка для артефактов (локальная)

# --- ИСПРАВЛЕНИЕ: СНАЧАЛА УСТАНОВИТЕ TRACKING URI ---
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")  # <-- Устанавливаем URI ПЕРЕД созданием клиента
os.makedirs(MLFLOW_ARTIFACTS, exist_ok=True)
print("MLflow tracking URI:", mlflow.get_tracking_uri())

# Управление экспериментом
EXPERIMENT_NAME = "Default"
client = mlflow.tracking.MlflowClient() 

try:
    # Пытаемся получить существующий эксперимент (используя новый URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    print(f"Используется существующий эксперимент '{EXPERIMENT_NAME}' (ID: {experiment_id})")
    
except AttributeError:
    # Если эксперимент не существует, создаем его
    artifact_uri = f"file:///{MLFLOW_ARTIFACTS.as_posix()}"
    experiment_id = client.create_experiment(
        EXPERIMENT_NAME, 
        artifact_location=artifact_uri 
    )
    print(f"Создан новый эксперимент '{EXPERIMENT_NAME}' (ID: {experiment_id}), Artifact URI: {artifact_uri}")

# Убеждаемся, что текущий запуск будет в этом эксперименте
mlflow.set_experiment(EXPERIMENT_NAME)
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
        mlflow.log_param("model", model_name)
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state) # Логируем random_state

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", float(acc))

        # Сохраняем модель локально
        joblib.dump(model, MODEL_OUT)
        
        # Логируем артефакт модели
        mlflow.log_artifact(str(MODEL_OUT), artifact_path="model")
        


        print(f"Run saved with id: {run.info.run_id}, accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()