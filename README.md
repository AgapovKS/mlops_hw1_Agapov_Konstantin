# mlops_hw1_Агапов_Константин

## Цель
Минималистичный MLOps pipeline: versioning данных через DVC, воспроизводимая подготовка и обучение, логирование экспериментов в MLflow.

## Быстрый старт
```bash
git clone <repo>
cd <repo>
python -m venv .venv && source .venv/Scripts/Activate.ps1   # PowerShell (Windows)
pip install -r requirements.txt

# Инициализация/получение данных (если repo пустой, dvc remote/pull)
dvc pull

# Запуск MLflow UI в новом терминале
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Восстановить и запустить pipeline
dvc repro
