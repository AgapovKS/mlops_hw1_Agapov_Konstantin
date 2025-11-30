# mlops_hw1_Агапов_Константин

## Цель
Минималистичный MLOps pipeline: versioning данных через DVC, воспроизводимая подготовка и обучение, логирование экспериментов в MLflow.

## Быстрый старт
```bash
git clone https://github.com/AgapovKS/mlops_hw1_Agapov_Konstantin
cd mlops_hw1_Agapov_Konstantin
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell (Windows)
pip install -r requirements.txt

# Инициализация/получение данных (если repo пустой, dvc remote/pull)
dvc pull

# Запуск MLflow UI в новом терминале
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Открыть в браузере: http://127.0.0.1:5000

# Восстановить и запустить pipeline
dvc repro
