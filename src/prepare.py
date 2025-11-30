# src/prepare.py
import os
from pathlib import Path
import pandas as pd
import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_or_create_raw():
    if RAW.exists():
        df = pd.read_csv(RAW)
        print(f"Loaded raw data from {RAW}")
    else:
        # Если вдруг нет raw, создаём Iris и сохраняем
        iris = load_iris(as_frame=True)
        df = iris.frame
        df.columns = list(iris.feature_names) + ["target"]
        RAW.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW, index=False)
        print(f"Raw data not found — created Iris at {RAW}")
    return df

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROJECT_ROOT / "params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    df = load_or_create_raw()

    # Перемешаем, удаляем NaN, сплит
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df = df.dropna()

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique())>1 else None
    )

    train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f" New prepareted. train and test written to {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
