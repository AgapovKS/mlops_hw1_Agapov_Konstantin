# src/prepare.py
import os
from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_params():
    with open(PROJECT_ROOT / "params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_or_create_raw():
    if RAW.exists():
        df = pd.read_csv(RAW)
        print(f"Загружен raw data из {RAW}")
    else:
        raise FileNotFoundError(f"Raw data не найден в {RAW}")
    return df

def basic_preprocess(df, params):
    for c in params["prepare"].get("drop_columns", []):
        if c in df.columns:
            df = df.drop(columns=[c])

    #  Age заполняем медианой
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # Embarked заполняем модой
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])

    # labelencode Sex и Embarked
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].astype(str)
        df = pd.get_dummies(df, columns=["Embarked"], prefix="Emb")

    # drop колонки с NA
    df = df.dropna().reset_index(drop=True)
    return df

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    params = load_params()
    target_col = params["prepare"]["target_col"]
    test_size = params["prepare"]["test_size"]
    random_state = params["prepare"]["random_state"]

    df = load_or_create_raw()
    # печатаем колонки для отладки
    print("Columns in raw:", list(df.columns))

    df = basic_preprocess(df, params)

    if target_col not in df.columns:
        raise KeyError(f"Target  '{target_col}' не найден. Список колонок: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if len(y.unique()) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"Prepared train/test written to {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
