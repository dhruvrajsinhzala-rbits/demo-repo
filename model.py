import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE


def load_data(path):
    df = pd.read_csv(path)
    return df


def split_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(name, y_true, y_pred, y_proba):
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_true, y_proba))


def baseline_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    evaluate_model("Baseline Logistic Regression", y_test, y_pred, y_proba)


def smote_logistic_regression(X_train, X_test, y_train, y_test):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE class distribution:")
    print(pd.Series(y_train_smote).value_counts())

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    evaluate_model("SMOTE Logistic Regression", y_test, y_pred, y_proba)


def random_forest_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    evaluate_model("Random Forest", y_test, y_pred, y_proba)


def main():
    df = load_data("creditcard.csv")

    print(df.head())
    print(df.info())
    print("\nClass distribution:")
    print(df["Class"].value_counts())

    X_train, X_test, y_train, y_test = split_data(df)

    print("\nTrain fraud cases:", y_train.sum())
    print("Test fraud cases:", y_test.sum())

    baseline_logistic_regression(X_train, X_test, y_train, y_test)
    smote_logistic_regression(X_train, X_test, y_train, y_test)
    random_forest_model(X_train, X_test, y_train, y_test)


main()