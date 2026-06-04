import json
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_logreg_baseline(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test=None,
    y_test=None,
    random_state=42,
):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])

    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "train_macro_f1": float(f1_score(y_train, train_pred, average="macro")),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_macro_f1": float(f1_score(y_val, val_pred, average="macro")),
    }

    if X_test is not None and y_test is not None:
        test_pred = clf.predict(X_test)
        metrics.update({
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        })

    result = {
        "model": "LogisticRegression baseline (no PCA)",
        "pca_dim": "none",
        **metrics,
    }

    return clf, metrics, result


def run_pca_logreg_model_selection(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test=None,
    y_test=None,
    pca_dims=(5, 10, 25, 50, 100, 200),
    random_state=42,
):
    results = []
    best_model = None
    best_dim = None
    best_f1 = -1

    for dim in pca_dims:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=dim, random_state=random_state)),
            ("logreg", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ])

        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)

        train_acc = float(accuracy_score(y_train, train_pred))
        train_macro_f1 = float(f1_score(y_train, train_pred, average="macro"))
        val_acc = float(accuracy_score(y_val, val_pred))
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro"))

        row = {
            "model": f"PCA-{dim} + LogisticRegression",
            "pca_dim": dim,
            "train_accuracy": train_acc,
            "train_macro_f1": train_macro_f1,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
        }

        if X_test is not None and y_test is not None:
            test_pred = clf.predict(X_test)
            row.update({
                "test_accuracy": float(accuracy_score(y_test, test_pred)),
                "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
            })

        results.append(row)

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_dim = dim
            best_model = clf

    return best_model, best_dim, pd.DataFrame(results)


def evaluate_sklearn_model(model, X, y, id2label):
    pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
    }

    report = classification_report(
        y,
        pred,
        target_names=[id2label[i] for i in range(3)],
        output_dict=True,
    )

    return pred, metrics, report
