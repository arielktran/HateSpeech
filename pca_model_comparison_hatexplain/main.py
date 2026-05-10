import pandas as pd
import os
import numpy as np
import json
from src.data import (load_cleaned_hatexplain, split_cleaned_dataset)
from src.model import load_model
from src.evaluate import evaluate_cleaned
from src.embedding_utils import extract_cls_embeddings
from src.pca_model import run_pca_logreg_model_selection, evaluate_sklearn_model
from src.config import ID2LABEL
from src.visualization import (
    plot_model_comparison,
    plot_confusion,
    plot_2d_pca_scatter,
    plot_group_macro_f1
)
from src.bootstrap import bootstrap_macro_f1_comparison
from src.group_analysis import add_group_columns, evaluate_group_metrics

def main():
    os.makedirs("outputs", exist_ok=True)

    tokenizer, model, device = load_model()

    df = load_cleaned_hatexplain("data/df_final.csv")

    train_df, val_df, test_df = split_cleaned_dataset(df)

    train_df.to_csv("outputs/train_split.csv", index=False)
    val_df.to_csv("outputs/val_split.csv", index=False)
    test_df.to_csv("outputs/test_split.csv", index=False)

    print("Saved splits to outputs/")

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    bert_results_path = "outputs/bert_hatexplain_cleaned_results.csv"

    if os.path.exists(bert_results_path):
        print("\nLoading saved BERT evaluation results...")
        results_df = pd.read_csv(bert_results_path)

    else:
        rows, y_true, y_pred = evaluate_cleaned(
            model=model,
            tokenizer=tokenizer,
            device=device,
            df=test_df
        )

        results_df = pd.DataFrame(rows)
        results_df.to_csv(bert_results_path, index=False)

        print("\nSaved results to:")
        print(bert_results_path)


    if (
            os.path.exists("outputs/X_train_cls.npy")
            and os.path.exists("outputs/X_val_cls.npy")
            and os.path.exists("outputs/X_test_cls.npy")
        ):
            print("\nLoading saved CLS embeddings...")

            X_train = np.load("outputs/X_train_cls.npy")
            y_train = np.load("outputs/y_train.npy")

            X_val = np.load("outputs/X_val_cls.npy")
            y_val = np.load("outputs/y_val.npy")

            X_test = np.load("outputs/X_test_cls.npy")
            y_test = np.load("outputs/y_test.npy")

    else:
            print("\nExtracting CLS embeddings for train/val/test splits...")

            X_train, y_train, train_texts = extract_cls_embeddings(
                train_df, tokenizer, model, device, batch_size=32
            )

            X_val, y_val, val_texts = extract_cls_embeddings(
                val_df, tokenizer, model, device, batch_size=32
            )

            X_test, y_test, test_texts = extract_cls_embeddings(
                test_df, tokenizer, model, device, batch_size=32
            )

            np.save("outputs/X_train_cls.npy", X_train)
            np.save("outputs/y_train.npy", y_train)

            np.save("outputs/X_val_cls.npy", X_val)
            np.save("outputs/y_val.npy", y_val)

            np.save("outputs/X_test_cls.npy", X_test)
            np.save("outputs/y_test.npy", y_test)

            print("Saved CLS embeddings to outputs/")

    print("Train embeddings:", X_train.shape)
    print("Val embeddings:", X_val.shape)
    print("Test embeddings:", X_test.shape)

    print("\nRunning PCA + Logistic Regression model selection...")

    best_pca_model, best_dim, pca_results_df = run_pca_logreg_model_selection(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        pca_dims=(5, 10, 25,50, 100, 200),
    )

    pca_results_df.to_csv("outputs/pca_model_selection_results.csv", index=False)

    print("\n=== PCA MODEL SELECTION RESULTS ===")
    print(pca_results_df)
    print(f"\nBest PCA dimension: {best_dim}")

    test_pred, test_metrics, test_report = evaluate_sklearn_model(
        model=best_pca_model,
        X=X_test,
        y=y_test,
        id2label=ID2LABEL,
    )

    print("\n=== BEST PCA MODEL TEST RESULTS ===")
    print("Accuracy:", test_metrics["accuracy"])
    print("Macro F1:", test_metrics["macro_f1"])

    pca_test_results = pd.DataFrame({
        "true_label": [ID2LABEL[i] for i in y_test],
        "pred_label": [ID2LABEL[i] for i in test_pred],
    })

    pca_test_results.to_csv("outputs/pca_logreg_test_results.csv", index=False)

    with open("outputs/pca_test_metrics.json", "w") as f:
        json.dump({
            "best_pca_dim": int(best_dim),
            "test_metrics": test_metrics,
            "classification_report": test_report,
        }, f, indent=2)

    print("\nSaved PCA outputs to:")
    print("outputs/pca_model_selection_results.csv")
    print("outputs/pca_logreg_test_results.csv")
    print("outputs/pca_test_metrics.json")


    print("\nCreating visualizations...")

    # Reload baseline BERT predictions
    bert_results_df = pd.read_csv("outputs/bert_hatexplain_cleaned_results.csv")

    bert_y_true = bert_results_df["true_label"].map({
        "hatespeech": 0,
        "normal": 1,
        "offensive": 2,
    }).to_numpy()

    bert_y_pred = bert_results_df["pred_label"].map({
        "hatespeech": 0,
        "normal": 1,
        "offensive": 2,
    }).to_numpy()

    # Model comparison chart
    plot_model_comparison(
        metrics_dict={
            "BERT Head": {
                "accuracy": 0.7379095163806553,
                "macro_f1": 0.7338711062915072,
            },
            f"PCA-{best_dim} + LogReg": {
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
            },
        },
        save_path="outputs/model_comparison.png"
    )

    # Confusion matrix: original BERT head
    plot_confusion(
        y_true=bert_y_true,
        y_pred=bert_y_pred,
        labels=[ID2LABEL[i] for i in range(3)],
        title="BERT-HateXplain Confusion Matrix",
        save_path="outputs/bert_confusion_matrix.png"
    )

    # Confusion matrix: PCA model
    plot_confusion(
        y_true=y_test,
        y_pred=test_pred,
        labels=[ID2LABEL[i] for i in range(3)],
        title=f"PCA-{best_dim} + Logistic Regression Confusion Matrix",
        save_path="outputs/pca_confusion_matrix.png"
    )

    # 2D PCA scatterplot of test CLS embeddings
    plot_2d_pca_scatter(
        X=X_test,
        y=y_test,
        labels=ID2LABEL,
        save_path="outputs/pca_scatter_test.png"
    )

    print("Saved visualizations to outputs/")


    print("\nRunning bootstrap comparison...")

    bootstrap_results, bootstrap_diffs = bootstrap_macro_f1_comparison(
        y_true=bert_y_true,
        bert_pred=bert_y_pred,
        pca_pred=test_pred,
        n_bootstrap=1000,
    )

    print("\n=== BOOTSTRAP RESULTS ===")
    print(bootstrap_results)

    with open("outputs/bootstrap_results.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)

    np.save("outputs/bootstrap_differences.npy", bootstrap_diffs)

    print("\nSaved bootstrap outputs:")
    print("outputs/bootstrap_results.json")
    print("outputs/bootstrap_differences.npy")

    print("\nRunning target-group analysis...")

    test_df_with_groups = add_group_columns(test_df, target_col="majority")

    group_metrics_df = evaluate_group_metrics(
        test_df=test_df_with_groups,
        bert_pred=bert_y_pred,
        pca_pred=test_pred,
        y_true=y_test,
        min_group_size=30,
    )

    group_metrics_df.to_csv("outputs/group_metrics.csv", index=False)

    plot_group_macro_f1(
    group_metrics_df,
    save_path="outputs/group_macro_f1_comparison.png"
)

    print("Saved group visualization to:")
    print("outputs/group_macro_f1_comparison.png")

    print("\n=== GROUP METRICS ===")
    print(group_metrics_df)

    print("\nSaved group metrics to:")
    print("outputs/group_metrics.csv")


if __name__ == "__main__":
    main()