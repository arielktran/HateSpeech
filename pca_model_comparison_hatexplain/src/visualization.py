import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA


def plot_model_comparison(metrics_dict, save_path):
    names = list(metrics_dict.keys())
    accuracies = [metrics_dict[name]["accuracy"] for name in names]
    macro_f1s = [metrics_dict[name]["macro_f1"] for name in names]

    x = range(len(names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], accuracies, width, label="Accuracy")
    plt.bar([i + width / 2 for i in x], macro_f1s, width, label="Macro F1")

    plt.xticks(x, names, rotation=20)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_2d_pca_scatter(X, y, labels, save_path, sample_size=3000, random_state=42):
    if len(X) > sample_size:
        import numpy as np
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_plot = X[idx]
        y_plot = y[idx]
    else:
        X_plot = X
        y_plot = y

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_plot)

    plt.figure(figsize=(8, 6))

    for label_id, label_name in labels.items():
        mask = y_plot == label_id
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=label_name,
            alpha=0.6,
            s=12
        )

    plt.title("2D PCA Projection of CLS Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_group_macro_f1(group_metrics_df, save_path):
    pivot = group_metrics_df.pivot(
        index="group",
        columns="model",
        values="macro_f1"
    )

    pivot = pivot.sort_index()

    ax = pivot.plot(kind="bar", figsize=(10, 5))

    ax.set_title("Macro F1 by Target Group")
    ax.set_xlabel("Target Group")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.legend(title="Model")

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()