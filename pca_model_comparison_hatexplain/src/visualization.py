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


def plot_2d_pca_scatter(
    X,
    y,
    labels,
    save_path,
    pc_x=1,
    pc_y=2,
    sample_size=3000,
    random_state=42,
):
    if pc_x < 1 or pc_y < 1:
        raise ValueError("Principal component axes are 1-indexed and must be >= 1.")

    if len(X) > sample_size:
        import numpy as np
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_plot = X[idx]
        y_plot = y[idx]
    else:
        X_plot = X
        y_plot = y

    pca = PCA(n_components=max(pc_x, pc_y), random_state=random_state)
    X_pca = pca.fit_transform(X_plot)

    x_idx = pc_x - 1
    y_idx = pc_y - 1

    plt.figure(figsize=(8, 6))

    for label_id, label_name in labels.items():
        mask = y_plot == label_id
        plt.scatter(
            X_pca[mask, x_idx],
            X_pca[mask, y_idx],
            label=label_name,
            alpha=0.6,
            s=12
        )

    plt.title(f"PCA Projection of CLS Embeddings: PC{pc_x} vs PC{pc_y}")
    plt.xlabel(f"PC{pc_x}")
    plt.ylabel(f"PC{pc_y}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pca_performance_vs_k(results_df, save_path):
    pca_df = results_df.dropna(subset=["pca_dim"]).copy()
    pca_df = pca_df.sort_values("pca_dim")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    plots = [
        ("accuracy", "Accuracy", axes[0]),
        ("macro_f1", "Macro F1", axes[1]),
    ]

    for metric, title, ax in plots:
        ax.plot(
            pca_df["pca_dim"],
            pca_df[f"train_{metric}"],
            marker="o",
            label="Train",
        )
        ax.plot(
            pca_df["pca_dim"],
            pca_df[f"test_{metric}"],
            marker="o",
            label="Test",
        )
        ax.set_title(f"{title} vs PCA k")
        ax.set_xlabel("PCA components (k)")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_pca_target_groups_multicolor(
    X,
    group_df,
    target_groups,
    save_path,
    pc_x=1,
    pc_y=2,
    sample_size=3000,
    random_state=42,
):
    if pc_x < 1 or pc_y < 1:
        raise ValueError("Principal component axes are 1-indexed and must be >= 1.")

    if len(X) > sample_size:
        import numpy as np
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_plot = X[idx]
        group_plot_df = group_df.iloc[idx].reset_index(drop=True)
    else:
        X_plot = X
        group_plot_df = group_df.reset_index(drop=True)

    group_cols = [f"group_{group}" for group in target_groups]
    missing_cols = [col for col in group_cols if col not in group_plot_df.columns]

    if missing_cols:
        raise ValueError(f"Missing target-group columns: {missing_cols}")

    pca = PCA(n_components=max(pc_x, pc_y), random_state=random_state)
    X_pca = pca.fit_transform(X_plot)

    x_idx = pc_x - 1
    y_idx = pc_y - 1
    colors = plt.get_cmap("tab10")
    any_group_mask = group_plot_df[group_cols].any(axis=1).to_numpy()

    plt.figure(figsize=(9, 7))

    plt.scatter(
        X_pca[~any_group_mask, x_idx],
        X_pca[~any_group_mask, y_idx],
        label="No tracked target",
        color="lightgray",
        alpha=0.25,
        s=10,
    )

    for idx, group in enumerate(target_groups):
        group_col = f"group_{group}"
        mask = group_plot_df[group_col].to_numpy().astype(bool)

        plt.scatter(
            X_pca[mask, x_idx],
            X_pca[mask, y_idx],
            label=group,
            color=colors(idx),
            alpha=0.65,
            s=14,
        )

    plt.title(f"PCA Projection by Target Group: PC{pc_x} vs PC{pc_y}")
    plt.xlabel(f"PC{pc_x}")
    plt.ylabel(f"PC{pc_y}")
    plt.legend(markerscale=1.4, fontsize=8, ncol=2)
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
