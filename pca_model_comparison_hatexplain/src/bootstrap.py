import numpy as np
from sklearn.metrics import f1_score


def bootstrap_macro_f1_comparison(
    y_true,
    bert_pred,
    pca_pred,
    n_bootstrap=1000,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    bert_scores = []
    pca_scores = []
    diffs = []

    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)

        y_sample = y_true[idx]

        bert_sample = bert_pred[idx]
        pca_sample = pca_pred[idx]

        bert_f1 = f1_score(
            y_sample,
            bert_sample,
            average="macro"
        )

        pca_f1 = f1_score(
            y_sample,
            pca_sample,
            average="macro"
        )

        diff = pca_f1 - bert_f1

        bert_scores.append(bert_f1)
        pca_scores.append(pca_f1)
        diffs.append(diff)

    bert_scores = np.array(bert_scores)
    pca_scores = np.array(pca_scores)
    diffs = np.array(diffs)

    results = {
        "bert_mean_macro_f1": float(np.mean(bert_scores)),
        "pca_mean_macro_f1": float(np.mean(pca_scores)),
        "mean_difference": float(np.mean(diffs)),
        "diff_ci_lower": float(np.percentile(diffs, 2.5)),
        "diff_ci_upper": float(np.percentile(diffs, 97.5)),
    }

    return results, diffs