import ast
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


TARGET_GROUPS = [
    "African",
    "Islam",
    "Jewish",
    "Gay",
    "Women",
    "Refugee",
    "Arab",
    "Caucasian",
    "Hispanic",
    "Asian",
]


GROUP_ALIASES = {
    "Homosexual": "Gay",
    "LGBTQ": "Gay",
    "Refugee/Immigrant": "Refugee",
}


def parse_target_list(value):
    """
    Safely parse target-group columns like:
    "['Islam', 'Women']"
    or already-existing Python lists.
    """
    if isinstance(value, list):
        targets = value
    elif pd.isna(value):
        targets = []
    else:
        try:
            targets = ast.literal_eval(value)
        except Exception:
            targets = []

    cleaned = []

    for target in targets:
        target = str(target).strip()
        target = GROUP_ALIASES.get(target, target)

        if target in TARGET_GROUPS:
            cleaned.append(target)

    return list(set(cleaned))


def add_group_columns(df, target_col="majority"):
    """
    Adds one binary column per target group.
    Example:
        group_Islam = 1 if Islam is in majority targets
    """
    df = df.copy()

    df["parsed_targets"] = df[target_col].apply(parse_target_list)

    for group in TARGET_GROUPS:
        df[f"group_{group}"] = df["parsed_targets"].apply(
            lambda targets: int(group in targets)
        )

    return df


def evaluate_group_metrics(
    test_df,
    bert_pred,
    pca_pred,
    y_true,
    min_group_size=30,
):
    rows = []

    for group in TARGET_GROUPS:
        group_col = f"group_{group}"

        if group_col not in test_df.columns:
            continue

        mask = test_df[group_col].to_numpy().astype(bool)
        n = mask.sum()

        if n < min_group_size:
            continue

        for model_name, preds in [
            ("BERT Head", bert_pred),
            ("PCA + LogReg", pca_pred),
        ]:
            rows.append({
                "group": group,
                "model": model_name,
                "n": int(n),
                "accuracy": accuracy_score(y_true[mask], preds[mask]),
                "macro_f1": f1_score(y_true[mask], preds[mask], average="macro"),
            })

    results_df = pd.DataFrame(rows)

    if not results_df.empty:
        pivot = results_df.pivot(
            index="group",
            columns="model",
            values="macro_f1"
        )

        if "PCA + LogReg" in pivot.columns and "BERT Head" in pivot.columns:
            pivot["macro_f1_diff_pca_minus_bert"] = (
                pivot["PCA + LogReg"] - pivot["BERT Head"]
            )

        results_df = results_df.merge(
            pivot[["macro_f1_diff_pca_minus_bert"]],
            left_on="group",
            right_index=True,
            how="left",
        )

    return results_df


def _macro_f1(y_true, y_pred, labels):
    return f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=labels,
        zero_division=0,
    )


def _accuracy(y_true, y_pred, labels):
    return accuracy_score(y_true, y_pred)


def _weighted_std(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(values) == 0 or np.sum(weights) == 0:
        return np.nan

    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    return float(np.sqrt(variance))


def _benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    valid_mask = ~np.isnan(p_values)

    if not valid_mask.any():
        return adjusted

    valid_p = p_values[valid_mask]
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    m = len(ranked_p)

    adjusted_ranked = ranked_p * m / np.arange(1, m + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)

    valid_adjusted = np.empty_like(valid_p)
    valid_adjusted[order] = adjusted_ranked
    adjusted[valid_mask] = valid_adjusted

    return adjusted


def bootstrap_group_delta_contrasts(
    test_df,
    bert_pred,
    pca_pred,
    y_true,
    group_metrics_df,
    n_bootstrap=1000,
    random_state=42,
    min_group_size=30,
    alpha=0.05,
):
    """
    Paired row-level bootstrap for differences in model delta Macro F1 across groups.

    Rows are resampled from the full test split, not independently within each group,
    so examples that belong to multiple target groups keep their overlapping
    memberships inside each bootstrap replicate.
    """
    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    bert_pred = np.asarray(bert_pred)
    pca_pred = np.asarray(pca_pred)
    labels = np.unique(y_true)
    n_rows = len(y_true)

    group_cols = {
        group: f"group_{group}"
        for group in TARGET_GROUPS
        if f"group_{group}" in test_df.columns
    }
    groups = [
        group
        for group, group_col in group_cols.items()
        if int(test_df[group_col].sum()) >= min_group_size
    ]

    observed_deltas = {}
    group_delta_samples = {group: [] for group in groups}
    group_masks = {
        group: test_df[group_cols[group]].to_numpy().astype(bool)
        for group in groups
    }

    for group, mask in group_masks.items():
        observed_deltas[group] = (
            _macro_f1(y_true[mask], pca_pred[mask], labels)
            - _macro_f1(y_true[mask], bert_pred[mask], labels)
        )

    pair_keys = list(itertools.combinations(groups, 2))
    pair_samples = {pair_key: [] for pair_key in pair_keys}

    for _ in range(n_bootstrap):
        idx = rng.choice(n_rows, size=n_rows, replace=True)
        sample_group_deltas = {}

        for group in groups:
            group_mask = group_masks[group][idx]

            if int(group_mask.sum()) < min_group_size:
                sample_group_deltas[group] = np.nan
                continue

            sample_idx = idx[group_mask]
            sample_group_deltas[group] = (
                _macro_f1(y_true[sample_idx], pca_pred[sample_idx], labels)
                - _macro_f1(y_true[sample_idx], bert_pred[sample_idx], labels)
            )

        for group, delta in sample_group_deltas.items():
            group_delta_samples[group].append(delta)

        for group_a, group_b in pair_keys:
            delta_a = sample_group_deltas[group_a]
            delta_b = sample_group_deltas[group_b]

            if np.isnan(delta_a) or np.isnan(delta_b):
                pair_samples[(group_a, group_b)].append(np.nan)
            else:
                pair_samples[(group_a, group_b)].append(delta_a - delta_b)

    group_summary_rows = []
    for group in groups:
        samples = np.asarray(group_delta_samples[group], dtype=float)
        valid_samples = samples[~np.isnan(samples)]

        if len(valid_samples) == 0:
            ci_lower = np.nan
            ci_upper = np.nan
            bootstrap_mean = np.nan
        else:
            ci_lower = float(np.percentile(valid_samples, 100 * alpha / 2))
            ci_upper = float(np.percentile(valid_samples, 100 * (1 - alpha / 2)))
            bootstrap_mean = float(np.mean(valid_samples))

        group_summary_rows.append({
            "group": group,
            "delta_macro_f1_observed": float(observed_deltas[group]),
            "delta_macro_f1_bootstrap_mean": bootstrap_mean,
            "delta_macro_f1_ci_lower": ci_lower,
            "delta_macro_f1_ci_upper": ci_upper,
            "delta_macro_f1_bootstrap_valid_n": int(len(valid_samples)),
        })

    group_delta_summary_df = pd.DataFrame(group_summary_rows)
    group_metrics_with_significance_df = group_metrics_df.merge(
        group_delta_summary_df,
        on="group",
        how="left",
    )

    pair_rows = []
    for group_a, group_b in pair_keys:
        samples = np.asarray(pair_samples[(group_a, group_b)], dtype=float)
        valid_samples = samples[~np.isnan(samples)]
        observed_contrast = observed_deltas[group_a] - observed_deltas[group_b]
        mask_a = group_masks[group_a]
        mask_b = group_masks[group_b]
        overlap_n = int(np.logical_and(mask_a, mask_b).sum())
        min_group_n = int(min(mask_a.sum(), mask_b.sum()))

        if len(valid_samples) == 0:
            ci_lower = np.nan
            ci_upper = np.nan
            p_value = np.nan
            bootstrap_mean = np.nan
        else:
            ci_lower = float(np.percentile(valid_samples, 100 * alpha / 2))
            ci_upper = float(np.percentile(valid_samples, 100 * (1 - alpha / 2)))
            p_lower = np.mean(valid_samples <= 0)
            p_upper = np.mean(valid_samples >= 0)
            p_value = float(min(1.0, 2 * min(p_lower, p_upper)))
            bootstrap_mean = float(np.mean(valid_samples))

        pair_rows.append({
            "group_a": group_a,
            "group_b": group_b,
            "n_group_a": int(mask_a.sum()),
            "n_group_b": int(mask_b.sum()),
            "overlap_n": overlap_n,
            "overlap_fraction_of_smaller_group": (
                float(overlap_n / min_group_n) if min_group_n else np.nan
            ),
            "delta_macro_f1_group_a": float(observed_deltas[group_a]),
            "delta_macro_f1_group_b": float(observed_deltas[group_b]),
            "delta_difference_group_a_minus_group_b": float(observed_contrast),
            "delta_difference_bootstrap_mean": bootstrap_mean,
            "delta_difference_ci_lower": ci_lower,
            "delta_difference_ci_upper": ci_upper,
            "p_value_two_sided_bootstrap": p_value,
            "bootstrap_valid_n": int(len(valid_samples)),
        })

    pairwise_df = pd.DataFrame(pair_rows)

    if not pairwise_df.empty:
        pairwise_df["p_value_fdr_bh"] = _benjamini_hochberg(
            pairwise_df["p_value_two_sided_bootstrap"].to_numpy()
        )
        pairwise_df["significant_fdr_0_05"] = pairwise_df["p_value_fdr_bh"] < alpha
        pairwise_df["ci_excludes_zero"] = (
            (pairwise_df["delta_difference_ci_lower"] > 0)
            | (pairwise_df["delta_difference_ci_upper"] < 0)
        )

    return group_metrics_with_significance_df, pairwise_df


def permutation_group_performance_heterogeneity(
    test_df,
    bert_pred,
    pca_pred,
    y_true,
    n_permutations=1000,
    random_state=42,
    min_group_size=30,
):
    """
    Permutation test for whether model performance differs across target groups.

    The full target-group membership vector is shuffled across test rows. This
    preserves group sizes and overlap patterns while removing the association
    between target-group membership and model performance.
    """
    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    bert_pred = np.asarray(bert_pred)
    pca_pred = np.asarray(pca_pred)
    labels = np.unique(y_true)

    group_cols = [
        f"group_{group}"
        for group in TARGET_GROUPS
        if f"group_{group}" in test_df.columns
    ]
    groups = [group_col.replace("group_", "") for group_col in group_cols]
    membership = test_df[group_cols].to_numpy().astype(bool)
    group_counts = membership.sum(axis=0).astype(int)
    eligible_mask = group_counts >= min_group_size
    eligible_groups = [
        group for group, is_eligible in zip(groups, eligible_mask) if is_eligible
    ]
    eligible_counts = group_counts[eligible_mask]
    membership = membership[:, eligible_mask]

    models = {
        "BERT Head": bert_pred,
        "PCA + LogReg": pca_pred,
    }
    metrics = {
        "accuracy": _accuracy,
        "macro_f1": _macro_f1,
    }

    observed = {}
    permutation_stats = {}

    for model_name, preds in models.items():
        for metric_name, metric_func in metrics.items():
            group_scores = []

            for group_idx in range(membership.shape[1]):
                mask = membership[:, group_idx]
                group_scores.append(metric_func(y_true[mask], preds[mask], labels))

            group_scores = np.asarray(group_scores, dtype=float)

            for statistic_name, statistic_value in [
                ("range", float(np.max(group_scores) - np.min(group_scores))),
                ("weighted_sd", _weighted_std(group_scores, eligible_counts)),
            ]:
                key = (model_name, metric_name, statistic_name)
                observed[key] = statistic_value
                permutation_stats[key] = []

    for _ in range(n_permutations):
        shuffled_membership = membership[rng.permutation(len(y_true)), :]

        for model_name, preds in models.items():
            for metric_name, metric_func in metrics.items():
                group_scores = []

                for group_idx in range(shuffled_membership.shape[1]):
                    mask = shuffled_membership[:, group_idx]
                    group_scores.append(
                        metric_func(y_true[mask], preds[mask], labels)
                    )

                group_scores = np.asarray(group_scores, dtype=float)

                permutation_stats[(model_name, metric_name, "range")].append(
                    float(np.max(group_scores) - np.min(group_scores))
                )
                permutation_stats[(model_name, metric_name, "weighted_sd")].append(
                    _weighted_std(group_scores, eligible_counts)
                )

    rows = []
    for key, observed_value in observed.items():
        model_name, metric_name, statistic_name = key
        null_values = np.asarray(permutation_stats[key], dtype=float)
        valid_null_values = null_values[~np.isnan(null_values)]

        if len(valid_null_values) == 0:
            p_value = np.nan
            null_mean = np.nan
            null_ci_lower = np.nan
            null_ci_upper = np.nan
        else:
            p_value = float(np.mean(valid_null_values >= observed_value))
            null_mean = float(np.mean(valid_null_values))
            null_ci_lower = float(np.percentile(valid_null_values, 2.5))
            null_ci_upper = float(np.percentile(valid_null_values, 97.5))

        rows.append({
            "model": model_name,
            "metric": metric_name,
            "heterogeneity_statistic": statistic_name,
            "observed_value": float(observed_value),
            "permutation_null_mean": null_mean,
            "permutation_null_ci_lower": null_ci_lower,
            "permutation_null_ci_upper": null_ci_upper,
            "p_value": p_value,
            "n_permutations": int(n_permutations),
            "n_groups": int(len(eligible_groups)),
            "groups_included": ",".join(eligible_groups),
            "min_group_size": int(min_group_size),
        })

    results_df = pd.DataFrame(rows)

    if not results_df.empty:
        results_df["p_value_fdr_bh"] = _benjamini_hochberg(
            results_df["p_value"].to_numpy()
        )
        results_df["significant_fdr_0_05"] = results_df["p_value_fdr_bh"] < 0.05

    return results_df
