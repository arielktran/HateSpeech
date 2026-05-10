import ast
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