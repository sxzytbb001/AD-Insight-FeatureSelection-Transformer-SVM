from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class ValidationFold:
    name: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    held_out_dataset: Optional[str] = None


def _require_dataset_column(labels_df):
    if "dataset" not in labels_df.columns:
        raise ValueError("sample_labels.csv must contain a dataset column for cohort-level validation.")


def build_stratified_outer_folds(y, n_splits=5, seeds=(42,)):
    y = np.asarray(y, dtype=int)
    folds = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
            folds.append(
                ValidationFold(
                    name=f"seed{seed}_fold{fold_idx}",
                    train_indices=np.asarray(train_idx, dtype=int),
                    validation_indices=np.asarray(val_idx, dtype=int),
                )
            )
    return folds


def build_loco_folds(labels_df, y, external_holdout_datasets=None, min_classes_per_split=2):
    _require_dataset_column(labels_df)
    labels_df = labels_df.reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    external_holdout_datasets = set(external_holdout_datasets or [])
    training_datasets = set(labels_df["dataset"].astype(str).unique())
    leaked_holdouts = sorted(training_datasets & external_holdout_datasets)
    if leaked_holdouts:
        raise ValueError(
            "External holdout datasets are present in training labels and cannot be used "
            f"for LOCO model selection: {leaked_holdouts}"
        )

    folds = []
    for dataset_name in sorted(training_datasets):
        validation_mask = labels_df["dataset"].astype(str).to_numpy() == dataset_name
        train_idx = np.where(~validation_mask)[0]
        val_idx = np.where(validation_mask)[0]
        if len(np.unique(y[train_idx])) < min_classes_per_split:
            raise ValueError(f"LOCO train split for {dataset_name} does not contain both classes.")
        if len(np.unique(y[val_idx])) < min_classes_per_split:
            raise ValueError(f"LOCO validation split for {dataset_name} does not contain both classes.")
        folds.append(
            ValidationFold(
                name=f"loco_{dataset_name}",
                train_indices=train_idx,
                validation_indices=val_idx,
                held_out_dataset=dataset_name,
            )
        )
    return folds


def summarize_gene_frequency(selected_gene_lists):
    counts = {}
    for genes in selected_gene_lists:
        for gene in genes:
            counts[gene] = counts.get(gene, 0) + 1
    rows = [
        {
            "gene": gene,
            "selection_count": count,
            "selection_frequency": count / max(len(selected_gene_lists), 1),
        }
        for gene, count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["selection_count", "gene"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _metric_rows(fold, y_true, model_outputs):
    rows = []
    for model_name, output in model_outputs.items():
        y_prob = np.asarray(output["prob"], dtype=float)
        threshold = float(output.get("threshold", 0.5))
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "Fold": fold.name,
                "HeldOutDataset": fold.held_out_dataset or "",
                "Model": model_name,
                "Samples": int(len(y_true)),
                "AUC": float(roc_auc_score(y_true, y_prob)),
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "Threshold": threshold,
            }
        )
    return rows


def _prediction_rows(fold, labels_val, y_true, model_outputs):
    sample_ids = labels_val["sample_id"].tolist() if "sample_id" in labels_val.columns else list(labels_val.index)
    rows = []
    for model_name, output in model_outputs.items():
        y_prob = np.asarray(output["prob"], dtype=float)
        threshold = float(output.get("threshold", 0.5))
        y_pred = (y_prob >= threshold).astype(int)
        for i, sample_id in enumerate(sample_ids):
            rows.append(
                {
                    "Fold": fold.name,
                    "HeldOutDataset": fold.held_out_dataset or "",
                    "Model": model_name,
                    "sample_id": sample_id,
                    "y_true": int(y_true[i]),
                    "y_prob": float(y_prob[i]),
                    "y_pred": int(y_pred[i]),
                    "Threshold": threshold,
                }
            )
    return rows


def run_fold_protocol(
    X_df,
    y,
    labels_df,
    folds,
    select_genes_fn: Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], Sequence[str]],
    evaluate_models_fn: Callable[
        [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, Sequence[str], ValidationFold],
        Dict[str, Dict[str, np.ndarray]],
    ],
):
    X_df = pd.DataFrame(X_df).reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    labels_df = labels_df.reset_index(drop=True).copy()
    if len(X_df) != len(y) or len(labels_df) != len(y):
        raise ValueError("X_df, y, and labels_df must contain the same number of samples.")

    summary_rows = []
    prediction_rows = []
    selected_gene_lists = []

    for fold in folds:
        train_idx = np.asarray(fold.train_indices, dtype=int)
        val_idx = np.asarray(fold.validation_indices, dtype=int)
        X_train = X_df.iloc[train_idx].copy()
        y_train = y[train_idx]
        labels_train = labels_df.iloc[train_idx].copy()
        X_val = X_df.iloc[val_idx].copy()
        y_val = y[val_idx]
        labels_val = labels_df.iloc[val_idx].copy()

        selected_genes = list(select_genes_fn(X_train, y_train, labels_train))
        if not selected_genes:
            raise ValueError(f"{fold.name} selected no genes.")
        selected_gene_lists.append(selected_genes)

        model_outputs = evaluate_models_fn(
            X_train,
            y_train,
            X_val,
            y_val,
            selected_genes,
            fold,
        )
        summary_rows.extend(_metric_rows(fold, y_val, model_outputs))
        prediction_rows.extend(_prediction_rows(fold, labels_val, y_val, model_outputs))

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(prediction_rows),
        summarize_gene_frequency(selected_gene_lists),
    )
