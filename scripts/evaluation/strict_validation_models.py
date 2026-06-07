from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

import config
from common import fit_rank_gauss_preprocessor, tensor_to_numpy, transform_with_preprocessor


DEFAULT_TRANSFORMER_PARAMS = {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dropout": 0.18,
    "lr": 7e-4,
    "batch_size": 32,
    "weight_decay": 0.006,
    "max_epochs": 180,
    "patience": 24,
}


@dataclass(frozen=True)
class FoldModelOptions:
    include_transformer: bool = True
    include_svm: bool = True
    transformer_seeds: Sequence[int] = (42, 49, 84)
    svm_inner_splits: int = 5
    svm_inner_repeats: int = 3
    threshold_splits: int = 5


def _best_threshold(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def _fit_transformer_ensemble(X_train, y_train, X_val, options):
    from scripts.training.train_transformer import (
        build_transformer,
        cross_validate_transformer,
        train_single_model,
    )

    params = DEFAULT_TRANSFORMER_PARAMS.copy()
    cv_result = cross_validate_transformer(
        X_train,
        y_train,
        params,
        plot_prefix="strict_fold_transformer",
        seeds=list(options.transformer_seeds),
    )

    preprocessor, X_train_processed = fit_rank_gauss_preprocessor(X_train)
    X_val_processed = transform_with_preprocessor(preprocessor, X_val)
    val_tensor = torch.FloatTensor(X_val_processed).to(config.DEVICE)
    probs = []

    for seed in options.transformer_seeds:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        train_idx, holdout_idx = train_test_split(
            np.arange(len(y_train)),
            test_size=0.15,
            stratify=y_train,
            random_state=int(seed),
        )
        model = build_transformer(X_train_processed.shape[1], params)
        model, _, _, _ = train_single_model(
            model,
            X_train_processed[train_idx],
            y_train[train_idx],
            X_train_processed[holdout_idx],
            y_train[holdout_idx],
            params,
        )
        model.eval()
        with torch.no_grad():
            logits = model(val_tensor)
            probs.append(tensor_to_numpy(torch.softmax(logits, dim=1)[:, 1]))

    return {
        "prob": np.mean(probs, axis=0),
        "threshold": float(cv_result["best_threshold"]),
    }


def _svm_oof_threshold(X_train_raw, y_train, best_params, n_splits):
    from scripts.training.train_svm import build_svm_model

    n_splits = min(int(n_splits), int(np.bincount(y_train).min()))
    if n_splits < 2:
        return 0.5

    oof_prob = np.zeros(len(y_train), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in cv.split(X_train_raw, y_train):
        preprocessor, X_inner_train = fit_rank_gauss_preprocessor(X_train_raw[train_idx])
        X_inner_val = transform_with_preprocessor(preprocessor, X_train_raw[val_idx])
        model = build_svm_model(best_params)
        model.fit(X_inner_train, y_train[train_idx])
        oof_prob[val_idx] = model.predict_proba(X_inner_val)[:, 1]

    return _best_threshold(y_train, oof_prob)


def _fit_svm_model(X_train, y_train, X_val, options):
    from scripts.training.train_svm import build_svm_model, select_best_svm_via_grid

    preprocessor, X_train_processed = fit_rank_gauss_preprocessor(X_train)
    X_val_processed = transform_with_preprocessor(preprocessor, X_val)
    best_params, _, _, _ = select_best_svm_via_grid(
        X_train_processed,
        y_train,
        cv_splits=min(options.svm_inner_splits, int(np.bincount(y_train).min())),
        cv_repeats=options.svm_inner_repeats,
        save_results_path=None,
        log_title="Strict fold inner SVM search",
    )
    threshold = _svm_oof_threshold(
        X_train,
        y_train,
        best_params,
        options.threshold_splits,
    )
    model = build_svm_model(best_params)
    model.fit(X_train_processed, y_train)
    return {
        "prob": model.predict_proba(X_val_processed)[:, 1],
        "threshold": threshold,
    }


def evaluate_fold_models(
    X_train_df,
    y_train,
    X_val_df,
    y_val,
    genes,
    fold,
    options=FoldModelOptions(),
):
    available_genes = [gene for gene in genes if gene in X_train_df.columns and gene in X_val_df.columns]
    if not available_genes:
        raise ValueError(f"{fold.name} has no selected genes available for model evaluation.")

    X_train = X_train_df[available_genes].to_numpy(dtype=np.float32)
    X_val = X_val_df[available_genes].to_numpy(dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int)

    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    if options.include_transformer:
        outputs["Transformer"] = _fit_transformer_ensemble(X_train, y_train, X_val, options)
    if options.include_svm:
        outputs["SVM"] = _fit_svm_model(X_train, y_train, X_val, options)
    return outputs
