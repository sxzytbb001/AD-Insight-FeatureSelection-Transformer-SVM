import argparse
import os

import pandas as pd

import config
from common import load_training_matrix_and_labels
from apps.evaluation.generalization_protocol import (
    build_stratified_outer_folds,
    run_fold_protocol,
)
from apps.evaluation.strict_validation_models import FoldModelOptions, evaluate_fold_models
from apps.preprocessing.feature_selection_core import run_ensemble_feature_selection


NESTED_VALIDATION_DIR = os.path.join(config.RESULTS_DIR, "nested_internal_validation")


def build_nested_validation_folds(y, n_splits=5, seeds=(42, 49, 84, 123, 256)):
    return build_stratified_outer_folds(y, n_splits=n_splits, seeds=seeds)


def _select_genes_factory(options):
    def select_genes(X_train, y_train, labels_train):
        result = run_ensemble_feature_selection(
            X_train,
            y_train,
            candidate_gene_count=options.candidate_gene_count,
            prefilter_top_n=options.prefilter_top_n,
            method_top_k=options.method_top_k,
            stability_iterations=options.stability_iterations,
        )
        return result.candidate_genes

    return select_genes


def _evaluate_models_factory(options):
    model_options = FoldModelOptions(
        include_transformer=not options.skip_transformer,
        include_svm=not options.skip_svm,
        transformer_seeds=tuple(options.transformer_seeds),
        svm_inner_splits=options.svm_inner_splits,
        svm_inner_repeats=options.svm_inner_repeats,
        threshold_splits=options.threshold_splits,
    )

    def evaluate_models(X_train, y_train, X_val, y_val, genes, fold):
        return evaluate_fold_models(
            X_train,
            y_train,
            X_val,
            y_val,
            genes,
            fold,
            options=model_options,
        )

    return evaluate_models


def run_nested_internal_validation(options=None):
    if options is None:
        options = build_arg_parser().parse_args([])

    gene_matrix, labels_df, y = load_training_matrix_and_labels()
    folds = build_nested_validation_folds(
        y,
        n_splits=options.outer_splits,
        seeds=tuple(options.outer_seeds),
    )

    summary, predictions, gene_frequency = run_fold_protocol(
        gene_matrix,
        y,
        labels_df,
        folds,
        select_genes_fn=_select_genes_factory(options),
        evaluate_models_fn=_evaluate_models_factory(options),
    )

    os.makedirs(NESTED_VALIDATION_DIR, exist_ok=True)
    summary.to_csv(os.path.join(NESTED_VALIDATION_DIR, "nested_summary.csv"), index=False)
    predictions.to_csv(os.path.join(NESTED_VALIDATION_DIR, "nested_oof_predictions.csv"), index=False)
    gene_frequency.to_csv(os.path.join(NESTED_VALIDATION_DIR, "selected_gene_frequency.csv"), index=False)
    return summary


def _parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run strict nested internal validation.")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--outer-seeds", type=_parse_int_list, default="42,49,84,123,256")
    parser.add_argument("--transformer-seeds", type=_parse_int_list, default="42,49,84")
    parser.add_argument("--candidate-gene-count", type=int, default=30)
    parser.add_argument("--prefilter-top-n", type=int, default=1200)
    parser.add_argument("--method-top-k", type=int, default=100)
    parser.add_argument("--stability-iterations", type=int, default=50)
    parser.add_argument("--svm-inner-splits", type=int, default=5)
    parser.add_argument("--svm-inner-repeats", type=int, default=3)
    parser.add_argument("--threshold-splits", type=int, default=5)
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument("--skip-svm", action="store_true")
    return parser


def main(argv=None):
    options = build_arg_parser().parse_args(argv)
    run_nested_internal_validation(options)


if __name__ == "__main__":
    main()
