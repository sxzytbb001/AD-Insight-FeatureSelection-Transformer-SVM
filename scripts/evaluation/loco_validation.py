import argparse
import os

import config
from common import load_training_matrix_and_labels
from scripts.evaluation.generalization_protocol import build_loco_folds, run_fold_protocol
from scripts.evaluation.strict_validation_models import FoldModelOptions, evaluate_fold_models
from scripts.preprocessing.feature_selection_core import run_ensemble_feature_selection


LOCO_VALIDATION_DIR = os.path.join(config.RESULTS_DIR, "loco_validation")
EXTERNAL_HOLDOUT_DATASETS = {"GSE109887", "GSE118553", "GSE122063", "GSE48350"}


def build_loco_validation_folds(labels_df, y):
    return build_loco_folds(
        labels_df,
        y,
        external_holdout_datasets=EXTERNAL_HOLDOUT_DATASETS,
    )


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


def run_loco_validation(options=None):
    if options is None:
        options = build_arg_parser().parse_args([])

    gene_matrix, labels_df, y = load_training_matrix_and_labels()
    folds = build_loco_validation_folds(labels_df, y)

    summary, predictions, gene_frequency = run_fold_protocol(
        gene_matrix,
        y,
        labels_df,
        folds,
        select_genes_fn=_select_genes_factory(options),
        evaluate_models_fn=_evaluate_models_factory(options),
    )

    os.makedirs(LOCO_VALIDATION_DIR, exist_ok=True)
    summary.to_csv(os.path.join(LOCO_VALIDATION_DIR, "loco_summary.csv"), index=False)
    predictions.to_csv(os.path.join(LOCO_VALIDATION_DIR, "loco_predictions.csv"), index=False)
    gene_frequency.to_csv(os.path.join(LOCO_VALIDATION_DIR, "loco_gene_frequency.csv"), index=False)
    return summary


def _parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run leave-one-cohort-out validation.")
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
    run_loco_validation(options)


if __name__ == "__main__":
    main()
