import unittest
from unittest import mock

import numpy as np
import pandas as pd


class GeneralizationProtocolTests(unittest.TestCase):
    def test_label_reader_preserves_dataset_metadata_for_cohort_validation(self):
        from apps import common

        labels = pd.DataFrame(
            {
                "sample_id": ["S1", "S2"],
                "label": ["control", "positive"],
                "dataset": ["GSE_A", "GSE_B"],
            }
        )

        with mock.patch("pandas.read_csv", return_value=labels):
            parsed = common._read_label_table("labels.csv")

        self.assertEqual(parsed.columns.tolist(), ["sample_id", "label", "dataset"])

    def test_cohort_class_weights_balance_each_dataset_label_group(self):
        from apps.common import compute_cohort_class_sample_weights

        y = np.array([0, 0, 0, 1, 1, 0])
        cohorts = np.array(["A", "A", "A", "A", "B", "B"])

        weights = compute_cohort_class_sample_weights(y, cohorts)

        group_totals = {}
        for label, cohort, weight in zip(y, cohorts, weights):
            group_totals[(cohort, label)] = group_totals.get((cohort, label), 0.0) + weight
        self.assertEqual(set(group_totals), {("A", 0), ("A", 1), ("B", 0), ("B", 1)})
        self.assertTrue(all(abs(total - 1.0) < 1e-8 for total in group_totals.values()))

    def test_run_fold_protocol_refits_feature_selection_inside_each_outer_fold(self):
        from apps.evaluation.generalization_protocol import ValidationFold, run_fold_protocol

        X = pd.DataFrame(
            {
                "GeneA": [0.1, 0.2, 2.0, 2.1, 0.3, 2.2],
                "GeneB": [1.0, 1.1, 0.0, 0.1, 1.2, 0.2],
                "GeneC": [5.0, 5.1, 5.2, 5.3, 5.4, 5.5],
            },
            index=[f"S{i}" for i in range(6)],
        )
        y = np.array([0, 0, 1, 1, 0, 1])
        labels = pd.DataFrame(
            {
                "sample_id": X.index,
                "label": ["control", "control", "positive", "positive", "control", "positive"],
                "dataset": ["A", "A", "A", "B", "B", "B"],
            }
        )
        folds = [
            ValidationFold("fold0", np.array([0, 1, 2, 3]), np.array([4, 5])),
            ValidationFold("fold1", np.array([0, 2, 4, 5]), np.array([1, 3])),
        ]
        selector_calls = []

        def select_genes(X_train, y_train, labels_train):
            selector_calls.append(set(labels_train["sample_id"]))
            return ["GeneA", "GeneB"]

        def evaluate_models(X_train, y_train, X_val, y_val, genes, fold):
            self.assertEqual(genes, ["GeneA", "GeneB"])
            return {"MockModel": {"prob": np.linspace(0.2, 0.8, len(y_val)), "threshold": 0.5}}

        summary, predictions, gene_frequency = run_fold_protocol(
            X,
            y,
            labels,
            folds,
            select_genes_fn=select_genes,
            evaluate_models_fn=evaluate_models,
        )

        self.assertEqual(len(selector_calls), 2)
        self.assertFalse({"S4", "S5"} & selector_calls[0])
        self.assertFalse({"S1", "S3"} & selector_calls[1])
        self.assertEqual(summary["Fold"].tolist(), ["fold0", "fold1"])
        self.assertEqual(predictions["Model"].unique().tolist(), ["MockModel"])
        self.assertEqual(gene_frequency.set_index("gene").loc["GeneA", "selection_count"], 2)

    def test_loco_folds_fail_if_external_holdout_is_in_training_labels(self):
        from apps.evaluation.generalization_protocol import build_loco_folds

        labels = pd.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3", "S4"],
                "dataset": ["GSE1297", "GSE1297", "GSE48350", "GSE48350"],
            }
        )
        y = np.array([0, 1, 0, 1])

        with self.assertRaisesRegex(ValueError, "External holdout datasets"):
            build_loco_folds(labels, y, external_holdout_datasets={"GSE48350"})

    def test_loco_folds_leave_one_training_dataset_out(self):
        from apps.evaluation.generalization_protocol import build_loco_folds

        labels = pd.DataFrame(
            {
                "sample_id": ["A0", "A1", "B0", "B1", "C0", "C1"],
                "dataset": ["A", "A", "B", "B", "C", "C"],
            }
        )
        y = np.array([0, 1, 0, 1, 0, 1])

        folds = build_loco_folds(labels, y)

        self.assertEqual([fold.name for fold in folds], ["loco_A", "loco_B", "loco_C"])
        for fold in folds:
            held_out = set(labels.iloc[fold.validation_indices]["dataset"])
            train_datasets = set(labels.iloc[fold.train_indices]["dataset"])
            self.assertEqual(held_out, {fold.held_out_dataset})
            self.assertNotIn(fold.held_out_dataset, train_datasets)

    def test_loco_validation_uses_strict_external_holdout_defaults(self):
        from apps.evaluation.loco_validation import (
            EXTERNAL_HOLDOUT_DATASETS,
            build_loco_validation_folds,
        )

        self.assertEqual(
            EXTERNAL_HOLDOUT_DATASETS,
            {"GSE109887", "GSE118553", "GSE122063", "GSE48350"},
        )
        labels = pd.DataFrame(
            {
                "sample_id": ["A0", "A1", "B0", "B1"],
                "dataset": ["A", "A", "B", "B"],
            }
        )
        y = np.array([0, 1, 0, 1])

        folds = build_loco_validation_folds(labels, y)

        self.assertEqual([fold.name for fold in folds], ["loco_A", "loco_B"])

    def test_nested_validation_builds_seeded_outer_folds(self):
        from apps.evaluation.nested_internal_validation import (
            build_arg_parser,
            build_nested_validation_folds,
        )

        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        options = build_arg_parser().parse_args([])
        self.assertEqual(options.outer_seeds, [42, 49, 84, 123, 256])
        self.assertEqual(options.transformer_seeds, [42, 49, 84])

        folds = build_nested_validation_folds(y, n_splits=2, seeds=(11, 22))

        self.assertEqual(
            [fold.name for fold in folds],
            ["seed11_fold1", "seed11_fold2", "seed22_fold1", "seed22_fold2"],
        )
        for fold in folds:
            self.assertEqual(len(set(fold.train_indices) & set(fold.validation_indices)), 0)


if __name__ == "__main__":
    unittest.main()
