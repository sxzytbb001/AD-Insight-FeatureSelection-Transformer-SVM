import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class ExternalDatasetConfigTests(unittest.TestCase):
    def test_external_dataset_parser_accepts_path_objects_with_label_flip(self):
        from apps import config

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            flip_dir = root / "cohort_flip"
            plain_dir = root / "cohort_plain"
            flip_dir.mkdir()
            plain_dir.mkdir()
            config_path = root / "external_datasets.json"
            config_path.write_text(
                json.dumps(
                    {
                        "cohort_flip": {
                            "path": str(flip_dir),
                            "label_flip": True,
                            "notes": "s1/s2 labels are reversed",
                        },
                        "cohort_plain": str(plain_dir),
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(
                "os.environ",
                {
                    "GENE_EXPR_EXTERNALS_FILE": str(config_path),
                    "GENE_EXPR_EXTERNAL_DATASETS": "",
                },
            ):
                parsed = config._parse_external_datasets()

        self.assertEqual(parsed["cohort_flip"]["path"], str(flip_dir.resolve()))
        self.assertTrue(parsed["cohort_flip"]["label_flip"])
        self.assertEqual(parsed["cohort_flip"]["notes"], "s1/s2 labels are reversed")
        self.assertEqual(parsed["cohort_plain"]["path"], str(plain_dir.resolve()))
        self.assertFalse(parsed["cohort_plain"]["label_flip"])

    def test_external_validation_applies_configured_label_flip_before_metrics(self):
        from apps.evaluation.external_validation import _apply_external_label_options

        y_raw = np.array([0, 1, 1, 0])
        group_info = {"label_source": "sample_labels.csv", "control": 2, "positive": 2}

        y_flipped, updated_info = _apply_external_label_options(
            y_raw,
            group_info,
            {"label_flip": True, "label_flip_reason": "confirmed reversed s1/s2"},
        )

        np.testing.assert_array_equal(y_flipped, np.array([1, 0, 0, 1]))
        self.assertTrue(updated_info["ConfiguredLabelFlip"])
        self.assertEqual(updated_info["LabelFlipReason"], "confirmed reversed s1/s2")
        self.assertEqual(group_info["label_source"], "sample_labels.csv")


if __name__ == "__main__":
    unittest.main()
