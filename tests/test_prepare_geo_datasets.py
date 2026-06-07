import tempfile
import unittest
import gzip
from pathlib import Path

import pandas as pd


class PrepareGeoDatasetsTests(unittest.TestCase):
    def test_parse_series_matrix_reads_metadata_and_expression_table(self):
        from scripts.data.prepare_geo_datasets import parse_series_matrix

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "GSETEST_series_matrix.txt"
            path.write_text(
                "\n".join(
                    [
                        '!Series_platform_id\t"GPLTEST"',
                        '!Sample_title\t"control sample"\t"AD sample"',
                        '!Sample_geo_accession\t"GSM1"\t"GSM2"',
                        '!Sample_characteristics_ch1\t"disease state: normal"\t"disease state: Alzheimer\'s Disease"',
                        "!series_matrix_table_begin",
                        '"ID_REF"\t"GSM1"\t"GSM2"',
                        '"probe_a"\t1.0\t3.0',
                        '"probe_b"\t2.0\t4.0',
                        "!series_matrix_table_end",
                    ]
                ),
                encoding="utf-8",
            )

            parsed = parse_series_matrix(path)

        self.assertEqual(parsed.platform_id, "GPLTEST")
        self.assertEqual(parsed.expression.shape, (2, 2))
        self.assertEqual(parsed.expression.index.tolist(), ["probe_a", "probe_b"])
        self.assertEqual(parsed.expression.columns.tolist(), ["GSM1", "GSM2"])
        self.assertEqual(parsed.metadata["sample_id"].tolist(), ["GSM1", "GSM2"])
        self.assertEqual(parsed.metadata["disease state"].tolist(), ["normal", "Alzheimer's Disease"])

    def test_aggregate_probe_matrix_by_symbol_averages_duplicate_gene_symbols(self):
        from scripts.data.prepare_geo_datasets import aggregate_probe_matrix_by_symbol

        expression = pd.DataFrame(
            {"GSM1": [1.0, 3.0, 10.0], "GSM2": [5.0, 7.0, 20.0]},
            index=["probe_a", "probe_b", "probe_missing"],
        )
        probe_to_symbol = {"probe_a": "GENE1", "probe_b": "GENE1"}

        aggregated = aggregate_probe_matrix_by_symbol(expression, probe_to_symbol)

        self.assertEqual(aggregated.index.tolist(), ["GENE1"])
        self.assertEqual(aggregated.loc["GENE1", "GSM1"], 2.0)
        self.assertEqual(aggregated.loc["GENE1", "GSM2"], 6.0)

    def test_build_sample_labels_excludes_non_binary_groups(self):
        from scripts.data.prepare_geo_datasets import build_sample_labels

        metadata = pd.DataFrame(
            {
                "sample_id": ["GSM1", "GSM2", "GSM3", "GSM4"],
                "disease state": ["control", "AD", "AsymAD", "normal"],
                "title": ["", "", "", ""],
            }
        )

        labels = build_sample_labels("GSE118553", metadata)

        self.assertEqual(labels["sample_id"].tolist(), ["GSM1", "GSM2", "GSM4"])
        self.assertEqual(labels["label"].tolist(), ["control", "positive", "control"])

    def test_load_probe_to_symbol_reads_geo_annotation_table(self):
        from scripts.data.prepare_geo_datasets import load_probe_to_symbol

        with tempfile.TemporaryDirectory() as tmp_dir:
            annotation_dir = Path(tmp_dir)
            annotation_path = annotation_dir / "GPLTEST.annot.gz"
            with gzip.open(annotation_path, "wt", encoding="utf-8") as file_obj:
                file_obj.write("^Annotation\n")
                file_obj.write("!Annotation_platform = GPLTEST\n")
                file_obj.write("#Gene symbol = Entrez Gene symbol\n")
                file_obj.write("!platform_table_begin\n")
                file_obj.write("ID\tGene symbol\tOther\n")
                file_obj.write("probe_a\tGENE1///GENE2\tignored\n")
                file_obj.write("probe_b\t---\tignored\n")
                file_obj.write("probe_c\tGENE3\tignored\n")
                file_obj.write("!platform_table_end\n")

            mapping = load_probe_to_symbol("GPLTEST", annotation_dir)

        self.assertEqual(mapping, {"probe_a": "GENE1", "probe_c": "GENE3"})


if __name__ == "__main__":
    unittest.main()
