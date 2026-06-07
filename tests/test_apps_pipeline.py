import tempfile
import unittest
from pathlib import Path

import pandas as pd


class PipelineAppTests(unittest.TestCase):
    def test_build_pipeline_steps_respects_skip_options(self):
        from apps.pipeline import PipelineOptions, build_pipeline_steps

        options = PipelineOptions(
            skip_feature_selection=True,
            skip_transformer=False,
            skip_svm=True,
            skip_external_validation=False,
            skip_statistics=True,
        )

        steps = build_pipeline_steps(options)

        self.assertEqual(
            [(step.name, step.enabled) for step in steps],
            [
                ("Feature selection", False),
                ("Transformer training", True),
                ("SVM training", False),
                ("External validation", True),
                ("Statistical analysis", False),
            ],
        )

    def test_main_uses_app_pipeline_entrypoint(self):
        import apps.pipeline
        from apps import main

        self.assertIs(main.run_full_pipeline, apps.pipeline.run_full_pipeline)

    def test_gene_interaction_network_plot_writes_file(self):
        from apps.visualization import plot_gene_interaction_network, select_top_interactions

        matrix = pd.DataFrame(
            [
                [0.0, 0.4, -0.1],
                [0.4, 0.0, 0.2],
                [-0.1, 0.2, 0.0],
            ],
            index=["GeneA", "GeneB", "GeneC"],
            columns=["GeneA", "GeneB", "GeneC"],
        )

        top_edges = select_top_interactions(matrix, top_n=2)
        self.assertEqual(top_edges[0]["source"], "GeneA")
        self.assertEqual(top_edges[0]["target"], "GeneB")
        self.assertAlmostEqual(top_edges[0]["weight"], 0.4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "network.png"
            plot_gene_interaction_network(matrix, output_path, top_n=2)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
