# Gene Expression Classification Pipeline

A reproducible Python pipeline for binary classification on gene-expression
matrices. The case study in this repository focuses on Alzheimer's disease
case/control cohorts, but the code is dataset-agnostic and can be used with any
binary transcriptomic dataset that provides an expression matrix and sample
labels.

[简体中文说明](README_CN.md)

The pipeline combines ensemble feature selection, a compact Transformer
classifier, SVM baselines, external cohort validation, strict sensitivity
validation, and statistical comparison.

## Highlights

- Matrix loader that accepts genes-as-rows or samples-as-rows input.
- Binary label normalization for common `control` and `positive` synonyms.
- Seven-method ensemble feature selection for candidate-gene panels.
- Lightweight `TransformerV3` with attention maps, gate weights, and gene
  interaction outputs.
- Single SVM, Voting SVM, and Bagging SVM baselines.
- External validation with explicit per-cohort label-polarity configuration.
- Nested internal validation and leave-one-cohort-out validation utilities.
- DeLong, McNemar, bootstrap confidence intervals, and publication-style plots.

## Repository Layout

```text
.
|-- apps/                         # Pipeline orchestration and plotting helpers
|-- data/                         # Case-study matrices and dataset manifest
|-- docs/                         # Reproducibility notes and result snapshots
|-- results/                      # Optional generated result snapshots
|-- scripts/
|   |-- analysis/                 # Statistical comparison
|   |-- data/                     # GEO preparation helpers
|   |-- evaluation/               # External, nested, and LOCO validation
|   |-- preprocessing/            # Preprocessing and feature selection
|   `-- training/                 # Transformer and SVM training
|-- tests/                        # Lightweight regression tests
|-- main.py                       # Compatibility CLI entry point
|-- config.py                     # Paths, labels, artifacts, and environment config
|-- requirements.txt              # Runtime dependencies
`-- requirements-dev.txt          # Runtime dependencies plus test tools
```

See [docs/README.md](docs/README.md) for the documentation index.

## Installation

Use Python 3.10 or 3.11. A CUDA-capable PyTorch environment is useful for full
Transformer training, but the tests and most data utilities also run on CPU.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If you use the LFS-tracked case-study matrices or model/result artifacts, install
Git LFS before cloning or run:

```bash
git lfs install
git lfs pull
```

## Data Format

Training data defaults to:

```text
data/train/cleaned_gene_matrix.csv
data/train/sample_labels.csv
```

The expression matrix may be either:

- genes as rows and samples as columns; or
- samples as rows and genes as columns.

The loader detects orientation by matching `sample_id` values from the label
table. CSV, TSV, TXT, XLSX, and XLS files are supported.

The label table must contain at least:

```csv
sample_id,label
sample_001,control
sample_002,positive
```

Accepted positive labels include `1`, `true`, `positive`, `case`, `disease`,
`AD`, and `Alzheimer`. Accepted negative labels include `0`, `false`,
`negative`, `control`, `normal`, `healthy`, and `non-demented`.

For other label names, set:

```powershell
$env:GENE_EXPR_POSITIVE_LABEL="tumor"
$env:GENE_EXPR_NEGATIVE_LABEL="normal"
```

## External Validation Config

External cohorts are configured with `external_datasets.json`.

```json
{
  "cohort_a": {
    "path": "data/external/cohort_a"
  },
  "cohort_b": {
    "path": "D:/datasets/cohort_b",
    "label_flip": true,
    "label_flip_reason": "Use only after confirming reversed label polarity."
  }
}
```

The short path-only form is still supported:

```json
{
  "cohort_a": "data/external/cohort_a"
}
```

Each external directory should contain an expression matrix named one of:

```text
matrix.csv
expression_matrix.csv
gene_matrix.csv
cleaned_gene_matrix.csv
matrix.tsv
geneMatrix.txt
```

and a label table named one of:

```text
labels.csv
sample_labels.csv
labels.tsv
sample_labels.tsv
```

Legacy `s1.txt` and `s2.txt` group files are still supported for old GEO-style
folders. Explicit `sample_labels.csv` labels are trusted by default. Use
`label_flip: true` only after confirming that the cohort's label polarity is
reversed for this pipeline's `positive` and `control` convention. The resulting
CSV outputs record `ConfiguredLabelFlip` and `LabelPolarity`.

## Usage

Preprocess a dataset:

```bash
python -m scripts.preprocessing.preprocess --matrix raw_matrix.csv --labels labels.csv
```

Run the full pipeline:

```bash
python main.py
```

Run selected stages:

```bash
python -m scripts.preprocessing.feature_selection
python -m scripts.training.train_transformer
python -m scripts.training.train_svm
python -m scripts.evaluation.external_validation
python -m scripts.analysis.statistical_analysis
```

Strict validation utilities:

```bash
python -m scripts.evaluation.nested_internal_validation
python -m scripts.evaluation.loco_validation
```

## Configuration

Common environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `GENE_EXPR_DATA_DIR` | Base data directory | `data/` |
| `GENE_EXPR_TRAIN_DIR` | Training dataset directory | `data/train/` |
| `GENE_EXPR_TRAIN_MATRIX` | Training matrix path | `data/train/cleaned_gene_matrix.csv` |
| `GENE_EXPR_TRAIN_LABELS` | Training labels path | `data/train/sample_labels.csv` |
| `GENE_EXPR_RESULTS_DIR` | Output directory | `results/` |
| `GENE_EXPR_EXTERNALS_FILE` | External cohort JSON file | `external_datasets.json` |
| `GENE_EXPR_EXTERNAL_DATASETS` | Inline `NAME=PATH` external cohort mapping | unset |
| `GENE_EXPR_POSITIVE_LABEL` | Positive class name | `positive` |
| `GENE_EXPR_NEGATIVE_LABEL` | Negative class name | `control` |

If `torch` emits NumPy DLL warnings in your environment, use NumPy `<2.0`; this
range is already pinned in `requirements.txt`.

## Outputs

Generated files are written under `results/`:

```text
results/
|-- feature_selection/
|-- transformer/
|-- svm/
|-- external_validation/
|-- nested_internal_validation/
|-- loco_validation/
`-- statistics/
```

New generated outputs are ignored by default. This repository may keep selected
result snapshots and LFS-tracked model artifacts for reproducibility. For a
code-only public release, publish large matrices and checkpoints through Git LFS
or GitHub Releases, and keep only documentation plus small result summaries in
the main repository.

Important output files include:

- `results/feature_selection/candidate_genes.txt`
- `results/transformer/oof_predictions.csv`
- `results/svm/oof_predictions.csv`
- `results/external_validation/external_validation_summary.csv`
- `results/statistics/transformer_vs_svm_statistics.csv`
- `docs/latest_results.md`

## Current Case-Study Results

The current reproducible AD case-study snapshot is documented in
[docs/latest_results.md](docs/latest_results.md). The headline result is that the
compact Transformer is comparable to strong SVM baselines, with slightly higher
single-SVM point estimates in internal OOF, nested internal validation, and some
external cohorts. The external results are platform-dependent; do not claim that
the Transformer is uniformly superior across all cohorts.

`GSE109887` is configured with `label_flip: true` because its s1/s2 label
polarity was confirmed to be reversed for this pipeline's convention.

## Testing

Run the lightweight regression suite:

```bash
python -m pytest -q
```

The GitHub Actions workflow in [.github/workflows/tests.yml](.github/workflows/tests.yml)
runs the same suite on Python 3.10 and 3.11.

## Reproducibility Notes

- Feature selection should be refit inside strict validation folds when
  estimating generalization. The `nested_internal_validation` and
  `loco_validation` scripts do this.
- External validation thresholds are reported with the `train_prior_quantile`
  strategy by default, with fixed internal and retrospective Youden summaries
  saved as supplemental files.
- Avoid tuning model settings on final external cohorts.
- Verify GEO dataset licenses and privacy constraints before redistributing
  processed matrices or trained checkpoints.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Keep changes scoped, run the test suite,
and document any new data or result-producing command.

## Citation

If this repository supports academic work, cite the repository and the original
source datasets. A starter citation file is provided in [CITATION.cff](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE).
