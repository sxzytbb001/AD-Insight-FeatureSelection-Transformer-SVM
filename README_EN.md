# Gene Expression Classification Pipeline

A reproducible Python pipeline for binary classification on gene-expression
matrices. The case study in this repository focuses on Alzheimer's disease
case/control cohorts, but the code is dataset-agnostic and can be used with any
binary transcriptomic dataset that provides an expression matrix and sample
labels.

[简体中文](README.md)

The pipeline combines ensemble feature selection, a compact Transformer
classifier, SVM baselines, external cohort validation, strict sensitivity
validation, and statistical comparison.

## Highlights

- Matrix loader that accepts genes-as-rows or samples-as-rows input.
- Binary label normalization for common `control` and `positive` synonyms.
- Seven-method ensemble feature selection with Welch t-test, Mutual
  Information, XGBoost, Random Forest, ElasticNet, mRMR, and Stability
  Selection.
- Lightweight `TransformerV3` with attention maps, gate weights, and gene
  interaction outputs.
- Single SVM, Voting SVM, and Bagging SVM baselines.
- External validation with explicit per-cohort label-polarity configuration.
- Nested internal validation and leave-one-cohort-out validation utilities.
- DeLong, McNemar, bootstrap confidence intervals, and publication-style plots.

## Method Workflow

- **Data preparation**: GEO expression matrices are mapped from probes to gene
  symbols through platform annotation. Duplicate gene symbols are averaged.
  High-range expression values are transformed with `log2(x + 1)`, then
  z-scored by gene within each dataset.
- **Training set construction**: training cohorts are merged on their common
  gene intersection. Sample IDs receive dataset prefixes, and the label table
  keeps `dataset` and `source_sample_id` metadata.
- **Matrix loading**: training and validation loaders infer matrix orientation
  from `sample_id` matches and convert input to a samples-by-genes matrix.
  Non-numeric expression values are coerced to missing values and filled with
  `0.0`.
- **Label handling**: common AD/control labels are normalized to binary labels,
  with `positive` as 1 and `control` as 0.
- **Feature selection**: genes are prefiltered with Welch t-test, FDR, and
  effect-size statistics, then ranked with Mutual Information, XGBoost gain,
  Random Forest importance, ElasticNet Logistic Regression, mRMR, and Stability
  Selection. Method rankings are combined with validation-AUC-weighted voting
  into a 30-gene panel.
- **Transformer modeling**: candidate-gene expression is rank-gauss
  standardized and passed to the compact `TransformerV3`. The model combines a
  CLS token, gene gate pooling, raw linear projection, and second-order
  interaction factors. Training uses class weights, label smoothing, auxiliary
  focal loss, data augmentation, and multi-seed OOF ensembling.
- **SVM baselines**: grid search selects SVM parameters, followed by single SVM,
  Voting SVM, and Bagging SVM training.
- **Validation and statistics**: internal OOF, nested internal validation,
  leave-one-cohort-out validation, and external cohort validation produce AUC,
  accuracy, confusion matrices, ROC plots, DeLong tests, McNemar tests, and
  bootstrap confidence intervals.

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

Python 3.10 and 3.11 are supported. Full Transformer training can use a
CUDA-capable PyTorch environment, while the tests and most data utilities run on
CPU.

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

Large case-study matrices, model checkpoints, and selected result artifacts can
be managed with Git LFS:

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

The label table contains at least:

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

## Case-Study Datasets

The current AD case study uses public GEO datasets for training and external
validation.

| Dataset | Role | Platform | Samples | Control | Positive | Genes |
|---|---|---:|---:|---:|---:|---:|
| GSE1297 | train | GPL96 | 31 | 9 | 22 | 13100 |
| GSE33000 | train | preprocessed | 467 | 157 | 310 | 17402 |
| GSE36980 | train | GPL6244 | 80 | 47 | 33 | 20003 |
| GSE5281 | train | GPL570 | 161 | 74 | 87 | 21753 |
| GSE109887 | external | preprocessed | 78 | 46 | 32 | 31682 |
| GSE118553 | external | GPL10558 | 267 | 100 | 167 | 20759 |
| GSE122063 | external | preprocessed | 100 | 44 | 56 | 32074 |
| GSE48350 | external | GPL570 | 220 | 140 | 80 | 21753 |

The merged training set contains 739 samples, including 287 control samples and
452 positive samples, with 9981 common genes. `GSE29378` is kept as an
exploratory dataset and is not part of the default external validation config.

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
    "label_flip_reason": "Confirmed reversed label polarity."
  }
}
```

The short path-only form is still supported:

```json
{
  "cohort_a": "data/external/cohort_a"
}
```

Each external directory contains an expression matrix named one of:

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
folders. Explicit `sample_labels.csv` labels are trusted by default.
`label_flip: true` records a cohort whose label polarity is reversed relative to
this pipeline's `positive` and `control` convention. The resulting CSV outputs
record `ConfiguredLabelFlip` and `LabelPolarity`.

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

Important output files include:

- `results/feature_selection/candidate_genes.txt`
- `results/transformer/oof_predictions.csv`
- `results/svm/oof_predictions.csv`
- `results/external_validation/external_validation_summary.csv`
- `results/statistics/transformer_vs_svm_statistics.csv`
- `docs/latest_results.md`

## Current Case-Study Results

The current reproducible AD case-study snapshot is documented in
[docs/latest_results.md](docs/latest_results.md). The compact Transformer is
comparable to strong SVM baselines and provides additional interpretability
outputs, including attention maps, gate weights, and gene interaction matrices.

The current candidate-gene panel contains 30 genes in
[results/feature_selection/candidate_genes.txt](results/feature_selection/candidate_genes.txt),
including `ITPKB`, `NRN1`, `PPP1R7`, `NEUROD6`, `GFAP`, `SST`, `CD200`,
`NRXN3`, `VGF`, and `PTPRN2`.

Internal OOF validation:

| Model | AUC | Accuracy |
|---|---:|---:|
| Transformer | 0.9322 | 0.8769 |
| Logistic Regression | 0.9199 | 0.8444 |
| SVM | 0.9256 | 0.8498 |
| Voting SVM | 0.9316 | 0.8687 |
| Bagging SVM | 0.9269 | 0.8566 |

Strict generalization checks:

| Protocol | Model | Mean AUC | Mean Accuracy |
|---|---|---:|---:|
| Nested internal validation | Transformer | 0.9250 | 0.8468 |
| Nested internal validation | SVM | 0.9188 | 0.8366 |
| Leave-one-cohort-out validation | Transformer | 0.8000 | 0.7143 |
| Leave-one-cohort-out validation | SVM | 0.7950 | 0.6756 |

Primary external validation results use the `train_prior_quantile` threshold
strategy:

| Dataset | Transformer AUC | Transformer Accuracy | SVM AUC | SVM Accuracy |
|---|---:|---:|---:|---:|
| GSE109887 | 0.8770 | 0.7949 | 0.8601 | 0.7692 |
| GSE118553 | 0.6914 | 0.6854 | 0.6778 | 0.6554 |
| GSE122063 | 0.8482 | 0.6900 | 0.8369 | 0.7300 |
| GSE48350 | 0.6691 | 0.5636 | 0.6824 | 0.5636 |

`GSE109887` is configured with `label_flip: true` because its s1/s2 label
polarity was confirmed to be reversed for this pipeline's convention.

## Testing

Run the lightweight regression suite:

```bash
python -m pytest -q
```

The GitHub Actions workflow in [.github/workflows/tests.yml](.github/workflows/tests.yml)
runs the same suite on Python 3.10 and 3.11.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE).
