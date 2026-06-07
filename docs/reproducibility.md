# Reproducibility Guide

This guide records the commands needed to verify the repository and reproduce
the current AD case-study outputs.

## 1. Install Dependencies

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

If the repository uses LFS-tracked data or model artifacts:

```bash
git lfs install
git lfs pull
```

## 2. Run Tests

```bash
python -m pytest -q
```

## 3. Run The Full Pipeline

```bash
python main.py
```

This runs feature selection, Transformer training, SVM training, external
validation, and statistical analysis in order.

## 4. Refresh Selected Stages

Run feature selection only:

```bash
python -m scripts.preprocessing.feature_selection
```

Run model training:

```bash
python -m scripts.training.train_transformer
python -m scripts.training.train_svm
```

Run external validation and statistics without retraining:

```bash
python -m scripts.evaluation.external_validation
python -m scripts.analysis.statistical_analysis
```

Run strict sensitivity checks:

```bash
python -m scripts.evaluation.nested_internal_validation
python -m scripts.evaluation.loco_validation
```

## 5. External Label Polarity

External cohorts are configured in `external_datasets.json`. Use
`label_flip: true` only for cohorts whose label polarity has been confirmed to
be reversed relative to this project's `positive/control` convention. The
external validation outputs record `ConfiguredLabelFlip` and `LabelPolarity`.

## 6. Result Files To Check

| Output | Purpose |
|---|---|
| `results/feature_selection/candidate_genes.txt` | Final candidate gene panel. |
| `results/transformer/oof_predictions.csv` | Internal Transformer OOF predictions. |
| `results/svm/oof_predictions.csv` | Internal SVM OOF predictions. |
| `results/external_validation/external_validation_summary.csv` | Main external validation table. |
| `results/statistics/transformer_vs_svm_statistics.csv` | Statistical comparison table. |
| `docs/latest_results.md` | Human-readable result snapshot. |
