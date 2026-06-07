# Data Split

This split keeps final validation datasets fully held out at the GEO cohort level.
Held-out datasets are not used for feature selection, model fitting, or hyperparameter selection.

## Roles

- `train`: merged into `data/train/` after per-dataset normalization and common-gene intersection.
- `external`: written to `data/external/<GSE>/` and referenced by `external_datasets.json`.
- `exploratory`: prepared for manual checks but excluded from the default final validation config.

## Training Summary

- Datasets: GSE1297, GSE33000, GSE36980, GSE5281
- Samples: 739 (287 control, 452 positive)
- Common genes: 9981

## Dataset Manifest

| Dataset | Role | Platform | Samples | Control | Positive | Genes |
|---|---|---:|---:|---:|---:|---:|
| GSE29378 | exploratory | GPL6947 | 63 | 32 | 31 | 19609 |
| GSE109887 | external | preprocessed | 78 | 46 | 32 | 31682 |
| GSE118553 | external | GPL10558 | 267 | 100 | 167 | 20759 |
| GSE122063 | external | preprocessed | 100 | 44 | 56 | 32074 |
| GSE48350 | external | GPL570 | 220 | 140 | 80 | 21753 |
| GSE1297 | train | GPL96 | 31 | 9 | 22 | 13100 |
| GSE33000 | train | preprocessed | 467 | 157 | 310 | 17402 |
| GSE36980 | train | GPL6244 | 80 | 47 | 33 | 20003 |
| GSE5281 | train | GPL570 | 161 | 74 | 87 | 21753 |
