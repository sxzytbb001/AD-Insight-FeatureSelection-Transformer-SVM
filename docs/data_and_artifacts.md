# Data And Artifact Policy

This repository is intended to be usable as both research code and a
reproducibility snapshot.

## Data

- `data/train/` contains the default training matrix and labels used by the
  current case-study pipeline.
- `data/external/` contains configured external validation cohorts and
  `dataset_manifest.csv`.
- Raw GEO downloads, local clinical files, thesis drafts, and temporary
  preprocessing folders should stay outside the repository.

Before publishing processed matrices, verify that the original dataset license
and privacy rules allow redistribution.

## Git LFS

Large matrices and model artifacts are configured for Git LFS in
`.gitattributes`. Use:

```bash
git lfs ls-files
```

to audit LFS-tracked files.

For a code-only release, remove large data and checkpoints from the public
history and publish data preparation instructions instead. For a reproducibility
release, keep large files in Git LFS or GitHub Releases.

## Results

Pipeline outputs are written to `results/`. New generated outputs are ignored by
default. If a result snapshot is intentionally kept in the repository, keep the
corresponding command documented in [reproducibility.md](reproducibility.md) and
summarize the headline metrics in [latest_results.md](latest_results.md).

## Model Claims

Do not present external validation results as uniformly superior unless every
configured external cohort supports that claim. The current AD snapshot should
be described as Transformer performance comparable to strong SVM baselines, with
additional interpretability outputs.
