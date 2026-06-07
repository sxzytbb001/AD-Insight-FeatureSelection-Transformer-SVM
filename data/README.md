# Data Directory

This directory stores the case-study matrices and labels used by the default
pipeline configuration.

```text
data/
|-- train/
|   |-- cleaned_gene_matrix.csv
|   `-- sample_labels.csv
`-- external/
    |-- dataset_manifest.csv
    |-- GSE109887/
    |-- GSE118553/
    |-- GSE122063/
    `-- GSE48350/
```

The default paths are defined in `apps/config.py` and can be overridden with
environment variables such as `GENE_EXPR_TRAIN_MATRIX`,
`GENE_EXPR_TRAIN_LABELS`, and `GENE_EXPR_EXTERNALS_FILE`.

Before publishing or redistributing these files, confirm that the source dataset
licenses and privacy rules allow it. Large matrices should be tracked with Git
LFS or distributed as release assets.
