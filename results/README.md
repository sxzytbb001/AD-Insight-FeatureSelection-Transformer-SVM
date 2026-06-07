# Results Directory

Pipeline outputs are written here.

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

New generated outputs are ignored by default. A repository may keep selected
result snapshots for reproducibility, but large checkpoints and generated plots
should be managed through Git LFS or release assets.

Useful summary files:

- `results/external_validation/external_validation_summary.csv`
- `results/statistics/transformer_vs_svm_statistics.csv`
- `docs/latest_results.md`
