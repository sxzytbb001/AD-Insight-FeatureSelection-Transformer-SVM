# Contributing

Thanks for helping improve this repository. This project is a research pipeline,
so changes should preserve reproducibility and keep model comparisons fair.

## Development Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Before Opening A Pull Request

Run the lightweight test suite:

```bash
python -m pytest -q
```

For changes that affect model outputs, also run the smallest relevant pipeline
stage and update the affected result snapshot or documentation. Do not tune
external validation datasets after looking at their final labels or metrics.

## Data And Artifacts

- Keep raw downloads, local clinical files, notebooks, and thesis drafts out of
  the repository.
- Keep large matrices, model checkpoints, and result snapshots in Git LFS or
  release assets if they must be published.
- Prefer documenting data preparation steps over committing new large files.

## Pull Request Checklist

- The change is scoped to one behavior or documentation goal.
- Tests pass with `python -m pytest -q`.
- New configuration options are documented in `README.md`.
- Result claims are backed by a reproducible command or existing result file.
- No private data, unpublished clinical metadata, or local paths are included.
