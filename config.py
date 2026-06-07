import json
import os

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path_value):
    if not path_value:
        return path_value
    path_value = os.path.expanduser(str(path_value))
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    return os.path.abspath(os.path.join(BASE_DIR, path_value))


def _path_from_env(env_name, default_value):
    return _resolve_path(os.environ.get(env_name, default_value))


def _parse_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _external_dataset_config(path_value, extra=None):
    payload = dict(extra or {})
    payload["path"] = _resolve_path(path_value)
    payload["label_flip"] = _parse_bool(payload.get("label_flip"), default=False)
    return payload


def _parse_external_datasets():
    """Load external validation cohorts from env or external_datasets.json.

    Supported env format:
        GENE_EXPR_EXTERNAL_DATASETS="cohort_a=path/to/a;cohort_b=D:/data/b"

    Supported JSON formats:
        {"cohort_a": "path/to/a"}
        {"cohort_a": {"path": "path/to/a", "label_flip": true}}
        [{"name": "cohort_a", "path": "path/to/a", "label_flip": true}]
    """
    env_value = os.environ.get("GENE_EXPR_EXTERNAL_DATASETS", "").strip()
    if env_value:
        datasets = {}
        for item in env_value.replace(",", ";").split(";"):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    "GENE_EXPR_EXTERNAL_DATASETS must use NAME=PATH entries."
                )
            name, path_value = item.split("=", 1)
            name = name.strip()
            path_value = path_value.strip()
            if name and path_value:
                datasets[name] = _external_dataset_config(path_value)
        return datasets

    config_path = _path_from_env(
        "GENE_EXPR_EXTERNALS_FILE",
        os.path.join(BASE_DIR, "external_datasets.json"),
    )
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if isinstance(payload, dict):
        datasets = {}
        for name, value in payload.items():
            if isinstance(value, dict):
                if "path" not in value:
                    raise ValueError(
                        "external_datasets.json object entries must contain path."
                    )
                extra = {key: item for key, item in value.items() if key != "path"}
                datasets[str(name)] = _external_dataset_config(value["path"], extra)
            else:
                datasets[str(name)] = _external_dataset_config(value)
        return datasets

    if isinstance(payload, list):
        datasets = {}
        for item in payload:
            if not isinstance(item, dict) or "name" not in item or "path" not in item:
                raise ValueError(
                    "external_datasets.json list entries must contain name and path."
                )
            extra = {key: value for key, value in item.items() if key not in {"name", "path"}}
            datasets[str(item["name"])] = _external_dataset_config(item["path"], extra)
        return datasets

    raise ValueError("external_datasets.json must be a JSON object or list.")


DATA_DIR = _path_from_env("GENE_EXPR_DATA_DIR", os.path.join(BASE_DIR, "data"))
TRAIN_DATA_DIR = _path_from_env(
    "GENE_EXPR_TRAIN_DIR",
    os.path.join(DATA_DIR, "train"),
)
RESULTS_DIR = _path_from_env("GENE_EXPR_RESULTS_DIR", os.path.join(BASE_DIR, "results"))

FEATURE_SELECTION_DIR = os.path.join(RESULTS_DIR, "feature_selection")
TRANSFORMER_DIR = os.path.join(RESULTS_DIR, "transformer")
SVM_DIR = os.path.join(RESULTS_DIR, "svm")
EXTERNAL_VALIDATION_DIR = os.path.join(RESULTS_DIR, "external_validation")
STATISTICS_DIR = os.path.join(RESULTS_DIR, "statistics")

CANDIDATE_GENES_PATH = os.path.join(FEATURE_SELECTION_DIR, "candidate_genes.txt")
FEATURE_STATS_PATH = os.path.join(FEATURE_SELECTION_DIR, "candidate_gene_stats.csv")

TRANSFORMER_MODEL_PATH = os.path.join(TRANSFORMER_DIR, "best_transformer_model.pth")
TRANSFORMER_PARAMS_PATH = os.path.join(TRANSFORMER_DIR, "best_params.json")
TRANSFORMER_SCALER_PATH = os.path.join(TRANSFORMER_DIR, "scaler.pkl")

SVM_MODEL_PATH = os.path.join(SVM_DIR, "best_svm_model.pkl")
SVM_VOTING_PATH = os.path.join(SVM_DIR, "voting_svm_model.pkl")
SVM_BAGGING_PATH = os.path.join(SVM_DIR, "bagging_svm_model.pkl")
SVM_ENSEMBLE_PATH = os.path.join(SVM_DIR, "ensemble_svm_model.pkl")
SVM_SCALER_PATH = os.path.join(SVM_DIR, "scaler.pkl")
SVM_PARAMS_PATH = os.path.join(SVM_DIR, "best_params.json")

# Backward-compatible aliases used by older scripts.
MODEL_PATH = TRANSFORMER_MODEL_PATH
BEST_PARAMS_PATH = TRANSFORMER_PARAMS_PATH

TRAIN_MATRIX_PATH = _path_from_env(
    "GENE_EXPR_TRAIN_MATRIX",
    os.path.join(TRAIN_DATA_DIR, "cleaned_gene_matrix.csv"),
)
TRAIN_LABELS_PATH = _path_from_env(
    "GENE_EXPR_TRAIN_LABELS",
    os.path.join(TRAIN_DATA_DIR, "sample_labels.csv"),
)

POSITIVE_LABEL = os.environ.get("GENE_EXPR_POSITIVE_LABEL", "positive")
NEGATIVE_LABEL = os.environ.get("GENE_EXPR_NEGATIVE_LABEL", "control")
EXTERNAL_DATASETS = _parse_external_datasets()
EXTERNAL_DATA_DIRS = {
    name: dataset_config["path"]
    for name, dataset_config in EXTERNAL_DATASETS.items()
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    dirs = [
        RESULTS_DIR,
        FEATURE_SELECTION_DIR,
        TRANSFORMER_DIR,
        SVM_DIR,
        EXTERNAL_VALIDATION_DIR,
        STATISTICS_DIR,
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def load_candidate_genes():
    if os.path.exists(CANDIDATE_GENES_PATH):
        with open(CANDIDATE_GENES_PATH, "r", encoding="utf-8") as file_obj:
            return [line.strip() for line in file_obj if line.strip()]
    return []


def load_best_params():
    if os.path.exists(TRANSFORMER_PARAMS_PATH):
        with open(TRANSFORMER_PARAMS_PATH, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    return {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.3,
        "lr": 0.001,
        "batch_size": 32,
        "weight_decay": 0.01,
        "max_epochs": 80,
    }
