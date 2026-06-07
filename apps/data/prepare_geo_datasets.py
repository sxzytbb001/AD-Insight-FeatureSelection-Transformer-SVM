import argparse
import csv
import gzip
import io
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from apps import common


ANNOTATION_URLS = {
    "GPL570": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz",
    "GPL96": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz",
    "GPL10558": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL10nnn/GPL10558/annot/GPL10558.annot.gz",
    "GPL6947": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6947/annot/GPL6947.annot.gz",
    "GPL6244": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6244/annot/GPL6244.annot.gz",
}


SERIES_ROLES = {
    "GSE5281": "train",
    "GSE36980": "train",
    "GSE1297": "train",
    "GSE118553": "external",
    "GSE48350": "external",
    "GSE29378": "exploratory",
}


LEGACY_DATASETS = {
    "GSE33000": {"source_dir": "GSE33000mx", "role": "train"},
    "GSE109887": {"source_dir": "GSE109887yz2", "role": "external"},
    "GSE122063": {"source_dir": "GSE122063yz1", "role": "external"},
}


@dataclass(frozen=True)
class ParsedSeriesMatrix:
    platform_id: str
    metadata: pd.DataFrame
    expression: pd.DataFrame


def _split_geo_line(line):
    return next(csv.reader([line.rstrip("\n")], delimiter="\t"))


def _clean_cell(value):
    return str(value).strip().strip('"')


def _read_series_metadata(path):
    meta = {}
    platform_id = ""
    table_begin = None
    table_end = None

    with Path(path).open("r", encoding="utf-8", errors="replace") as file_obj:
        for line_no, line in enumerate(file_obj):
            if line.startswith("!series_matrix_table_begin"):
                table_begin = line_no
                continue
            if line.startswith("!series_matrix_table_end"):
                table_end = line_no
                break
            if line.startswith("!Series_platform_id"):
                parts = _split_geo_line(line)
                if len(parts) > 1:
                    platform_id = _clean_cell(parts[1])
            if line.startswith("!Sample_"):
                parts = _split_geo_line(line)
                meta.setdefault(parts[0], []).append([_clean_cell(item) for item in parts[1:]])

    if table_begin is None or table_end is None:
        raise ValueError(f"{path} is missing series matrix table markers.")

    sample_ids = meta.get("!Sample_geo_accession", [[]])[0]
    titles = meta.get("!Sample_title", [[]])[0]
    characteristics = meta.get("!Sample_characteristics_ch1", [])
    rows = []
    for index, sample_id in enumerate(sample_ids):
        row = {
            "sample_id": sample_id,
            "title": titles[index] if index < len(titles) else "",
        }
        char_values = []
        for values in characteristics:
            value = values[index] if index < len(values) else ""
            char_values.append(value)
            if ":" not in value:
                continue
            key, parsed_value = value.split(":", 1)
            key = key.strip().lower()
            parsed_value = parsed_value.strip()
            if key in row:
                suffix = 2
                while f"{key}_{suffix}" in row:
                    suffix += 1
                key = f"{key}_{suffix}"
            row[key] = parsed_value
        row["characteristics_all"] = " | ".join(char_values)
        rows.append(row)

    return platform_id, pd.DataFrame(rows), table_begin, table_end


def parse_series_matrix(path):
    path = Path(path)
    platform_id, metadata, table_begin, table_end = _read_series_metadata(path)
    expression = pd.read_csv(
        path,
        sep="\t",
        skiprows=table_begin + 1,
        nrows=table_end - table_begin - 2,
        index_col=0,
        quotechar='"',
    )
    expression.index = expression.index.astype(str).str.strip()
    expression.columns = expression.columns.astype(str).str.strip()
    expression = expression.apply(pd.to_numeric, errors="coerce")
    return ParsedSeriesMatrix(platform_id=platform_id, metadata=metadata, expression=expression)


def _first_symbol(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan" or text in {"---", "--", "NA", "N/A"}:
        return None
    for separator in [" /// ", "///", " // ", "//", " / ", ";", ","]:
        if separator in text:
            text = text.split(separator, 1)[0].strip()
    if not text or text in {"---", "--"}:
        return None
    return text.upper()


def _annotation_symbol_column(columns):
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in [
        "gene symbol",
        "gene_symbol",
        "genesymbol",
        "symbol",
        "orf",
        "gene_assignment",
    ]:
        if candidate in normalized:
            return normalized[candidate]
    for col in columns:
        col_lower = str(col).lower()
        if "symbol" in col_lower or col_lower == "orf":
            return col
    raise ValueError(f"Cannot find gene symbol column in annotation columns: {list(columns)}")


def download_platform_annotation(platform_id, annotation_dir):
    annotation_dir = Path(annotation_dir)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    target = annotation_dir / f"{platform_id}.annot.gz"
    if target.exists():
        return target
    if platform_id not in ANNOTATION_URLS:
        raise ValueError(f"No annotation URL configured for {platform_id}")
    urllib.request.urlretrieve(ANNOTATION_URLS[platform_id], target)
    return target


def load_probe_to_symbol(platform_id, annotation_dir):
    annotation_path = download_platform_annotation(platform_id, annotation_dir)
    with gzip.open(annotation_path, "rt", encoding="utf-8", errors="replace") as file_obj:
        table_lines = []
        in_table = False
        for line in file_obj:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                table_lines.append(line)
    if not table_lines:
        raise ValueError(f"{annotation_path} is missing a platform annotation table.")
    annotation = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t", dtype=str, low_memory=False)
    id_column = annotation.columns[0]
    symbol_column = _annotation_symbol_column(annotation.columns)
    mapping = {}
    for probe_id, raw_symbol in zip(annotation[id_column], annotation[symbol_column]):
        symbol = _first_symbol(raw_symbol)
        if symbol:
            mapping[str(probe_id).strip()] = symbol
    if not mapping:
        raise ValueError(f"No probe-to-symbol mappings loaded for {platform_id}")
    return mapping


def aggregate_probe_matrix_by_symbol(expression, probe_to_symbol):
    expression = expression.copy()
    symbols = pd.Series(expression.index.astype(str), index=expression.index).map(probe_to_symbol)
    keep_mask = symbols.notna()
    filtered = expression.loc[keep_mask].copy()
    filtered.index = symbols.loc[keep_mask].astype(str).str.upper()
    filtered = filtered.apply(pd.to_numeric, errors="coerce")
    return filtered.groupby(level=0).mean().sort_index()


def _label_row(sample_id, label):
    return {"sample_id": sample_id, "label": label}


def build_sample_labels(dataset_name, metadata):
    rows = []
    for row in metadata.to_dict("records"):
        sample_id = str(row.get("sample_id", "")).strip()
        title = str(row.get("title", "")).strip()
        label = None

        if dataset_name == "GSE1297":
            group = str(row.get("group", "")).strip().lower()
            if group == "control":
                label = "control"
            elif group in {"incipient", "moderate", "severe"}:
                label = "positive"
        elif dataset_name == "GSE36980":
            title_lower = title.lower()
            if title_lower.startswith("non-ad"):
                label = "control"
            elif title_lower.startswith("ad"):
                label = "positive"
        elif dataset_name == "GSE48350":
            individual = str(row.get("individual", "")).strip()
            title_lower = title.lower()
            if individual.endswith(", C"):
                label = "control"
            elif "_ad_" in title_lower or title_lower.endswith("_ad"):
                label = "positive"
        elif dataset_name == "GSE29378":
            title_lower = title.lower()
            if title_lower.startswith("control"):
                label = "control"
            elif title_lower.startswith("ad"):
                label = "positive"
        else:
            value = str(row.get("disease state", "")).strip()
            value_lower = value.lower()
            if value_lower in {"normal", "control"}:
                label = "control"
            elif value_lower in {"ad", "alzheimer's disease", "alzheimer disease"}:
                label = "positive"

        if sample_id and label:
            rows.append(_label_row(sample_id, label))

    return pd.DataFrame(rows, columns=["sample_id", "label"])


def normalize_expression_by_dataset(gene_matrix):
    numeric = gene_matrix.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size and np.nanpercentile(finite, 95) > 50:
        values = np.log2(np.clip(values, a_min=0, a_max=None) + 1.0)
    means = np.nanmean(values, axis=1, keepdims=True)
    stds = np.nanstd(values, axis=1, keepdims=True)
    stds[stds < 1e-8] = 1.0
    normalized = (values - means) / stds
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(normalized, index=gene_matrix.index.astype(str).str.upper(), columns=gene_matrix.columns)


def _write_dataset(dataset_dir, gene_matrix, labels):
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    labels = labels.copy()
    labels["sample_id"] = labels["sample_id"].astype(str).str.strip()
    gene_matrix = gene_matrix.loc[:, labels["sample_id"].tolist()]
    gene_matrix.to_csv(dataset_dir / "geneMatrix.txt", sep="\t", index_label="geneNames")
    labels.to_csv(dataset_dir / "sample_labels.csv", index=False)


def process_series_dataset(dataset_name, series_path, annotation_dir, output_dir):
    parsed = parse_series_matrix(series_path)
    labels = build_sample_labels(dataset_name, parsed.metadata)
    probe_to_symbol = load_probe_to_symbol(parsed.platform_id, annotation_dir)
    gene_matrix = aggregate_probe_matrix_by_symbol(parsed.expression, probe_to_symbol)
    labels = labels[labels["sample_id"].isin(gene_matrix.columns)].drop_duplicates(subset=["sample_id"])
    gene_matrix = normalize_expression_by_dataset(gene_matrix.loc[:, labels["sample_id"].tolist()])
    _write_dataset(output_dir, gene_matrix, labels)
    return {
        "dataset": dataset_name,
        "role": SERIES_ROLES.get(dataset_name, "external"),
        "platform": parsed.platform_id,
        "samples": int(len(labels)),
        "control": int((labels["label"] == "control").sum()),
        "positive": int((labels["label"] == "positive").sum()),
        "genes": int(gene_matrix.shape[0]),
    }


def _read_matrix(path):
    return common.read_expression_matrix(str(path))


def _legacy_labels(source_dir):
    labels_path = source_dir / "sample_labels.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        if "sample_id" not in labels.columns:
            labels = labels.rename(columns={labels.columns[0]: "sample_id"})
        if "label" not in labels.columns:
            labels = labels.rename(columns={labels.columns[1]: "label"})
        labels = labels[["sample_id", "label"]].copy()
        labels["label"] = labels["label"].map(lambda value: "positive" if common.normalize_binary_label(value) == 1 else "control")
        return labels

    s1_path = source_dir / "s1.txt"
    s2_path = source_dir / "s2.txt"
    if not s1_path.exists() or not s2_path.exists():
        raise FileNotFoundError(f"{source_dir} is missing sample labels and s1/s2 files.")
    s1 = [line.strip() for line in s1_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    s2 = [line.strip() for line in s2_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return pd.DataFrame([_label_row(sample_id, "control") for sample_id in s1] + [_label_row(sample_id, "positive") for sample_id in s2])


def process_legacy_dataset(dataset_name, source_dir, output_dir):
    source_dir = Path(source_dir)
    matrix_path = source_dir / "geneMatrix.txt"
    if not matrix_path.exists():
        matrix_path = source_dir / "cleaned_gene_matrix.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"{source_dir} is missing geneMatrix.txt or cleaned_gene_matrix.csv")
    labels = _legacy_labels(source_dir)
    matrix = _read_matrix(matrix_path)
    matrix.index = matrix.index.astype(str).str.upper()
    matrix = matrix.groupby(level=0).mean()
    labels = labels[labels["sample_id"].isin(matrix.columns)].drop_duplicates(subset=["sample_id"])
    matrix = normalize_expression_by_dataset(matrix.loc[:, labels["sample_id"].tolist()])
    _write_dataset(output_dir, matrix, labels)
    return {
        "dataset": dataset_name,
        "role": LEGACY_DATASETS[dataset_name]["role"],
        "platform": "preprocessed",
        "samples": int(len(labels)),
        "control": int((labels["label"] == "control").sum()),
        "positive": int((labels["label"] == "positive").sum()),
        "genes": int(matrix.shape[0]),
    }


def _locate_legacy_root(project_root):
    required_dirs = {cfg["source_dir"] for cfg in LEGACY_DATASETS.values()}
    for candidate in project_root.parent.iterdir():
        if candidate == project_root or not candidate.is_dir():
            continue
        child_names = {child.name for child in candidate.iterdir() if child.is_dir()}
        if required_dirs.issubset(child_names):
            return candidate
    return None


def _prefix_dataset_samples(dataset_name, matrix, labels):
    rename_map = {sample_id: f"{dataset_name}__{sample_id}" for sample_id in labels["sample_id"].astype(str)}
    prefixed_matrix = matrix.rename(columns=rename_map)
    prefixed_labels = labels.copy()
    prefixed_labels["source_sample_id"] = prefixed_labels["sample_id"]
    prefixed_labels["dataset"] = dataset_name
    prefixed_labels["sample_id"] = prefixed_labels["sample_id"].map(rename_map)
    return prefixed_matrix, prefixed_labels


def build_training_dataset(processed_root, train_root, manifest):
    train_datasets = [item["dataset"] for item in manifest if item["role"] == "train"]
    matrices = {}
    labels_by_dataset = {}
    common_genes = None
    for dataset_name in train_datasets:
        dataset_dir = processed_root / dataset_name
        matrix = common.read_expression_matrix(str(dataset_dir / "geneMatrix.txt"))
        labels = pd.read_csv(dataset_dir / "sample_labels.csv")
        matrix.index = matrix.index.astype(str).str.upper()
        genes = set(matrix.index)
        common_genes = genes if common_genes is None else common_genes & genes
        matrices[dataset_name] = matrix
        labels_by_dataset[dataset_name] = labels

    if not common_genes:
        raise ValueError("No common genes found across training datasets.")

    ordered_genes = sorted(common_genes)
    combined_matrices = []
    combined_labels = []
    for dataset_name in train_datasets:
        matrix = matrices[dataset_name].loc[ordered_genes]
        labels = labels_by_dataset[dataset_name]
        matrix, labels = _prefix_dataset_samples(dataset_name, matrix, labels)
        combined_matrices.append(matrix)
        combined_labels.append(labels)

    train_root.mkdir(parents=True, exist_ok=True)
    combined_matrix = pd.concat(combined_matrices, axis=1)
    combined_label_df = pd.concat(combined_labels, ignore_index=True)
    combined_matrix.to_csv(train_root / "cleaned_gene_matrix.csv", index_label="geneNames")
    combined_label_df.to_csv(train_root / "sample_labels.csv", index=False)
    return {
        "train_datasets": train_datasets,
        "samples": int(combined_label_df.shape[0]),
        "control": int((combined_label_df["label"] == "control").sum()),
        "positive": int((combined_label_df["label"] == "positive").sum()),
        "genes": int(combined_matrix.shape[0]),
    }


def copy_external_datasets(processed_root, external_root, manifest):
    external_config = {}
    for item in manifest:
        role = item["role"]
        if role not in {"external", "exploratory"}:
            continue
        dataset_name = item["dataset"]
        source_dir = processed_root / dataset_name
        target_dir = external_root / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / "geneMatrix.txt", target_dir / "geneMatrix.txt")
        shutil.copy2(source_dir / "sample_labels.csv", target_dir / "sample_labels.csv")
        if role == "external":
            external_config[dataset_name] = str(target_dir.relative_to(external_root.parent.parent).as_posix())
    return external_config


def write_split_report(project_root, manifest, training_summary):
    report_path = project_root / "docs" / "data_split.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["| Dataset | Role | Platform | Samples | Control | Positive | Genes |", "|---|---|---:|---:|---:|---:|---:|"]
    for item in manifest:
        rows.append(
            f"| {item['dataset']} | {item['role']} | {item['platform']} | "
            f"{item['samples']} | {item['control']} | {item['positive']} | {item['genes']} |"
        )
    content = "\n".join(
        [
            "# Data Split",
            "",
            "This split keeps final validation datasets fully held out at the GEO cohort level.",
            "Held-out datasets are not used for feature selection, model fitting, or hyperparameter selection.",
            "",
            "## Roles",
            "",
            "- `train`: merged into `data/train/` after per-dataset normalization and common-gene intersection.",
            "- `external`: written to `data/external/<GSE>/` and referenced by `external_datasets.json`.",
            "- `exploratory`: prepared for manual checks but excluded from the default final validation config.",
            "",
            "## Training Summary",
            "",
            f"- Datasets: {', '.join(training_summary['train_datasets'])}",
            f"- Samples: {training_summary['samples']} ({training_summary['control']} control, {training_summary['positive']} positive)",
            f"- Common genes: {training_summary['genes']}",
            "",
            "## Dataset Manifest",
            "",
            *rows,
            "",
        ]
    )
    report_path.write_text(content, encoding="utf-8")
    return report_path


def prepare_datasets(project_root):
    project_root = Path(project_root)
    raw_root = project_root / "data" / "external" / "raw"
    processed_root = project_root / "data" / "external" / "processed"
    annotation_dir = raw_root / "platforms"
    train_root = project_root / "data" / "train"
    external_root = project_root / "data" / "external"
    legacy_root = _locate_legacy_root(project_root)

    manifest = []
    for dataset_name, role in SERIES_ROLES.items():
        series_path = raw_root / dataset_name / f"{dataset_name}_series_matrix.txt"
        if not series_path.exists():
            continue
        manifest.append(
            process_series_dataset(
                dataset_name,
                series_path,
                annotation_dir,
                processed_root / dataset_name,
            )
        )

    if legacy_root is not None:
        for dataset_name, cfg in LEGACY_DATASETS.items():
            source_dir = legacy_root / cfg["source_dir"]
            if source_dir.exists():
                manifest.append(process_legacy_dataset(dataset_name, source_dir, processed_root / dataset_name))

    manifest = sorted(manifest, key=lambda item: (item["role"], item["dataset"]))
    training_summary = build_training_dataset(processed_root, train_root, manifest)
    external_config = copy_external_datasets(processed_root, external_root, manifest)

    with (project_root / "external_datasets.json").open("w", encoding="utf-8") as file_obj:
        json.dump(external_config, file_obj, indent=2, ensure_ascii=False)

    manifest_path = project_root / "data" / "external" / "dataset_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    report_path = write_split_report(project_root, manifest, training_summary)
    return {
        "manifest": manifest,
        "training_summary": training_summary,
        "manifest_path": manifest_path,
        "report_path": report_path,
        "external_config": external_config,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare GEO AD datasets for training and held-out validation.")
    parser.add_argument("--project-root", default=".", help="Repository root.")
    args = parser.parse_args(argv)
    result = prepare_datasets(Path(args.project_root).resolve())
    print(f"Prepared {len(result['manifest'])} datasets.")
    print(f"Training samples: {result['training_summary']['samples']}")
    print(f"Training genes: {result['training_summary']['genes']}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Split report: {result['report_path']}")


if __name__ == "__main__":
    main()
