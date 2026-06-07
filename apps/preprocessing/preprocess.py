import argparse
import os

import pandas as pd

from apps import config
from apps.common import (
    _read_label_table,
    normalize_binary_label,
    orient_expression_matrix,
    read_expression_matrix,
)

config.ensure_dirs()


def preprocess_data(
    matrix_path=None,
    labels_path=None,
    output_matrix_path=None,
    output_labels_path=None,
    positive_label=None,
    negative_label=None,
):
    matrix_path = matrix_path or config.TRAIN_MATRIX_PATH
    labels_path = labels_path or config.TRAIN_LABELS_PATH
    output_matrix_path = output_matrix_path or config.TRAIN_MATRIX_PATH
    output_labels_path = output_labels_path or config.TRAIN_LABELS_PATH
    positive_label = positive_label or config.POSITIVE_LABEL
    negative_label = negative_label or config.NEGATIVE_LABEL

    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Expression matrix not found: {matrix_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Label table not found: {labels_path}")

    raw_matrix = read_expression_matrix(matrix_path)
    labels_df = _read_label_table(labels_path)
    labels_by_id = labels_df.drop_duplicates(subset=["sample_id"], keep="first").set_index("sample_id")

    sample_gene_matrix = orient_expression_matrix(raw_matrix, labels_by_id.index.tolist())
    labels_aligned = labels_by_id.loc[sample_gene_matrix.index].rename_axis("sample_id").reset_index()
    parsed_labels = labels_aligned["label"].map(
        lambda value: normalize_binary_label(value, positive_label, negative_label)
    )

    if parsed_labels.isna().any():
        bad_values = sorted(labels_aligned.loc[parsed_labels.isna(), "label"].astype(str).unique().tolist())
        raise ValueError(
            "Unable to parse binary labels. Use --positive-label and --negative-label "
            f"or normalize the label file first. Unparsed values: {bad_values}"
        )

    labels_out = pd.DataFrame(
        {
            "sample_id": labels_aligned["sample_id"],
            "label": [
                positive_label if label == 1 else negative_label
                for label in parsed_labels.astype(int).tolist()
            ],
            "binary_label": parsed_labels.astype(int),
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_matrix_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_labels_path)), exist_ok=True)

    # Keep the historical output orientation: genes as rows, samples as columns.
    sample_gene_matrix.T.to_csv(output_matrix_path)
    labels_out.to_csv(output_labels_path, index=False)

    print("=" * 60)
    print("Preprocessing complete")
    print(f"Samples: {sample_gene_matrix.shape[0]}")
    print(f"Genes: {sample_gene_matrix.shape[1]}")
    print(f"Positive: {int(labels_out['binary_label'].sum())}")
    print(f"Negative: {int((labels_out['binary_label'] == 0).sum())}")
    print(f"Matrix: {output_matrix_path}")
    print(f"Labels: {output_labels_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Standardize a binary gene-expression dataset for the training pipeline."
    )
    parser.add_argument("--matrix", default=config.TRAIN_MATRIX_PATH, help="Input expression matrix.")
    parser.add_argument("--labels", default=config.TRAIN_LABELS_PATH, help="Input sample label table.")
    parser.add_argument(
        "--output-matrix",
        default=config.TRAIN_MATRIX_PATH,
        help="Output cleaned matrix path.",
    )
    parser.add_argument(
        "--output-labels",
        default=config.TRAIN_LABELS_PATH,
        help="Output normalized labels path.",
    )
    parser.add_argument(
        "--positive-label",
        default=config.POSITIVE_LABEL,
        help="Label value to treat as class 1.",
    )
    parser.add_argument(
        "--negative-label",
        default=config.NEGATIVE_LABEL,
        help="Label value to treat as class 0.",
    )
    args = parser.parse_args()

    preprocess_data(
        matrix_path=args.matrix,
        labels_path=args.labels,
        output_matrix_path=args.output_matrix,
        output_labels_path=args.output_labels,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )


if __name__ == "__main__":
    main()
