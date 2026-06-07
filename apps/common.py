import os
import json
import copy
import pickle
import re
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
from torch.utils.data import WeightedRandomSampler
from sklearn.preprocessing import QuantileTransformer, StandardScaler, quantile_transform
from sklearn.manifold import TSNE

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from apps import config

config.ensure_dirs()
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def tensor_to_numpy(tensor):
    """在 Torch 与 NumPy 二进制不兼容时，避免直接调用 tensor.numpy()。"""
    if isinstance(tensor, torch.Tensor):
        return np.asarray(tensor.detach().cpu().tolist())
    return np.asarray(tensor)


POSITIVE_LABEL_VALUES = {
    "1",
    "true",
    "yes",
    "y",
    "positive",
    "pos",
    "case",
    "disease",
    "diseased",
    "ad",
    "alzheimer",
    "alzheimer disease",
    "alzheimer's disease",
}
NEGATIVE_LABEL_VALUES = {
    "0",
    "false",
    "no",
    "n",
    "negative",
    "neg",
    "control",
    "normal",
    "healthy",
    "cn",
    "non-demented",
    "nondemented",
}


def _clean_label_text(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_binary_label(raw_value, positive_label=None, negative_label=None):
    if pd.isna(raw_value):
        return None

    text = _clean_label_text(raw_value)
    if not text or text == "nan":
        return None

    configured_negative = _clean_label_text(negative_label or config.NEGATIVE_LABEL)
    configured_positive = _clean_label_text(positive_label or config.POSITIVE_LABEL)
    if configured_negative and text == configured_negative:
        return 0
    if configured_positive and text == configured_positive:
        return 1

    try:
        number = float(text)
        if number == 0.0:
            return 0
        if number == 1.0:
            return 1
    except ValueError:
        pass

    if text in NEGATIVE_LABEL_VALUES:
        return 0
    if text in POSITIVE_LABEL_VALUES:
        return 1

    if any(token in text for token in ["control", "normal", "healthy", "non-demented", "nondemented"]):
        return 0
    if "alzheimer" in text or re.search(r"\bad\b", text):
        return 1
    if configured_negative and configured_negative in text:
        return 0
    if configured_positive and configured_positive in text:
        return 1

    return None


def _read_table(path, index_col=0):
    lower_path = path.lower()
    if lower_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, index_col=index_col)
    if lower_path.endswith((".tsv", ".txt")):
        try:
            frame = pd.read_csv(path, sep="\t", index_col=index_col)
            if frame.shape[1] > 1:
                return frame
        except Exception:
            pass
        return pd.read_csv(path, sep=r"\s+", index_col=index_col)
    return pd.read_csv(path, index_col=index_col)


def read_expression_matrix(path):
    matrix = _read_table(path, index_col=0)
    matrix.index = matrix.index.astype(str).str.strip()
    matrix.columns = matrix.columns.astype(str).str.strip()
    return matrix


def _read_label_table(path):
    lower_path = path.lower()
    if lower_path.endswith((".xlsx", ".xls")):
        labels_df = pd.read_excel(path)
    elif lower_path.endswith((".tsv", ".txt")):
        labels_df = pd.read_csv(path, sep=None, engine="python")
    else:
        labels_df = pd.read_csv(path)

    if "sample_id" not in labels_df.columns:
        labels_df = labels_df.rename(columns={labels_df.columns[0]: "sample_id"})
    if "label" not in labels_df.columns:
        if len(labels_df.columns) < 2:
            raise ValueError("Label table must contain sample_id and label columns.")
        second_col = [col for col in labels_df.columns if col != "sample_id"][0]
        labels_df = labels_df.rename(columns={second_col: "label"})

    ordered_columns = ["sample_id", "label"] + [
        col for col in labels_df.columns if col not in {"sample_id", "label"}
    ]
    labels_df = labels_df[ordered_columns].dropna(subset=["sample_id"]).copy()
    labels_df["sample_id"] = labels_df["sample_id"].astype(str).str.strip()
    return labels_df


def orient_expression_matrix(matrix_df, sample_ids):
    sample_ids = [str(sample_id).strip() for sample_id in sample_ids]
    matrix_df = matrix_df.copy()
    matrix_df.index = matrix_df.index.astype(str).str.strip()
    matrix_df.columns = matrix_df.columns.astype(str).str.strip()

    column_matches = [sample_id for sample_id in sample_ids if sample_id in matrix_df.columns]
    index_matches = [sample_id for sample_id in sample_ids if sample_id in matrix_df.index]

    if not column_matches and not index_matches:
        raise ValueError(
            "Expression matrix does not contain any sample IDs from the label table "
            "in either rows or columns."
        )

    if len(column_matches) >= len(index_matches):
        oriented = matrix_df.loc[:, column_matches].T
    else:
        oriented = matrix_df.loc[index_matches, :]

    oriented.index = oriented.index.astype(str).str.strip()
    oriented = oriented.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return oriented


def load_training_matrix_and_labels():
    if not os.path.exists(config.TRAIN_MATRIX_PATH):
        raise FileNotFoundError(f'找不到训练矩阵文件: {config.TRAIN_MATRIX_PATH}')
    if not os.path.exists(config.TRAIN_LABELS_PATH):
        raise FileNotFoundError(f'找不到训练标签文件: {config.TRAIN_LABELS_PATH}')

    raw_matrix = read_expression_matrix(config.TRAIN_MATRIX_PATH)
    labels_df = _read_label_table(config.TRAIN_LABELS_PATH)
    labels_by_id = labels_df.drop_duplicates(subset=["sample_id"], keep="first").set_index("sample_id")
    gene_matrix = orient_expression_matrix(raw_matrix, labels_by_id.index.tolist())

    labels_aligned = labels_by_id.loc[gene_matrix.index].rename_axis("sample_id").reset_index()
    parsed_labels = labels_aligned["label"].map(normalize_binary_label)
    if parsed_labels.isna().any():
        bad_values = sorted(labels_aligned.loc[parsed_labels.isna(), "label"].astype(str).unique().tolist())
        raise ValueError(
            "Unable to parse binary labels. Set GENE_EXPR_POSITIVE_LABEL/"
            f"GENE_EXPR_NEGATIVE_LABEL or normalize labels first. Unparsed values: {bad_values}"
        )

    labels_aligned["binary_label"] = parsed_labels.astype(int)
    y = labels_aligned["binary_label"].to_numpy(dtype=int)
    return gene_matrix, labels_aligned, y


def load_training_data(candidate_genes):
    gene_matrix, labels_df, y = load_training_matrix_and_labels()
    available_genes = [g for g in candidate_genes if g in gene_matrix.columns]
    X = gene_matrix[available_genes].values
    return X, y, available_genes, gene_matrix, labels_df


def _normalize_external_label(raw_value):
    return normalize_binary_label(raw_value)


def _load_external_labels_from_clinical(data_dir, sample_ids):
    clinical_path = os.path.join(data_dir, 'clinical.xlsx')
    if not os.path.exists(clinical_path):
        return None

    try:
        clinical_df = pd.read_excel(clinical_path)
    except Exception:
        return None

    accession_cols = [col for col in clinical_df.columns if 'geo_accession' in str(col).lower()]
    if not accession_cols:
        return None
    accession_col = accession_cols[0]

    preferred_cols = []
    for col in clinical_df.columns:
        col_lower = str(col).lower()
        if col == 'Type' or 'diagnosis' in col_lower or 'disease state' in col_lower:
            preferred_cols.append(col)
    for col in clinical_df.columns:
        if col not in preferred_cols and 'characteristics' in str(col).lower():
            preferred_cols.append(col)

    if not preferred_cols:
        return None

    clinical_df = clinical_df.copy()
    clinical_df[accession_col] = clinical_df[accession_col].astype(str).str.strip()

    for label_col in preferred_cols:
        subset = clinical_df[[accession_col, label_col]].dropna().copy()
        if subset.empty:
            continue

        subset['parsed_label'] = subset[label_col].map(_normalize_external_label)
        subset = subset.dropna(subset=['parsed_label'])
        if subset.empty:
            continue

        label_map = (
            subset.drop_duplicates(subset=[accession_col], keep='first')
            .set_index(accession_col)['parsed_label']
            .astype(int)
            .to_dict()
        )
        if all(sample_id in label_map for sample_id in sample_ids):
            return np.asarray([label_map[sample_id] for sample_id in sample_ids], dtype=int)

    return None


def _find_first_existing(data_dir, filenames):
    for filename in filenames:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path
    return None


def _align_features_to_candidates(X, candidate_genes):
    X_final = pd.DataFrame(index=X.index)
    missing_genes = []
    for gene in candidate_genes:
        if gene in X.columns:
            X_final[gene] = X[gene]
        else:
            X_final[gene] = 0.0
            missing_genes.append(gene)
    return X_final, missing_genes


def load_external_data(data_dir, dataset_name, candidate_genes):
    matrix_path = _find_first_existing(
        data_dir,
        [
            "matrix.csv",
            "expression_matrix.csv",
            "gene_matrix.csv",
            "cleaned_gene_matrix.csv",
            "matrix.tsv",
            "geneMatrix.txt",
        ],
    )
    if matrix_path is None:
        raise FileNotFoundError(
            f"{dataset_name} is missing an expression matrix. Expected one of: "
            "matrix.csv, expression_matrix.csv, gene_matrix.csv, cleaned_gene_matrix.csv, "
            "matrix.tsv, geneMatrix.txt"
        )

    raw_matrix = read_expression_matrix(matrix_path)
    label_path = _find_first_existing(
        data_dir,
        ["labels.csv", "sample_labels.csv", "labels.tsv", "sample_labels.tsv", "clinical.xlsx"],
    )

    if label_path is not None and not label_path.endswith("clinical.xlsx"):
        labels_df = _read_label_table(label_path)
        labels_by_id = labels_df.drop_duplicates(subset=["sample_id"], keep="first").set_index("sample_id")
        X = orient_expression_matrix(raw_matrix, labels_by_id.index.tolist())
        labels_aligned = labels_by_id.loc[X.index]
        parsed_labels = labels_aligned["label"].map(normalize_binary_label)
        if parsed_labels.isna().any():
            bad_values = sorted(labels_aligned.loc[parsed_labels.isna(), "label"].astype(str).unique().tolist())
            raise ValueError(f"{dataset_name} has unparsed label values: {bad_values}")
        y = parsed_labels.to_numpy(dtype=int)
        label_source = os.path.basename(label_path)
    else:
        s1_path = os.path.join(data_dir, "s1.txt")
        s2_path = os.path.join(data_dir, "s2.txt")
        if not os.path.exists(s1_path) or not os.path.exists(s2_path):
            raise FileNotFoundError(
                f"{dataset_name} is missing labels. Provide labels.csv/sample_labels.csv "
                "or legacy s1.txt/s2.txt group files."
            )

        with open(s1_path, "r", encoding="utf-8") as file_obj:
            s1 = [x.strip() for x in file_obj if x.strip()]
        with open(s2_path, "r", encoding="utf-8") as file_obj:
            s2 = [x.strip() for x in file_obj if x.strip()]

        raw_matrix.columns = raw_matrix.columns.astype(str).str.strip()
        valid_s1 = [sample_id for sample_id in s1 if sample_id in raw_matrix.columns]
        valid_s2 = [sample_id for sample_id in s2 if sample_id in raw_matrix.columns]
        if not valid_s1 and not valid_s2:
            raise ValueError(f"{dataset_name} s1/s2 sample IDs do not match matrix columns.")

        X = pd.concat([raw_matrix[valid_s1].T, raw_matrix[valid_s2].T], axis=0)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        clinical_labels = _load_external_labels_from_clinical(data_dir, X.index.tolist())
        if clinical_labels is not None:
            y = clinical_labels
            label_source = "clinical.xlsx"
        else:
            y = np.array([0] * len(valid_s1) + [1] * len(valid_s2))
            label_source = "s1_s2_default"

    X_final, missing_genes = _align_features_to_candidates(X, candidate_genes)

    control_count = int(np.sum(y == 0))
    positive_count = int(np.sum(y == 1))
    present_genes = len(candidate_genes) - len(missing_genes)
    coverage_ratio = present_genes / max(len(candidate_genes), 1)
    return X_final.values, y, {
        "control": control_count,
        "ad": positive_count,
        "positive": positive_count,
        "present_genes": present_genes,
        "missing_genes": len(missing_genes),
        "coverage_ratio": coverage_ratio,
        "missing_gene_names": missing_genes,
        "label_source": label_source,
    }


def align_dataset_labels(X_train, y_train, X_val, y_val, allow_flip=True):
    train_sig = np.mean(X_train[y_train == 1], axis=0) - np.mean(X_train[y_train == 0], axis=0)
    val_sig = np.mean(X_val[y_val == 1], axis=0) - np.mean(X_val[y_val == 0], axis=0)

    valid_idx = np.where((np.abs(train_sig) > 1e-9) | (np.abs(val_sig) > 1e-9))[0]
    if len(valid_idx) < 3:
        return y_val, np.nan, False

    corr = np.corrcoef(train_sig[valid_idx], val_sig[valid_idx])[0, 1]
    if np.isnan(corr):
        return y_val, corr, False
    if corr < 0 and allow_flip:
        return 1 - y_val, corr, True
    return y_val, corr, False


def _clip_array(X, clip_range: Optional[Union[float, Sequence[float]]]):
    if clip_range is None:
        return X
    if isinstance(clip_range, tuple) and len(clip_range) == 2:
        return np.clip(X, clip_range[0], clip_range[1])
    if isinstance(clip_range, list) and len(clip_range) == 2:
        return np.clip(X, clip_range[0], clip_range[1])
    clip_value_array = np.asarray(clip_range, dtype=float).reshape(-1)
    if clip_value_array.size != 1:
        raise ValueError('clip_range must be None, a scalar, or a 2-element sequence.')
    clip_value = float(clip_value_array[0])
    return np.clip(X, -clip_value, clip_value)


def align_data_distribution(
    X_train,
    X_ext,
    clip_range: Optional[Union[float, Sequence[float]]] = 2.0,
):
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0) + 1e-8
    ext_mean = X_ext.mean(axis=0)
    ext_std = X_ext.std(axis=0) + 1e-8

    X_ext_aligned = (X_ext - ext_mean) / ext_std
    X_ext_aligned = X_ext_aligned * train_std + train_mean
    return _clip_array(X_ext_aligned, clip_range)


def align_data_distribution_soft(
    X_train,
    X_ext,
    alpha=0.5,
    clip_range: Optional[Union[float, Sequence[float]]] = 3.0,
):
    """温和的分布对齐：保留部分原始数据结构，适用于 Transformer 等非线性模型。

    alpha=1.0 等同于全量对齐（传统方法），alpha=0.0 不做任何对齐。
    默认 alpha=0.5 在对齐和保留原始非线性结构之间取折中。
    """
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0) + 1e-8
    ext_mean = X_ext.mean(axis=0)
    ext_std = X_ext.std(axis=0) + 1e-8

    # 全量对齐版本
    X_full_aligned = (X_ext - ext_mean) / ext_std * train_std + train_mean
    # 仅做中心对齐（保留相对结构）
    X_center_aligned = X_ext - ext_mean + train_mean

    # 混合：alpha 控制对齐程度
    X_blended = alpha * X_full_aligned + (1 - alpha) * X_center_aligned
    return _clip_array(X_blended, clip_range)


def quantile_normalize_external(X):
    X_qt = quantile_transform(
        X,
        n_quantiles=min(100, X.shape[0]),
        output_distribution='normal',
        copy=True
    )
    return np.clip(X_qt, -2, 2)


def fit_rank_gauss_preprocessor(X, max_quantiles=128):
    n_quantiles = max(16, min(int(max_quantiles), int(X.shape[0])))
    quantile = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution='normal',
        copy=True,
        subsample=int(1e9),
        random_state=42,
    )
    scaler = StandardScaler()
    X_rank = quantile.fit_transform(X)
    X_scaled = scaler.fit_transform(X_rank)
    return {
        'type': 'rank_gauss_standard',
        'quantile': quantile,
        'scaler': scaler,
    }, np.clip(X_scaled, -4, 4)


def transform_with_preprocessor(preprocessor, X):
    if isinstance(preprocessor, dict) and preprocessor.get('type') == 'rank_gauss_standard':
        quantile = preprocessor.get('quantile')
        scaler = preprocessor.get('scaler')
        if quantile is None or scaler is None:
            raise TypeError('Rank-gauss preprocessor is missing quantile/scaler objects.')
        X_rank = quantile.transform(X)
        X_scaled = scaler.transform(X_rank)
        return np.clip(X_scaled, -4, 4)
    if isinstance(preprocessor, dict):
        raise TypeError('Unsupported dict preprocessor type for transform.')
    transform = getattr(preprocessor, 'transform', None)
    if callable(transform):
        return transform(X)
    raise TypeError('Unsupported preprocessor type for transform.')


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty(n, dtype=float)
    cumulative = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        cumulative = min(cumulative, value)
        adjusted[i] = cumulative
    result = np.empty(n, dtype=float)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def cohens_d(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n1 = len(x1)
    n2 = len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1 = np.var(x1, ddof=1)
    s2 = np.var(x2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(n1 + n2 - 2, 1) + 1e-8)
    return float((np.mean(x1) - np.mean(x2)) / pooled)


def create_weighted_sampler(y):
    class_counts = np.bincount(y.astype(int))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y.astype(int)]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True
    )

def compute_cohort_class_sample_weights(y, cohorts=None):
    """Balance sampling mass across cohort/class groups."""
    y = np.asarray(y, dtype=int)
    if cohorts is None:
        cohorts = np.asarray(["all"] * len(y), dtype=object)
    else:
        cohorts = np.asarray(cohorts, dtype=object)

    if len(cohorts) != len(y):
        raise ValueError("cohorts and y must contain the same number of samples.")

    keys = list(zip(cohorts.tolist(), y.tolist()))
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1

    return np.asarray([1.0 / counts[key] for key in keys], dtype=float)


def create_cohort_class_sampler(y, cohorts=None):
    sample_weights = compute_cohort_class_sample_weights(y, cohorts)
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.best_state = None
        self.counter = 0

    def step(self, score, model):
        if self.best_score is None:
            improved = True
        elif self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=return_attention,
            average_attn_weights=False
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attn_weights if return_attention else None


class TransformerV3(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.input_dim = input_dim
        self.raw_proj = nn.Linear(input_dim, d_model)
        self.interaction_factors = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.value_proj = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.gene_embedding = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)
        self.gene_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )

    def forward(self, x, return_attention=False, return_features=False, return_gene_weights=False):
        raw_x = x
        x = x.unsqueeze(-1)
        x = self.value_proj(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.input_norm(x + self.gene_embedding[:, :x.size(1), :])
        x = self.input_dropout(x)

        attention_maps = []
        for layer in self.encoder_layers:
            x, attn = layer(x, return_attention=return_attention)
            if return_attention and attn is not None:
                attention_maps.append(attn)

        cls_feature = x[:, 0]
        gene_tokens = x[:, 1:]
        gate_logits = self.gene_gate(gene_tokens).squeeze(-1)
        gate_weights = torch.softmax(gate_logits, dim=1)
        pooled_feature = torch.sum(gene_tokens * gate_weights.unsqueeze(-1), dim=1)
        linear_feature = self.raw_proj(raw_x)
        factor_input = raw_x.unsqueeze(-1) * self.interaction_factors.unsqueeze(0)
        interaction_feature = 0.5 * ((factor_input.sum(dim=1) ** 2) - (factor_input ** 2).sum(dim=1))
        features = (
            0.50 * cls_feature
            + 0.20 * pooled_feature
            + 0.15 * linear_feature
            + 0.15 * interaction_feature
        )
        logits = self.classifier(features)

        outputs = [logits]
        if return_features:
            outputs.append(features)
        if return_attention:
            outputs.append(attention_maps)
        if return_gene_weights:
            outputs.append(gate_weights)
        if len(outputs) == 1:
            return logits
        return tuple(outputs)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def plot_probability_distribution(y_true, y_prob, threshold, save_path, title):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_prob[y_true == 0], fill=True, alpha=0.3, color='#4c72b0', label='Control', warn_singular=False)
    sns.kdeplot(y_prob[y_true == 1], fill=True, alpha=0.3, color='#dd8452', label='Positive', warn_singular=False)
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold={threshold:.2f}')
    plt.title(title)
    plt.xlabel('Positive-class Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tsne(features, y_true, save_path, title):
    try:
        perplexity = min(20, max(5, len(y_true) - 1))
        embedded = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            init='random',
            learning_rate='auto'
        ).fit_transform(features)

        plt.figure(figsize=(7, 7))
        scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=y_true, cmap='coolwarm', alpha=0.8, edgecolors='k')
        handles, _ = scatter.legend_elements()
        plt.legend(handles, ['Control', 'Positive'], title='Group')
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception:
        plt.close()


def plot_attention_heatmap(attention_maps, gene_names, save_path):
    if not attention_maps:
        return
    stacked = torch.stack(attention_maps, dim=0)
    avg_map = tensor_to_numpy(stacked.mean(dim=(0, 1, 2)))
    if avg_map.shape[0] == len(gene_names) + 1:
        avg_map = avg_map[1:, 1:]
    plt.figure(figsize=(12, 10))
    sns.heatmap(avg_map, xticklabels=gene_names, yticklabels=gene_names, cmap='RdBu_r', center=avg_map.mean())
    plt.title('Transformer Attention Heatmap')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def compute_gene_interaction_matrix(model, X, gene_names, device, save_csv_path=None, save_png_path=None):
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    n_genes = len(gene_names)
    interaction = np.zeros((n_genes, n_genes), dtype=float)

    with torch.no_grad():
        base_prob = tensor_to_numpy(torch.softmax(model(torch.FloatTensor(X).to(device)), dim=1)[:, 1])

        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                X_i = X.copy()
                X_j = X.copy()
                X_ij = X.copy()
                X_i[:, i] = 0
                X_j[:, j] = 0
                X_ij[:, i] = 0
                X_ij[:, j] = 0

                prob_i = tensor_to_numpy(torch.softmax(model(torch.FloatTensor(X_i).to(device)), dim=1)[:, 1])
                prob_j = tensor_to_numpy(torch.softmax(model(torch.FloatTensor(X_j).to(device)), dim=1)[:, 1])
                prob_ij = tensor_to_numpy(torch.softmax(model(torch.FloatTensor(X_ij).to(device)), dim=1)[:, 1])

                individual = np.mean(base_prob - prob_i) + np.mean(base_prob - prob_j)
                combined = np.mean(base_prob - prob_ij)
                interaction[i, j] = combined - individual
                interaction[j, i] = interaction[i, j]

    interaction_df = pd.DataFrame(interaction, index=gene_names, columns=gene_names)
    if save_csv_path:
        interaction_df.to_csv(save_csv_path)
    if save_png_path:
        plt.figure(figsize=(12, 10))
        sns.heatmap(interaction_df, cmap='coolwarm', center=0)
        plt.title('Transformer Gene Interaction Matrix')
        plt.tight_layout()
        plt.savefig(save_png_path, dpi=300)
        plt.close()
    return interaction_df
