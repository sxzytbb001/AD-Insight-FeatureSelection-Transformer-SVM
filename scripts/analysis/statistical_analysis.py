import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import config

config.ensure_dirs()


DISPLAY_NAME_MAP = {
    "Internal_OOF": "Internal OOF",
}


def _bootstrap_ci(metric_fn, y_true, values, n_bootstraps=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    values = np.asarray(values)
    scores = []
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n_samples, n_samples)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2 and metric_fn is roc_auc_score:
            continue
        scores.append(metric_fn(y_boot, values[idx]))

    if not scores:
        return float("nan"), float("nan")

    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def _compute_midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    midranks = np.zeros(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        midranks[start:end] = 0.5 * (start + end - 1) + 1
        start = end
    result = np.empty(len(x), dtype=float)
    result[order] = midranks
    return result


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def _delong_roc_test(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)

    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds = np.vstack([pred_a, pred_b])[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    auc_diff = float(aucs[0] - aucs[1])

    if cov.ndim == 0:
        return 1.0, auc_diff

    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 1e-12:
        return 1.0, auc_diff

    z = abs(auc_diff) / np.sqrt(var)
    p_value = float(2.0 * norm.sf(z))
    return p_value, auc_diff


def _mcnemar_test(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)

    b = int(np.sum((pred_a == y_true) & (pred_b != y_true)))
    c = int(np.sum((pred_a != y_true) & (pred_b == y_true)))
    discordant = b + c
    if discordant == 0:
        return 1.0, b, c

    if discordant < 25:
        p_value = float(binomtest(min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue)
    else:
        stat = (abs(b - c) - 1.0) ** 2 / discordant
        p_value = float(chi2.sf(stat, 1))
    return p_value, b, c


def _assemble_stats_row(dataset_name, y_true, transformer_prob, svm_prob, transformer_threshold, svm_threshold):
    transformer_prob = np.asarray(transformer_prob, dtype=float)
    svm_prob = np.asarray(svm_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    transformer_pred = (transformer_prob >= transformer_threshold).astype(int)
    svm_pred = (svm_prob >= svm_threshold).astype(int)

    transformer_auc = float(roc_auc_score(y_true, transformer_prob))
    svm_auc = float(roc_auc_score(y_true, svm_prob))
    transformer_acc = float(accuracy_score(y_true, transformer_pred))
    svm_acc = float(accuracy_score(y_true, svm_pred))

    transformer_auc_ci = _bootstrap_ci(roc_auc_score, y_true, transformer_prob)
    svm_auc_ci = _bootstrap_ci(roc_auc_score, y_true, svm_prob, random_state=43)
    transformer_acc_ci = _bootstrap_ci(accuracy_score, y_true, transformer_pred, random_state=44)
    svm_acc_ci = _bootstrap_ci(accuracy_score, y_true, svm_pred, random_state=45)

    delong_p, auc_diff = _delong_roc_test(y_true, transformer_prob, svm_prob)
    mcnemar_p, discordant_tf_win, discordant_svm_win = _mcnemar_test(y_true, transformer_pred, svm_pred)

    return {
        "Dataset": dataset_name,
        "N": int(len(y_true)),
        "Transformer_AUC": transformer_auc,
        "Transformer_AUC_CI_Low": transformer_auc_ci[0],
        "Transformer_AUC_CI_High": transformer_auc_ci[1],
        "SVM_AUC": svm_auc,
        "SVM_AUC_CI_Low": svm_auc_ci[0],
        "SVM_AUC_CI_High": svm_auc_ci[1],
        "AUC_Diff_Transformer_minus_SVM": auc_diff,
        "DeLong_P": delong_p,
        "Transformer_Accuracy": transformer_acc,
        "Transformer_Accuracy_CI_Low": transformer_acc_ci[0],
        "Transformer_Accuracy_CI_High": transformer_acc_ci[1],
        "SVM_Accuracy": svm_acc,
        "SVM_Accuracy_CI_Low": svm_acc_ci[0],
        "SVM_Accuracy_CI_High": svm_acc_ci[1],
        "Accuracy_Diff_Transformer_minus_SVM": transformer_acc - svm_acc,
        "McNemar_P": mcnemar_p,
        "Discordant_Transformer_Only_Correct": discordant_tf_win,
        "Discordant_SVM_Only_Correct": discordant_svm_win,
        "Transformer_Threshold": float(transformer_threshold),
        "SVM_Threshold": float(svm_threshold),
    }


def _format_metric_with_ci(value, low, high):
    return f"{value:.4f} ({low:.4f}, {high:.4f})"


def _format_p_value(value):
    if pd.isna(value):
        return "NA"
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def _dataset_display_name(name):
    return DISPLAY_NAME_MAP.get(name, name)


def _build_markdown_report(stats_df):
    lines = [
        "# Transformer 与 SVM 统计学分析",
        "",
        "## 结果汇总",
        "",
        "| 数据集 | 样本量 | Transformer AUC（95% CI） | SVM AUC（95% CI） | AUC差值（Transformer-SVM） | DeLong P值 | Transformer 准确率（95% CI） | SVM 准确率（95% CI） | 准确率差值（Transformer-SVM） | McNemar P值 |",
        "| --- | ---: | --- | --- | ---: | ---: | --- | --- | ---: | ---: |",
    ]

    for _, row in stats_df.iterrows():
        lines.append(
            "| {dataset} | {n} | {tf_auc} | {svm_auc} | {auc_diff:.4f} | {delong_p} | {tf_acc} | {svm_acc} | {acc_diff:.4f} | {mcnemar_p} |".format(
                dataset=_dataset_display_name(row["Dataset"]),
                n=int(row["N"]),
                tf_auc=_format_metric_with_ci(
                    row["Transformer_AUC"],
                    row["Transformer_AUC_CI_Low"],
                    row["Transformer_AUC_CI_High"],
                ),
                svm_auc=_format_metric_with_ci(
                    row["SVM_AUC"],
                    row["SVM_AUC_CI_Low"],
                    row["SVM_AUC_CI_High"],
                ),
                auc_diff=row["AUC_Diff_Transformer_minus_SVM"],
                delong_p=_format_p_value(row["DeLong_P"]),
                tf_acc=_format_metric_with_ci(
                    row["Transformer_Accuracy"],
                    row["Transformer_Accuracy_CI_Low"],
                    row["Transformer_Accuracy_CI_High"],
                ),
                svm_acc=_format_metric_with_ci(
                    row["SVM_Accuracy"],
                    row["SVM_Accuracy_CI_Low"],
                    row["SVM_Accuracy_CI_High"],
                ),
                acc_diff=row["Accuracy_Diff_Transformer_minus_SVM"],
                mcnemar_p=_format_p_value(row["McNemar_P"]),
            )
        )

    lines.extend(
        [
            "",
            "## 判别分歧与阈值",
            "",
            "| 数据集 | 仅 Transformer 判对样本数 | 仅 SVM 判对样本数 | Transformer 阈值 | SVM 阈值 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, row in stats_df.iterrows():
        lines.append(
            "| {dataset} | {tf_only} | {svm_only} | {tf_threshold:.4f} | {svm_threshold:.4f} |".format(
                dataset=_dataset_display_name(row["Dataset"]),
                tf_only=int(row["Discordant_Transformer_Only_Correct"]),
                svm_only=int(row["Discordant_SVM_Only_Correct"]),
                tf_threshold=row["Transformer_Threshold"],
                svm_threshold=row["SVM_Threshold"],
            )
        )

    lines.extend(
        [
            "",
            "## 简要结论",
            "",
        ]
    )

    for _, row in stats_df.iterrows():
        dataset_name = _dataset_display_name(row["Dataset"])
        lines.append(
            "- {dataset}: Transformer AUC={tf_auc:.4f}, SVM AUC={svm_auc:.4f}, "
            "DeLong P={delong_p}; Transformer Accuracy={tf_acc:.4f}, "
            "SVM Accuracy={svm_acc:.4f}, McNemar P={mcnemar_p}.".format(
                dataset=dataset_name,
                tf_auc=row["Transformer_AUC"],
                svm_auc=row["SVM_AUC"],
                delong_p=_format_p_value(row["DeLong_P"]),
                tf_acc=row["Transformer_Accuracy"],
                svm_acc=row["SVM_Accuracy"],
                mcnemar_p=_format_p_value(row["McNemar_P"]),
            )
        )

    lines.append("")

    return "\n".join(lines)


def _plot_internal_roc_comparison():
    transformer_path = os.path.join(config.TRANSFORMER_DIR, "oof_predictions.csv")
    svm_path = os.path.join(config.SVM_DIR, "oof_predictions.csv")
    if not os.path.exists(transformer_path) or not os.path.exists(svm_path):
        raise FileNotFoundError("缺少内部 OOF 预测文件，无法绘制内部 ROC 对比图。")

    transformer_df = pd.read_csv(transformer_path)
    svm_df = pd.read_csv(svm_path)
    merged = transformer_df.merge(svm_df[["sample_id", "SVM_prob"]], on="sample_id", how="inner")
    if merged.empty:
        raise ValueError("内部 OOF 预测结果为空，无法绘制内部 ROC 对比图。")

    y_true = merged["label"].to_numpy(dtype=int)
    transformer_prob = merged["Transformer_prob"].to_numpy(dtype=float)
    svm_prob = merged["SVM_prob"].to_numpy(dtype=float)

    transformer_auc = float(roc_auc_score(y_true, transformer_prob))
    svm_auc = float(roc_auc_score(y_true, svm_prob))
    transformer_fpr, transformer_tpr, _ = roc_curve(y_true, transformer_prob)
    svm_fpr, svm_tpr, _ = roc_curve(y_true, svm_prob)

    output_path = os.path.join(config.RESULTS_DIR, "internal_roc_comparison.png")

    plt.figure(figsize=(7.2, 6.0))
    plt.plot(
        transformer_fpr,
        transformer_tpr,
        color="#1f77b4",
        lw=2.2,
        label=f"Transformer (AUC={transformer_auc:.4f})",
    )
    plt.plot(
        svm_fpr,
        svm_tpr,
        color="#ff7f0e",
        lw=2.2,
        label=f"SVM (AUC={svm_auc:.4f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="#808080", lw=1.2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Internal 5-fold OOF ROC Comparison")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def _load_internal_comparison():
    transformer_path = os.path.join(config.TRANSFORMER_DIR, "oof_predictions.csv")
    svm_path = os.path.join(config.SVM_DIR, "oof_predictions.csv")
    if not os.path.exists(transformer_path) or not os.path.exists(svm_path):
        raise FileNotFoundError("缺少内部 OOF 预测文件，无法运行统计分析。")

    transformer_df = pd.read_csv(transformer_path)
    svm_df = pd.read_csv(svm_path)
    merged = transformer_df.merge(svm_df[["sample_id", "SVM_prob"]], on="sample_id", how="inner")

    with open(config.TRANSFORMER_PARAMS_PATH, "r", encoding="utf-8") as f:
        transformer_params = json.load(f)
    with open(config.SVM_PARAMS_PATH, "r", encoding="utf-8") as f:
        svm_params = json.load(f)

    return _assemble_stats_row(
        "Internal_OOF",
        merged["label"].values,
        merged["Transformer_prob"].values,
        merged["SVM_prob"].values,
        transformer_params["best_threshold"],
        svm_params["thresholds"]["SVM"],
    )


def _load_external_comparisons():
    summary_path = os.path.join(config.EXTERNAL_VALIDATION_DIR, "external_validation_summary.csv")
    if not os.path.exists(summary_path):
        return []

    summary_df = pd.read_csv(summary_path)
    rows = []

    for dataset_name in summary_df["Dataset"].unique():
        pred_path = os.path.join(config.EXTERNAL_VALIDATION_DIR, f"predictions_{dataset_name}.csv")
        if not os.path.exists(pred_path):
            continue
        pred_df = pd.read_csv(pred_path)

        transformer_row = summary_df[
            (summary_df["Dataset"] == dataset_name) & (summary_df["Model"] == "Transformer")
        ].iloc[0]
        svm_row = summary_df[
            (summary_df["Dataset"] == dataset_name) & (summary_df["Model"] == "SVM")
        ].iloc[0]

        rows.append(
            _assemble_stats_row(
                dataset_name,
                pred_df["y_true"].values,
                pred_df["Transformer_prob"].values,
                pred_df["SVM_prob"].values,
                transformer_row["Threshold"],
                svm_row["Threshold"],
            )
        )
    return rows


def run_statistical_analysis():
    os.makedirs(config.STATISTICS_DIR, exist_ok=True)
    internal_roc_path = _plot_internal_roc_comparison()

    rows = [_load_internal_comparison()]
    rows.extend(_load_external_comparisons())
    stats_df = pd.DataFrame(rows)

    csv_path = os.path.join(config.STATISTICS_DIR, "transformer_vs_svm_statistics.csv")
    stats_df.to_csv(csv_path, index=False)

    md_path = os.path.join(config.STATISTICS_DIR, "transformer_vs_svm_statistics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown_report(stats_df))

    print("=" * 60)
    print("统计分析完成")
    print(f"结果保存至: {config.STATISTICS_DIR}")
    print(f"内部 ROC 对比图: {internal_roc_path}")
    print(stats_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    run_statistical_analysis()
