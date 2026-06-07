import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve

import config
from apps.visualization import plot_confusion_matrix, plot_grouped_metric_barplot
from common import (
    TransformerV3,
    align_data_distribution,
    align_dataset_labels,
    load_external_data,
    load_training_data,
    plot_probability_distribution,
    plot_tsne,
    quantile_normalize_external,
    tensor_to_numpy,
    transform_with_preprocessor,
)

config.ensure_dirs()

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_transformer_model():
    checkpoint = torch.load(config.TRANSFORMER_MODEL_PATH, map_location=config.DEVICE, weights_only=False)
    model_config = checkpoint["model_config"]

    ensemble_states = checkpoint.get("ensemble_states", None)
    ensemble_configs = checkpoint.get("ensemble_configs", None)
    if ensemble_states and len(ensemble_states) > 1:
        models = []
        for i, state_dict in enumerate(ensemble_states):
            cfg = ensemble_configs[i] if ensemble_configs and i < len(ensemble_configs) else model_config
            model = TransformerV3(
                input_dim=cfg["input_dim"],
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                num_classes=2,
            ).to(config.DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        print(f"  已加载 {len(models)} 个 Transformer 集成模型")
        return models, checkpoint

    model = TransformerV3(
        input_dim=model_config["input_dim"],
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        num_classes=2,
    ).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return [model], checkpoint


def load_adaptive_external_branches(checkpoint):
    adaptation = checkpoint.get("external_adaptation")
    if not adaptation:
        return None

    branch_models = {}
    for branch_key, branch_info in adaptation.get("branches", {}).items():
        models = []
        states = branch_info.get("states", [])
        configs = branch_info.get("configs", [])
        for i, state_dict in enumerate(states):
            cfg = configs[i]
            model = TransformerV3(
                input_dim=cfg["input_dim"],
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                num_classes=2,
            ).to(config.DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        branch_models[branch_key] = models

    if branch_models:
        print(f"  已加载外部泛化辅助分支: {list(branch_models.keys())}")
    return branch_models if branch_models else None


def load_svm_models():
    models = {}
    if os.path.exists(config.SVM_MODEL_PATH):
        with open(config.SVM_MODEL_PATH, "rb") as f:
            models["SVM"] = pickle.load(f)
    if os.path.exists(config.SVM_VOTING_PATH):
        with open(config.SVM_VOTING_PATH, "rb") as f:
            models["Voting_SVM"] = pickle.load(f)
    if os.path.exists(config.SVM_BAGGING_PATH):
        with open(config.SVM_BAGGING_PATH, "rb") as f:
            models["Bagging_SVM"] = pickle.load(f)

    ensemble_obj = None
    if os.path.exists(config.SVM_ENSEMBLE_PATH):
        with open(config.SVM_ENSEMBLE_PATH, "rb") as f:
            ensemble_obj = pickle.load(f)

    if not models and ensemble_obj is None:
        raise FileNotFoundError("未找到可用的 SVM 模型文件，请先运行 python -m scripts.training.train_svm")
    return models, ensemble_obj


def load_svm_thresholds():
    thresholds = {
        "SVM": 0.5,
        "Voting_SVM": 0.5,
        "Bagging_SVM": 0.5,
        "Ensemble_SVM": 0.5,
    }
    if not os.path.exists(config.SVM_PARAMS_PATH):
        return thresholds

    with open(config.SVM_PARAMS_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)

    nested = params.get("thresholds", {})
    if isinstance(nested, dict):
        for key in thresholds:
            if key in nested:
                thresholds[key] = float(nested[key])
    return thresholds


def predict_from_ensemble_obj(ensemble_obj, X):
    if ensemble_obj is None:
        return None
    if hasattr(ensemble_obj, "predict_proba"):
        return ensemble_obj.predict_proba(X)[:, 1]
    if isinstance(ensemble_obj, dict):
        pair_models = []
        for key in ("voting", "bagging"):
            model = ensemble_obj.get(key)
            if hasattr(model, "predict_proba"):
                pair_models.append(model)
        if pair_models:
            probs = [model.predict_proba(X)[:, 1] for model in pair_models]
            return np.mean(np.vstack(probs), axis=0)
    if isinstance(ensemble_obj, (list, tuple)):
        probs = [model.predict_proba(X)[:, 1] for model in ensemble_obj if hasattr(model, "predict_proba")]
        if probs:
            return np.mean(np.vstack(probs), axis=0)
    return None


def _compute_shift_score(X_train, X_ext_aligned):
    z = (X_ext_aligned - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
    return float(np.mean(np.abs(z)))


def _quantile_threshold_from_positive_rate(y_prob, positive_rate):
    positive_rate = float(np.clip(positive_rate, 1e-3, 1.0 - 1e-3))
    return float(np.quantile(y_prob, 1.0 - positive_rate))


def _predict_transformer_branch(models, X_input, n_tta=1, temperature=1.0, tta_noise_scales=None):
    if tta_noise_scales is None:
        tta_noise_scales = [0.0] * max(n_tta, 1)
    X_tensor = torch.FloatTensor(X_input).to(config.DEVICE)
    all_probs = []
    all_features = []
    for single_model in models:
        single_model.eval()
        with torch.no_grad():
            for tta_i in range(n_tta):
                noise_scale = tta_noise_scales[tta_i]
                x_input = X_tensor if noise_scale <= 0 else X_tensor + torch.randn_like(X_tensor) * noise_scale
                logits_tta, feat_tta = single_model(x_input, return_features=True)
                prob_tta = tensor_to_numpy(torch.softmax(logits_tta / temperature, dim=1)[:, 1])
                all_probs.append(prob_tta)
                all_features.append(tensor_to_numpy(feat_tta))
    return np.mean(all_probs, axis=0), np.mean(all_features, axis=0)


def plot_cm(cm, title, save_path):
    plot_confusion_matrix(
        cm,
        save_path,
        title,
        labels=("Control", "Positive"),
    )


def _apply_external_label_options(y_raw, group_info, dataset_options):
    y = np.asarray(y_raw, dtype=int).copy()
    updated_info = dict(group_info)
    dataset_options = dataset_options or {}

    configured_flip = bool(dataset_options.get("label_flip", False))
    if configured_flip:
        y = 1 - y

    updated_info["control"] = int(np.sum(y == 0))
    updated_info["positive"] = int(np.sum(y == 1))
    updated_info["ad"] = updated_info["positive"]
    updated_info["ConfiguredLabelFlip"] = configured_flip
    updated_info["LabelFlipReason"] = str(dataset_options.get("label_flip_reason", "") or "")
    updated_info["LabelPolarity"] = "flipped_by_config" if configured_flip else "as_provided"
    return y, updated_info


def evaluate_external_models():
    print("=" * 60)
    print("外部验证：统一评估 Transformer / SVM / Voting / Bagging")
    print("=" * 60)
    print(f"使用设备: {config.DEVICE}")

    if not config.EXTERNAL_DATA_DIRS:
        print(
            "未配置外部验证数据集，已跳过。可通过 external_datasets.json 或 "
            "GENE_EXPR_EXTERNAL_DATASETS=NAME=PATH 配置。"
        )
        return pd.DataFrame()

    candidate_genes = config.load_candidate_genes()
    if not candidate_genes:
        print("错误：未找到候选基因，请先运行 python -m scripts.preprocessing.feature_selection")
        return

    for path, name in [
        (config.TRANSFORMER_MODEL_PATH, "Transformer 模型"),
        (config.TRANSFORMER_SCALER_PATH, "Transformer 预处理器"),
        (config.SVM_SCALER_PATH, "SVM 预处理器"),
    ]:
        if not os.path.exists(path):
            print(f"错误：未找到 {name}")
            return

    X_train, y_train, available_genes, _, _ = load_training_data(candidate_genes)
    X_train = np.asarray(X_train, dtype=np.float32)
    train_positive_rate = float(np.mean(y_train))
    print(f"加载训练基因: {len(available_genes)} 个")

    transformer_model, transformer_ckpt = load_transformer_model()
    adaptive_branches = load_adaptive_external_branches(transformer_ckpt)
    try:
        svm_models, ensemble_obj = load_svm_models()
    except FileNotFoundError as exc:
        print(f"错误：{exc}")
        return

    with open(config.TRANSFORMER_SCALER_PATH, "rb") as f:
        transformer_scaler = pickle.load(f)
    with open(config.SVM_SCALER_PATH, "rb") as f:
        svm_scaler = pickle.load(f)

    default_transformer_threshold = float(transformer_ckpt.get("best_threshold", 0.5))
    default_transformer_roc_threshold = float(transformer_ckpt.get("roc_threshold", default_transformer_threshold))
    svm_thresholds = load_svm_thresholds()

    summary_rows = []
    fixed_internal_rows = []
    retrospective_rows = []
    external_dataset_configs = getattr(config, "EXTERNAL_DATASETS", {})
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    for dataset_name, dataset_dir in config.EXTERNAL_DATA_DIRS.items():
        print("\n" + "=" * 60)
        print(f"外部数据集: {dataset_name}")
        print("=" * 60)

        X_ext, y_ext_raw, group_info = load_external_data(dataset_dir, dataset_name, available_genes)
        dataset_options = external_dataset_configs.get(dataset_name, {})
        y_ext_raw, group_info = _apply_external_label_options(y_ext_raw, group_info, dataset_options)
        X_ext = np.asarray(X_ext, dtype=np.float32)
        print(
            f"样本数: {len(y_ext_raw)} "
            f"(Control={group_info['control']}, Positive={group_info['positive']})"
        )
        print(
            f"基因覆盖: {group_info['present_genes']}/{len(available_genes)} "
            f"({group_info['coverage_ratio']:.1%}), 缺失={group_info['missing_genes']}"
        )

        allow_label_flip = group_info.get("label_source") == "s1_s2_default"
        y_ext, corr, flipped = align_dataset_labels(
            X_train,
            y_train,
            X_ext,
            y_ext_raw,
            allow_flip=allow_label_flip,
        )
        print(f"标签来源: {group_info.get('label_source', 'unknown')}")
        if np.isnan(corr):
            print("标签相关性检查: 有效基因过少，跳过翻转判断")
        else:
            print(f"标签相关性: {corr:.4f} | 翻转标签: {'是' if flipped else '否'}")

        if not allow_label_flip and not np.isnan(corr):
            print("显式标签已启用, 已禁用自动翻转。")

        X_ext_aligned = align_data_distribution(X_train, X_ext, clip_range=None)
        print(f"分布对齐后范围: [{X_ext_aligned.min():.4f}, {X_ext_aligned.max():.4f}]")

        X_ext_svm = transform_with_preprocessor(svm_scaler, X_ext_aligned)
        if isinstance(transformer_scaler, dict):
            main_preprocessor = transformer_scaler.get("main", transformer_scaler)
            branch_preprocessors = transformer_scaler.get("external_branches", {})
        else:
            main_preprocessor = transformer_scaler
            branch_preprocessors = {}
        X_ext_trans = transform_with_preprocessor(main_preprocessor, X_ext_aligned)

        main_transformer_prob, transformer_features = _predict_transformer_branch(
            transformer_model,
            X_ext_trans,
            n_tta=1,
            temperature=1.0,
            tta_noise_scales=[0.0],
        )
        transformer_prob_raw = main_transformer_prob
        transformer_fixed_threshold = default_transformer_threshold
        transformer_strategy = "main_30_hard_aligned"
        print(f"Transformer 主模型推理完成 ({len(transformer_model)} 模型 x 1 TTA = {len(transformer_model)} 次推理)")

        if adaptive_branches and branch_preprocessors:
            adaptation = transformer_ckpt.get("external_adaptation", {})
            branch_meta = adaptation.get("branches", {})
            shift_score = _compute_shift_score(X_train, X_ext_aligned)
            shift_threshold = float(adaptation.get("shift_threshold", 0.95))

            if shift_score >= shift_threshold:
                selected_branch = str(int(adaptation.get("high_shift_gene_count", 20)))
                branch_source = quantile_normalize_external(X_ext[:, : int(selected_branch)])
                branch_input = transform_with_preprocessor(
                    branch_preprocessors[selected_branch],
                    branch_source,
                )
                transformer_prob_raw, _ = _predict_transformer_branch(
                    adaptive_branches[selected_branch],
                    branch_input,
                    n_tta=1,
                    temperature=1.0,
                    tta_noise_scales=[0.0],
                )
                transformer_fixed_threshold = float(
                    branch_meta.get(selected_branch, {}).get("cv_threshold", default_transformer_roc_threshold)
                )
                transformer_strategy = "adaptive_top20_quantile_only"
            else:
                selected_branch = str(int(adaptation.get("low_shift_gene_count", 28)))
                branch_input = transform_with_preprocessor(
                    branch_preprocessors[selected_branch],
                    X_ext_aligned[:, : int(selected_branch)],
                )
                branch_prob_raw, _ = _predict_transformer_branch(
                    adaptive_branches[selected_branch],
                    branch_input,
                    n_tta=1,
                    temperature=1.0,
                    tta_noise_scales=[0.0],
                )
                main_weight = float(adaptation.get("low_shift_main_weight", 0.10))
                branch_weight = float(adaptation.get("low_shift_branch_weight", 0.90))
                weight_sum = max(main_weight + branch_weight, 1e-8)
                main_weight /= weight_sum
                branch_weight /= weight_sum
                transformer_prob_raw = main_weight * main_transformer_prob + branch_weight * branch_prob_raw
                transformer_fixed_threshold = float(
                    adaptation.get("low_shift_threshold", default_transformer_roc_threshold)
                )
                transformer_strategy = "adaptive_main_top28_blend"

            print(
                f"外部自适应分支启用: shift={shift_score:.4f}, "
                f"threshold={shift_threshold:.4f}, selected_top_genes={selected_branch}, "
                f"strategy={transformer_strategy}"
            )

        transformer_prob = transformer_prob_raw
        transformer_primary_threshold = _quantile_threshold_from_positive_rate(
            transformer_prob,
            train_positive_rate,
        )

        svm_probs = {}
        for model_name, model in svm_models.items():
            svm_probs[model_name] = model.predict_proba(X_ext_svm)[:, 1]

        ensemble_prob = predict_from_ensemble_obj(ensemble_obj, X_ext_svm)
        if ensemble_prob is not None:
            svm_probs["Ensemble_SVM"] = ensemble_prob

        all_model_probs = {"Transformer": transformer_prob}
        all_model_probs.update(svm_probs)

        prediction_frame = pd.DataFrame({
            "sample_index": np.arange(len(y_ext)),
            "y_true": y_ext,
            "ConfiguredLabelFlip": group_info["ConfiguredLabelFlip"],
        })

        model_outputs = []
        ordered_names = ["Transformer", "SVM", "Voting_SVM", "Bagging_SVM", "Ensemble_SVM"]
        for name in ordered_names:
            if name not in all_model_probs:
                continue
            y_prob = all_model_probs[name]
            prediction_frame[f"{name}_prob"] = y_prob
            if name == "Transformer":
                primary_threshold = transformer_primary_threshold
                fixed_threshold = transformer_fixed_threshold
            else:
                primary_threshold = _quantile_threshold_from_positive_rate(y_prob, train_positive_rate)
                fixed_threshold = svm_thresholds.get(name, 0.5)
            fpr_m, tpr_m, thresholds_m = roc_curve(y_ext, y_prob)
            youden_idx = np.argmax(tpr_m - fpr_m)
            retrospective_threshold = float(thresholds_m[youden_idx])
            model_outputs.append((name, y_prob, primary_threshold, fixed_threshold, retrospective_threshold))

        prediction_frame.to_csv(
            os.path.join(config.EXTERNAL_VALIDATION_DIR, f"predictions_{dataset_name}.csv"),
            index=False,
        )

        print("各模型阈值（primary=train_prior_quantile / fixed_internal / retrospective-Youden）:")
        for name, _, primary_thr, fixed_thr, retro_thr in model_outputs:
            print(
                f"  {name:<13} primary={primary_thr:.4f} | "
                f"fixed={fixed_thr:.4f} | retro={retro_thr:.4f}"
            )

        for model_name, y_prob, primary_threshold, fixed_threshold, retrospective_threshold in model_outputs:
            fpr, tpr, _ = roc_curve(y_ext, y_prob)
            model_auc = auc(fpr, tpr)
            y_pred = (y_prob >= primary_threshold).astype(int)
            model_acc = accuracy_score(y_ext, y_pred)
            inference_strategy = transformer_strategy if model_name == "Transformer" else "fixed_model"

            summary_rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "AUC": model_auc,
                "Accuracy": model_acc,
                "Threshold": primary_threshold,
                "ThresholdStrategy": "train_prior_quantile",
                "InferenceStrategy": inference_strategy,
                "PresentGenes": group_info["present_genes"],
                "MissingGenes": group_info["missing_genes"],
                "CoverageRatio": group_info["coverage_ratio"],
                "LabelCorr": corr,
                "LabelFlipped": flipped,
                "ConfiguredLabelFlip": group_info["ConfiguredLabelFlip"],
                "LabelFlipReason": group_info["LabelFlipReason"],
                "LabelPolarity": group_info["LabelPolarity"],
            })
            print(
                f"{model_name:<13} AUC={model_auc:.4f}, Accuracy={model_acc:.4f}, "
                f"PrimaryThreshold={primary_threshold:.4f}"
            )

            fixed_internal_pred = (y_prob >= fixed_threshold).astype(int)
            fixed_internal_rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "AUC": model_auc,
                "Accuracy": accuracy_score(y_ext, fixed_internal_pred),
                "Threshold": fixed_threshold,
                "ThresholdStrategy": "fixed_internal",
                "InferenceStrategy": inference_strategy,
                "PresentGenes": group_info["present_genes"],
                "MissingGenes": group_info["missing_genes"],
                "CoverageRatio": group_info["coverage_ratio"],
                "LabelCorr": corr,
                "LabelFlipped": flipped,
                "ConfiguredLabelFlip": group_info["ConfiguredLabelFlip"],
                "LabelFlipReason": group_info["LabelFlipReason"],
                "LabelPolarity": group_info["LabelPolarity"],
            })

            retrospective_pred = (y_prob >= retrospective_threshold).astype(int)
            retrospective_rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "AUC": model_auc,
                "Accuracy": accuracy_score(y_ext, retrospective_pred),
                "Threshold": retrospective_threshold,
                "ThresholdStrategy": "retrospective_youden",
                "InferenceStrategy": inference_strategy,
                "PresentGenes": group_info["present_genes"],
                "MissingGenes": group_info["missing_genes"],
                "CoverageRatio": group_info["coverage_ratio"],
                "LabelCorr": corr,
                "LabelFlipped": flipped,
                "ConfiguredLabelFlip": group_info["ConfiguredLabelFlip"],
                "LabelFlipReason": group_info["LabelFlipReason"],
                "LabelPolarity": group_info["LabelPolarity"],
            })

            if model_name == "Transformer":
                plot_tsne(
                    transformer_features,
                    y_ext,
                    os.path.join(config.EXTERNAL_VALIDATION_DIR, f"tsne_{dataset_name}_{model_name}.png"),
                    f"{dataset_name} - Transformer Feature t-SNE",
                )

            plot_probability_distribution(
                y_ext,
                y_prob,
                primary_threshold,
                os.path.join(config.EXTERNAL_VALIDATION_DIR, f"prob_{dataset_name}_{model_name}.png"),
                f"{dataset_name} - {model_name} Probability Distribution",
            )
            plot_cm(
                confusion_matrix(y_ext, y_pred),
                f"{dataset_name} - {model_name} Confusion Matrix",
                os.path.join(config.EXTERNAL_VALIDATION_DIR, f"cm_{dataset_name}_{model_name}.png"),
            )

            curve_label = f"{dataset_name}-{model_name} (AUC={model_auc:.3f})"
            lw = 1.8 if model_name == "Transformer" else 1.2
            alpha = 0.9 if model_name == "Transformer" else 0.7
            plt.plot(fpr, tpr, lw=lw, alpha=alpha, label=curve_label)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(config.EXTERNAL_VALIDATION_DIR, "external_validation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    fixed_internal_df = pd.DataFrame(fixed_internal_rows)
    fixed_internal_path = os.path.join(
        config.EXTERNAL_VALIDATION_DIR,
        "external_validation_fixed_internal_summary.csv",
    )
    fixed_internal_df.to_csv(fixed_internal_path, index=False)

    retrospective_df = pd.DataFrame(retrospective_rows)
    retrospective_path = os.path.join(
        config.EXTERNAL_VALIDATION_DIR,
        "external_validation_retrospective_summary.csv",
    )
    retrospective_df.to_csv(retrospective_path, index=False)

    representative_df = summary_df[summary_df["Model"].isin(["Transformer", "Bagging_SVM"])].copy()
    representative_path = os.path.join(
        config.EXTERNAL_VALIDATION_DIR,
        "external_validation_representative_summary.csv",
    )
    representative_df.to_csv(representative_path, index=False)

    pivot_auc = summary_df.set_index(["Dataset", "Model"])["AUC"].unstack()
    pivot_acc = summary_df.set_index(["Dataset", "Model"])["Accuracy"].unstack()
    pivot_auc.to_csv(os.path.join(config.EXTERNAL_VALIDATION_DIR, "external_validation_auc_pivot.csv"))
    pivot_acc.to_csv(os.path.join(config.EXTERNAL_VALIDATION_DIR, "external_validation_accuracy_pivot.csv"))

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("External Validation ROC Summary")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.EXTERNAL_VALIDATION_DIR, "external_validation_roc.png"), dpi=300)
    plt.close()

    plot_grouped_metric_barplot(
        summary_df,
        x="Dataset",
        y="AUC",
        hue="Model",
        save_path=os.path.join(config.EXTERNAL_VALIDATION_DIR, "external_validation_auc_comparison.png"),
        title="External Validation AUC Comparison",
        xlabel="Dataset",
        ylabel="AUC",
    )
    plot_grouped_metric_barplot(
        summary_df,
        x="Dataset",
        y="Accuracy",
        hue="Model",
        save_path=os.path.join(config.EXTERNAL_VALIDATION_DIR, "external_validation_accuracy_comparison.png"),
        title="External Validation Accuracy Comparison",
        xlabel="Dataset",
        ylabel="Accuracy",
    )

    print("\n" + "=" * 60)
    print("外部验证完成")
    print(f"结果保存至: {config.EXTERNAL_VALIDATION_DIR}")
    print(f"主结果文件(train_prior_quantile): {summary_path}")
    print(f"补充结果文件(fixed_internal): {fixed_internal_path}")
    print(f"补充结果文件(retrospective): {retrospective_path}")
    print(f"SVM 家族代表摘要: {representative_path}")
    print(summary_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    evaluate_external_models()
