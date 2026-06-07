import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.svm import SVC

import config
from apps.visualization import plot_confusion_matrix, plot_ranked_importance
from common import fit_rank_gauss_preprocessor, load_training_data, save_json, transform_with_preprocessor

config.ensure_dirs()

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

SVM_PARAM_GRID = [
    {
        'kernel': ['rbf'],
        'C': [0.25, 0.5, 1, 2, 5, 10, 20],
        'gamma': ['scale', 0.0005, 0.001, 0.002, 0.005, 0.01]
    },
    {
        'kernel': ['poly'],
        'C': [0.5, 1, 2, 5, 10],
        'gamma': ['scale', 0.0005, 0.001, 0.002, 0.005],
        'degree': [2, 3]
    }
]


def build_svm_model(params):
    return SVC(
        kernel=params.get('kernel', 'rbf'),
        C=params.get('C', 1.0),
        gamma=params.get('gamma', 'scale'),
        degree=params.get('degree', 3),
        probability=True,
        class_weight='balanced',
        random_state=42
    )


def _gamma_priority(value):
    if value == 'scale':
        return 0.00075
    if value == 'auto':
        return 0.0015
    return float(value)


def _extract_param_value(row, key, default=None):
    value = row.get(f'param_{key}', default)
    if pd.isna(value):
        return default
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def _stable_candidate_frame(ranking_df):
    frame = ranking_df.copy()
    frame['kernel_rank'] = frame['param_kernel'].map({'rbf': 0, 'poly': 1}).fillna(9)
    frame['c_rank'] = pd.to_numeric(frame['param_C'], errors='coerce').fillna(999.0)
    frame['degree_rank'] = pd.to_numeric(frame.get('param_degree', 3), errors='coerce').fillna(3.0)
    frame['gamma_rank'] = frame['param_gamma'].apply(_gamma_priority)
    return frame


def _select_stable_best_row(ranking_df):
    ordered = _stable_candidate_frame(ranking_df).sort_values(
        by=['mean_test_score', 'std_test_score', 'c_rank', 'degree_rank', 'gamma_rank', 'kernel_rank'],
        ascending=[False, True, True, True, True, True]
    ).reset_index(drop=True)
    ordered['original_rank'] = np.arange(1, len(ordered) + 1)
    top_row = ordered.iloc[0]
    stable_floor = float(top_row['mean_test_score'] - top_row['std_test_score'])
    stable_candidates = ordered[ordered['mean_test_score'] >= stable_floor].copy()
    stable_candidates = stable_candidates.sort_values(
        by=['std_test_score', 'c_rank', 'degree_rank', 'gamma_rank', 'kernel_rank', 'mean_test_score'],
        ascending=[True, True, True, True, True, False]
    ).reset_index(drop=True)
    return stable_candidates.iloc[0], stable_candidates, stable_floor


def select_best_svm_via_grid(
    X,
    y,
    cv_splits=5,
    cv_repeats=3,
    save_results_path=None,
    log_title='GridSearchCV 搜索最佳 SVM 参数'
):
    print(f'\n>>> {log_title}...')
    base_model = SVC(class_weight='balanced', random_state=42)
    cv5 = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=42)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=SVM_PARAM_GRID,
        scoring='roc_auc',
        cv=cv5,
        n_jobs=-1,
        verbose=0,
        return_train_score=False
    )
    grid.fit(X, y)

    ranking_df = pd.DataFrame(grid.cv_results_)
    ranking_df = _stable_candidate_frame(ranking_df).sort_values(
        by=['mean_test_score', 'std_test_score', 'c_rank', 'degree_rank', 'gamma_rank', 'kernel_rank'],
        ascending=[False, True, True, True, True, True]
    ).reset_index(drop=True)
    best_row, stable_candidates, stable_floor = _select_stable_best_row(ranking_df)
    if save_results_path:
        ranking_df.to_csv(save_results_path, index=False)
        stable_candidates.to_csv(
            os.path.splitext(save_results_path)[0] + '_stable_candidates.csv',
            index=False
        )

    best_params = {
        'kernel': _extract_param_value(best_row, 'kernel', 'rbf'),
        'C': float(_extract_param_value(best_row, 'C', 1.0)),
        'gamma': _extract_param_value(best_row, 'gamma', 'scale')
    }
    selected_degree = _extract_param_value(best_row, 'degree')
    if selected_degree is not None:
        best_params['degree'] = int(selected_degree)

    best_score = float(best_row['mean_test_score'])
    selection_meta = {
        'cv_splits': int(cv_splits),
        'cv_repeats': int(cv_repeats),
        'best_mean_auc': float(ranking_df.iloc[0]['mean_test_score']),
        'best_std_auc': float(ranking_df.iloc[0]['std_test_score']),
        'stable_floor_auc': float(stable_floor),
        'stable_candidate_count': int(len(stable_candidates)),
        'selected_mean_auc': float(best_row['mean_test_score']),
        'selected_std_auc': float(best_row['std_test_score']),
        'selected_rank': int(best_row['original_rank'])
    }
    print(f'  最佳参数: {best_params}')
    print(f'  {cv_splits}x{cv_repeats} repeated CV AUC: {best_score:.4f}')
    print(f'  Stable floor AUC: {stable_floor:.4f} | candidates: {len(stable_candidates)}')
    return best_params, best_score, ranking_df, selection_meta


def build_voting_ensemble(best_params):
    C = best_params.get('C', 1.0)
    gamma = best_params.get('gamma', 'scale')
    degree = best_params.get('degree', 3)

    voting = VotingClassifier(
        estimators=[
            ('svm_rbf', SVC(kernel='rbf', C=C, gamma=gamma, probability=True, class_weight='balanced', random_state=42)),
            ('svm_poly', SVC(kernel='poly', C=max(0.5, C), gamma=gamma, degree=degree, probability=True, class_weight='balanced', random_state=42)),
            ('svm_linear', SVC(kernel='linear', C=C, probability=True, class_weight='balanced', random_state=42))
        ],
        voting='soft',
        weights=[3, 2, 1],
        n_jobs=-1
    )
    return voting


def build_bagging_svm(best_params):
    base = SVC(
        kernel=best_params.get('kernel', 'rbf'),
        C=best_params.get('C', 1.0),
        gamma=best_params.get('gamma', 'scale'),
        degree=best_params.get('degree', 3),
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    return BaggingClassifier(
        estimator=base,
        n_estimators=10,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )


def _calc_best_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[best_idx]), fpr, tpr


def cross_validate_models(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = {
        'SVM': np.zeros(len(y), dtype=float),
        'Voting_SVM': np.zeros(len(y), dtype=float),
        'Bagging_SVM': np.zeros(len(y), dtype=float)
    }
    fold_param_records = []

    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        preprocessor, X_train = fit_rank_gauss_preprocessor(X[train_idx])
        X_val = transform_with_preprocessor(preprocessor, X[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        fold_best_params, fold_inner_auc, _, fold_selection_meta = select_best_svm_via_grid(
            X_train,
            y_train,
            cv_splits=4,
            cv_repeats=2,
            save_results_path=None,
            log_title=f'Fold {fold} 内层4折搜索最佳 SVM 参数'
        )
        fold_param_records.append({
            'fold': fold,
            'best_params': fold_best_params,
            'inner_auc': fold_inner_auc,
            'selection_meta': fold_selection_meta,
        })

        model_single = build_svm_model(fold_best_params)
        model_voting = build_voting_ensemble(fold_best_params)
        model_bagging = build_bagging_svm(fold_best_params)

        model_single.fit(X_train, y_train)
        model_voting.fit(X_train, y_train)
        model_bagging.fit(X_train, y_train)

        prob_single = model_single.predict_proba(X_val)[:, 1]
        prob_voting = model_voting.predict_proba(X_val)[:, 1]
        prob_bagging = model_bagging.predict_proba(X_val)[:, 1]

        oof_probs['SVM'][val_idx] = prob_single
        oof_probs['Voting_SVM'][val_idx] = prob_voting
        oof_probs['Bagging_SVM'][val_idx] = prob_bagging

        auc_single = roc_auc_score(y_val, prob_single)
        auc_voting = roc_auc_score(y_val, prob_voting)
        auc_bagging = roc_auc_score(y_val, prob_bagging)

        print(
            f'  Fold {fold}: innerAUC={fold_inner_auc:.4f}, '
            f'SVM={auc_single:.4f}, Voting={auc_voting:.4f}, Bagging={auc_bagging:.4f}'
        )

        fpr_v, tpr_v, _ = roc_curve(y_val, prob_voting)
        plt.plot(fpr_v, tpr_v, lw=1.0, alpha=0.35, label=f'Fold {fold} Voting (AUC={auc_voting:.3f})')

    summary = {}
    for name, probs in oof_probs.items():
        threshold, fpr, tpr = _calc_best_threshold(y, probs)
        auc_score = auc(fpr, tpr)
        pred = (probs >= threshold).astype(int)
        acc_score = accuracy_score(y, pred)
        summary[name] = {
            'oof_prob': probs,
            'auc': float(auc_score),
            'acc': float(acc_score),
            'threshold': float(threshold),
            'fpr': fpr,
            'tpr': tpr
        }

    plt.plot(summary['SVM']['fpr'], summary['SVM']['tpr'], lw=2, label=f'SVM OOF (AUC={summary["SVM"]["auc"]:.3f})')
    plt.plot(summary['Voting_SVM']['fpr'], summary['Voting_SVM']['tpr'], lw=2, label=f'Voting OOF (AUC={summary["Voting_SVM"]["auc"]:.3f})')
    plt.plot(summary['Bagging_SVM']['fpr'], summary['Bagging_SVM']['tpr'], lw=2, label=f'Bagging OOF (AUC={summary["Bagging_SVM"]["auc"]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM 5-Fold Cross-Validation ROC')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.SVM_DIR, 'svm_roc.png'), dpi=300)
    plt.close()

    summary['fold_best_params'] = fold_param_records
    return summary


def train_svm():
    print('=' * 60)
    print('SVM 训练：5折网格搜索 + Voting + Bagging')
    print('=' * 60)

    candidate_genes = config.load_candidate_genes()
    if not candidate_genes:
        print('错误：未找到 候选基因，请先运行 python -m scripts.preprocessing.feature_selection')
        return

    X, y, available_genes, _, labels_df = load_training_data(candidate_genes)
    X = np.asarray(X, dtype=np.float32)
    print(f'加载候选基因: {len(available_genes)} 个')
    print(f'训练数据: {X.shape[0]} 样本, {X.shape[1]} 基因')
    print(f'类别分布: Positive={y.sum()}, Control={len(y) - y.sum()}')

    preprocessor, X_processed = fit_rank_gauss_preprocessor(X)

    best_params, best_grid_auc, ranking_df, selection_meta = select_best_svm_via_grid(
        X_processed,
        y,
        cv_splits=5,
        cv_repeats=3,
        save_results_path=os.path.join(config.SVM_DIR, 'grid_search_results.csv'),
        log_title='步骤2: 5折 GridSearchCV 搜索最佳 SVM 参数'
    )
    cv_summary = cross_validate_models(X, y)

    print('\n>>> 步骤3: 训练最终模型并保存...')
    final_single = build_svm_model(best_params)
    final_voting = build_voting_ensemble(best_params)
    final_bagging = build_bagging_svm(best_params)

    final_single.fit(X_processed, y)
    final_voting.fit(X_processed, y)
    final_bagging.fit(X_processed, y)

    with open(config.SVM_MODEL_PATH, 'wb') as f:
        pickle.dump(final_single, f)
    with open(config.SVM_VOTING_PATH, 'wb') as f:
        pickle.dump(final_voting, f)
    with open(config.SVM_BAGGING_PATH, 'wb') as f:
        pickle.dump(final_bagging, f)
    with open(config.SVM_ENSEMBLE_PATH, 'wb') as f:
        pickle.dump({'voting': final_voting, 'bagging': final_bagging}, f)
    with open(config.SVM_SCALER_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)

    params_to_save = {
        'best_grid_params': best_params,
        'best_grid_auc': best_grid_auc,
        'selection_meta': selection_meta,
        'search_space': SVM_PARAM_GRID,
        'thresholds': {
            'SVM': cv_summary['SVM']['threshold'],
            'Voting_SVM': cv_summary['Voting_SVM']['threshold'],
            'Bagging_SVM': cv_summary['Bagging_SVM']['threshold']
        },
        'oof_metrics': {
            'SVM': {'auc': cv_summary['SVM']['auc'], 'acc': cv_summary['SVM']['acc']},
            'Voting_SVM': {'auc': cv_summary['Voting_SVM']['auc'], 'acc': cv_summary['Voting_SVM']['acc']},
            'Bagging_SVM': {'auc': cv_summary['Bagging_SVM']['auc'], 'acc': cv_summary['Bagging_SVM']['acc']}
        },
        'fold_best_params': cv_summary['fold_best_params'],
        'preprocessing': 'rank_gauss_standard',
        'candidate_gene_count': len(available_genes)
    }
    save_json(params_to_save, config.SVM_PARAMS_PATH)

    comparison_df = pd.DataFrame([
        {'Model': 'SVM', 'AUC': cv_summary['SVM']['auc'], 'Accuracy': cv_summary['SVM']['acc']},
        {'Model': 'Voting_SVM', 'AUC': cv_summary['Voting_SVM']['auc'], 'Accuracy': cv_summary['Voting_SVM']['acc']},
        {'Model': 'Bagging_SVM', 'AUC': cv_summary['Bagging_SVM']['auc'], 'Accuracy': cv_summary['Bagging_SVM']['acc']}
    ])
    comparison_df.to_csv(os.path.join(config.SVM_DIR, 'model_comparison.csv'), index=False)

    oof_df = pd.DataFrame({
        'sample_id': labels_df['sample_id'],
        'label': y,
        'SVM_prob': cv_summary['SVM']['oof_prob'],
        'Voting_SVM_prob': cv_summary['Voting_SVM']['oof_prob'],
        'Bagging_SVM_prob': cv_summary['Bagging_SVM']['oof_prob'],
    })
    oof_df.to_csv(os.path.join(config.SVM_DIR, 'oof_predictions.csv'), index=False)

    print('\n>>> 步骤4: 生成可视化结果...')
    single_prob = final_single.predict_proba(X_processed)[:, 1]
    single_pred = (single_prob >= cv_summary['SVM']['threshold']).astype(int)
    cm = confusion_matrix(y, single_pred)
    plot_confusion_matrix(
        cm,
        os.path.join(config.SVM_DIR, 'svm_confusion_matrix.png'),
        'SVM Confusion Matrix',
    )

    perm = permutation_importance(final_single, X_processed, y, n_repeats=20, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({
        'gene': available_genes,
        'importance': perm.importances_mean,
        'std': perm.importances_std
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(config.SVM_DIR, 'svm_feature_importance.csv'), index=False)

    plot_ranked_importance(
        importance_df,
        score_col='importance',
        label_col='gene',
        save_path=os.path.join(config.SVM_DIR, 'svm_feature_importance.png'),
        title='SVM Feature Importance (Top 15)',
        xlabel='Permutation Importance',
        top_n=15,
    )

    print('\n>>> 训练结果汇总')
    print(f"GridSearch 最佳参数: {best_params}, AUC={best_grid_auc:.4f}")
    print(f"SVM OOF AUC={cv_summary['SVM']['auc']:.4f}, Acc={cv_summary['SVM']['acc']:.4f}")
    print(f"Voting OOF AUC={cv_summary['Voting_SVM']['auc']:.4f}, Acc={cv_summary['Voting_SVM']['acc']:.4f}")
    print(f"Bagging OOF AUC={cv_summary['Bagging_SVM']['auc']:.4f}, Acc={cv_summary['Bagging_SVM']['acc']:.4f}")
    print(f'结果保存至: {config.SVM_DIR}')


if __name__ == '__main__':
    train_svm()
