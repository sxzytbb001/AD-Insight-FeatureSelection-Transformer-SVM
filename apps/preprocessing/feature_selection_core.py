from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from common import benjamini_hochberg, cohens_d
from apps.preprocessing.feature_selection import (
    _evaluate_gene_set_auc,
    _mrmr_ranking,
    _rank_score,
    _stability_selection,
    _xgboost_gain_ranking,
)


@dataclass(frozen=True)
class FeatureSelectionResult:
    candidate_genes: List[str]
    ranking: pd.DataFrame
    method_weights: pd.DataFrame
    method_rankings: Dict[str, List[str]]


def run_ensemble_feature_selection(
    X_df,
    y,
    candidate_gene_count=30,
    prefilter_top_n=1200,
    method_top_k=100,
    stability_iterations=50,
    stability_prefilter_top_n=400,
    min_vote_count=2,
    max_fdr=0.2,
):
    """Run the ensemble feature-selection algorithm on a caller-provided split.

    This is the fold-safe interface used by nested and LOCO validation. Callers
    pass only training samples, so validation samples cannot influence the gene
    panel.
    """
    X_df = pd.DataFrame(X_df).copy()
    y = np.asarray(y, dtype=int)
    if len(X_df) != len(y):
        raise ValueError("X_df and y must contain the same number of samples.")
    if len(np.unique(y)) != 2:
        raise ValueError("Feature selection requires both binary classes.")

    top_n = min(int(prefilter_top_n), X_df.shape[1])
    method_top_k = min(int(method_top_k), top_n)
    candidate_gene_count = min(int(candidate_gene_count), top_n)

    group_positive = X_df.iloc[y == 1]
    group_control = X_df.iloc[y == 0]
    t_stats, p_vals = stats.ttest_ind(
        group_positive,
        group_control,
        axis=0,
        equal_var=False,
        nan_policy="omit",
    )
    effect_sizes = [
        cohens_d(group_positive[col].values, group_control[col].values)
        for col in X_df.columns
    ]
    fdr_vals = benjamini_hochberg(np.nan_to_num(p_vals, nan=1.0))

    stats_df = pd.DataFrame(
        {
            "gene": X_df.columns,
            "t_stat": np.nan_to_num(t_stats, nan=0.0),
            "p_value": np.nan_to_num(p_vals, nan=1.0),
            "fdr": fdr_vals,
            "cohens_d": effect_sizes,
            "mean_positive": group_positive.mean(axis=0).values,
            "mean_control": group_control.mean(axis=0).values,
            "abs_t_stat": np.abs(np.nan_to_num(t_stats, nan=0.0)),
            "abs_effect_size": np.abs(effect_sizes),
        }
    ).sort_values(["p_value", "abs_effect_size"], ascending=[True, False])

    prefilter_genes = stats_df.head(top_n)["gene"].tolist()
    X_prefilter = X_df[prefilter_genes]
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X_prefilter),
        columns=prefilter_genes,
        index=X_df.index,
    )

    mi_scores = mutual_info_classif(X_scaled, y, discrete_features=False, random_state=42)
    mi_df = pd.DataFrame({"gene": prefilter_genes, "score": mi_scores}).sort_values(
        "score",
        ascending=False,
    )
    mi_top = mi_df.head(method_top_k)["gene"].tolist()

    xgb_top, xgb_series, _ = _xgboost_gain_ranking(X_scaled, y, top_k=method_top_k)

    enet = LogisticRegressionCV(
        Cs=10,
        cv=min(5, max(2, int(np.bincount(y).min()))),
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.5],
        scoring="roc_auc",
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    enet.fit(X_scaled, y)
    enet_series = pd.Series(
        np.abs(enet.coef_).ravel(),
        index=prefilter_genes,
    ).sort_values(ascending=False)
    enet_top = enet_series.head(method_top_k).index.tolist()

    mrmr_top, mrmr_series = _mrmr_ranking(
        X_scaled,
        y,
        top_k=method_top_k,
        max_candidates=min(300, len(prefilter_genes)),
    )

    stability_candidate_genes = mi_df.head(
        min(stability_prefilter_top_n, len(prefilter_genes))
    )["gene"].tolist()
    stability_top, stability_freq = _stability_selection(
        X_scaled[stability_candidate_genes],
        y,
        top_k=method_top_k,
        n_iterations=stability_iterations,
        threshold=0.8,
        sample_ratio=0.8,
        max_iter=1500,
        tol=1e-3,
        verbose_every=max(1, stability_iterations // 5),
    )

    ttest_top = (
        stats_df[stats_df["gene"].isin(prefilter_genes)]
        .sort_values("abs_t_stat", ascending=False)
        .head(method_top_k)["gene"]
        .tolist()
    )

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_scaled, y)
    rf_series = pd.Series(
        rf_model.feature_importances_,
        index=prefilter_genes,
    ).sort_values(ascending=False)
    rf_top = rf_series.head(method_top_k).index.tolist()

    method_rankings = {
        "TTest": ttest_top,
        "MutualInfo": mi_top,
        "XGBoost": xgb_top,
        "RandomForest": rf_top,
        "ElasticNet": enet_top,
        "mRMR": mrmr_top,
        "Stability": stability_top,
    }

    method_perf = {}
    cv_splits = min(3, int(np.bincount(y).min()))
    for method, ranking in method_rankings.items():
        method_perf[method] = _evaluate_gene_set_auc(
            X_scaled,
            y,
            ranking,
            n_splits=max(2, cv_splits),
            max_genes=min(30, len(ranking)),
        )

    perf_series = pd.Series(method_perf)
    raw_weights = np.clip(perf_series.values - 0.5, 1e-3, None)
    norm_weights = raw_weights / np.sum(raw_weights)
    method_weights = dict(zip(perf_series.index.tolist(), norm_weights.tolist()))

    score_map = {gene: 0.0 for gene in prefilter_genes}
    for method, ranking in method_rankings.items():
        scores = _rank_score(ranking, top_k=method_top_k, weight=method_weights[method])
        for gene, score in scores.items():
            score_map[gene] += score

    vote_count = {
        gene: sum(gene in method_rankings[method] for method in method_rankings)
        for gene in prefilter_genes
    }

    merged_df = stats_df.copy()
    merged_df["mi_score"] = merged_df["gene"].map(mi_df.set_index("gene")["score"]).fillna(0.0)
    merged_df["xgb_gain"] = merged_df["gene"].map(xgb_series).fillna(0.0)
    merged_df["random_forest_importance"] = merged_df["gene"].map(rf_series).fillna(0.0)
    merged_df["elastic_net_coef"] = merged_df["gene"].map(enet_series).fillna(0.0)
    merged_df["mrmr_score"] = merged_df["gene"].map(mrmr_series).fillna(0.0)
    merged_df["stability_freq"] = merged_df["gene"].map(stability_freq).fillna(0.0)
    merged_df["vote_score"] = merged_df["gene"].map(score_map).fillna(0.0)
    merged_df["vote_count"] = merged_df["gene"].map(vote_count).fillna(0).astype(int)
    merged_df["quality_score"] = (
        0.45 * merged_df["vote_score"]
        + 0.20 * (merged_df["abs_t_stat"] / (merged_df["abs_t_stat"].max() + 1e-8))
        + 0.20
        * (merged_df["abs_effect_size"] / (merged_df["abs_effect_size"].max() + 1e-8))
        + 0.15 * (1 - merged_df["fdr"])
    )

    filtered_df = merged_df[
        (merged_df["vote_count"] >= min_vote_count) & (merged_df["fdr"] < max_fdr)
    ].copy()
    if len(filtered_df) < candidate_gene_count:
        filtered_df = merged_df[merged_df["vote_count"] >= min_vote_count].copy()
    if len(filtered_df) < candidate_gene_count:
        filtered_df = merged_df.copy()

    final_df = filtered_df.sort_values(
        ["quality_score", "vote_count", "abs_effect_size"],
        ascending=False,
    ).head(candidate_gene_count)
    method_weights_df = pd.DataFrame(
        {
            "method": list(method_rankings.keys()),
            "validation_auc": [method_perf[method] for method in method_rankings],
            "weight": [method_weights[method] for method in method_rankings],
        }
    ).sort_values("validation_auc", ascending=False)

    return FeatureSelectionResult(
        candidate_genes=final_df["gene"].tolist(),
        ranking=merged_df.sort_values("quality_score", ascending=False),
        method_weights=method_weights_df,
        method_rankings=method_rankings,
    )


def write_feature_selection_result(result: FeatureSelectionResult, output_dir):
    import os

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "candidate_genes.txt"), "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(result.candidate_genes))
    result.ranking.to_csv(os.path.join(output_dir, "feature_selection_ranking.csv"), index=False)
    result.method_weights.to_csv(os.path.join(output_dir, "method_weights.csv"), index=False)
