"""E2 runner: Additive vs. Threshold synthetic regime + main-effects ablation.

For each regime in {additive, threshold} and each WAF variant
{full-with-interactions, main-effects-only}:

1. Generate the SEAB-style synthetic dataset (16 attribute combos x 100 = 1600
   samples, uniform emotion).
2. Compute traditional fairness metrics (EO / SP / FPR) per attribute-emotion.
3. Fit the linear WAF on the synthetic dataset (no speech features).
4. Compute Pearson correlations between attribute-error MI and each metric.

Outputs a single wide CSV that mirrors Table V from the paper.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from databases.constants import demographic_columns, id2label, label2id, labels
from databases.synthetic.constants import THRESHOLD_K, mappings
from databases.synthetic.generate_dataset import (calculate_contributions,
                                                  compute_loss,
                                                  generate_dataset)
from databases.synthetic.mutual_info import (calculate_anova,
                                             calculate_mi_continuous)
from fairness.constants import privileged_groups
from fairness.trad_metrics.trad_metrics import calculate_traditional_fairness_metrics
from fairness.waf.linear_waf import (bce_per_class_np, build_feature_matrix,
                                     fit_linear_waf, pairwise_interactions)
from utils import set_seed


def _attribute_error_mi(dataset: pd.DataFrame, mapping) -> pd.DataFrame:
    """Compute attribute-error MI per (attribute, emotion) pair.

    Returns a DataFrame with columns [protected_attribute, emotion, mi].
    """
    # Expand ``loss`` (per-class 6-vector) into per-class scalar columns.
    losses = pd.DataFrame(dataset["loss"].tolist(), columns=labels)
    df = pd.concat(
        [dataset.reset_index(drop=True), losses.reset_index(drop=True)], axis=1
    )
    rows = []
    for c, emo in enumerate(labels):
        mi = calculate_mi_continuous(df, demographic_columns, emo)
        for i, attr in enumerate(demographic_columns):
            rows.append({"protected_attribute": attr, "emotion": emo, "mi": float(mi[i])})
    return pd.DataFrame(rows)


def _waf_scores_linear(
    dataset: pd.DataFrame, include_interactions: bool
):
    """Fit linear WAF on the synthetic data with no speech features.

    Returns
    -------
    waf_df : DataFrame [protected_attribute, emotion, waf]  -- main-effect
             coefficients (what the WAF metric is defined as).
    fit_summary : dict with average R^2 across emotions and full parameter
                  table (including interaction terms for the ablation).
    """
    demog = list(demographic_columns)
    X = build_feature_matrix(
        dataset, demographic_cols=demog, speech_cols=[],
        include_interactions=include_interactions,
    )
    interaction_cols = (
        list(pairwise_interactions(dataset, demog).columns)
        if include_interactions else []
    )
    y_idx = np.array([label2id[e] for e in dataset["emotion"]])
    y_onehot = np.eye(len(labels))[y_idx]
    scores = np.stack([np.asarray(s, dtype=np.float32) for s in dataset["pred_scores"]])
    Y = bce_per_class_np(scores, y_onehot)

    fit = fit_linear_waf(
        X=X, Y=Y, emotion_labels=labels,
        demographic_cols=demog, interaction_cols=interaction_cols,
        speech_cols=[],
    )
    tbl = fit.table_iii()
    waf_df = tbl[tbl["feature_type"] == "main"][["feature", "emotion", "coef"]]
    waf_df = waf_df.rename(columns={"feature": "protected_attribute", "coef": "waf"})

    r2s = [f.result.rsquared for f in fit.fits]
    fit_summary = {
        "mean_r2": float(np.mean(r2s)),
        "r2_per_emotion": {f.emotion: float(f.result.rsquared) for f in fit.fits},
        "table": tbl,
    }
    return waf_df, fit_summary


def _trad_metrics(dataset: pd.DataFrame) -> pd.DataFrame:
    """Compute SP/EO/FPR per (attribute, emotion) using aif360.

    ``calculate_traditional_fairness_metrics`` expects:
      * ``predicted`` to be an integer id in [0..len(labels)-1]
      * demographic columns to be raw categorical strings that appear in
        ``fairness.constants.privileged_groups`` (it re-binarises internally).
    Our synthetic dataset has string emotion labels and +-1 numeric demographic
    columns, so we massage the frame here.
    """
    df = dataset.copy()
    # 1) predicted: string -> int id
    df["predicted"] = df["predicted"].map(label2id).astype(int)
    # 2) demographic columns: +-1 -> canonical strings from privileged_groups
    canon = {
        "Sex": ("Male", "Female"),
        "Race": ("Caucasian", "Other"),
        "AgeGroup": ("20-25", "45-55"),
        "Ethnicity": ("Not Hispanic", "Hispanic"),
    }
    for col in demographic_columns:
        priv_val, unpriv_val = canon[col]
        df[col] = df[col].map({1: priv_val, -1: unpriv_val})
    return calculate_traditional_fairness_metrics(df)


def _merge_and_correlate(
    mi_df: pd.DataFrame, waf_df: pd.DataFrame, trad_df: pd.DataFrame
):
    """Merge on (attribute, emotion) and return Pearson & Spearman r
    for MI vs each metric, plus p-values."""
    merged = mi_df.merge(waf_df, on=["protected_attribute", "emotion"], how="left")
    merged = merged.merge(trad_df, on=["protected_attribute", "emotion"], how="left")

    out = {}
    for m in ["Equal Opportunity", "Statistical Parity", "False Positive Rate", "waf"]:
        v = merged[m].astype(float)
        u = merged["mi"].astype(float)
        mask = np.isfinite(v) & np.isfinite(u)
        if mask.sum() < 3 or v[mask].nunique() < 2 or u[mask].nunique() < 2:
            out[m] = {"pearson_r": float("nan"), "pearson_p": float("nan"),
                      "spearman_r": float("nan"), "spearman_p": float("nan")}
            continue
        pr, pp = pearsonr(u[mask], v[mask])
        sr, sp = spearmanr(u[mask], v[mask])
        out[m] = {"pearson_r": float(pr), "pearson_p": float(pp),
                  "spearman_r": float(sr), "spearman_p": float(sp)}
    return out, merged


def main():
    ap = argparse.ArgumentParser(description="E2 -- Additive vs Threshold synthetic")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--k", type=int, default=THRESHOLD_K)
    args = ap.parse_args()

    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)

    # Paper Explainable_SER (Table I) uses the SEAB-Multiple mapping only,
    # which we call ``multi`` here. Reviewer R1/R3 asked for a genuinely
    # non-additive control that WAF's 2-way interaction terms should be
    # able to detect (while single-attribute metrics SP/EO/FPR should
    # fail).
    #
    # We use an ``xor`` regime: each emotion is triggered when *exactly one*
    # of a specific attribute pair is privileged. Under this rule the pair
    # has zero main-effect signal (E[x_a * I(bias)] = 0 by symmetry) so
    # SP/EO/FPR/main-only-WAF all fail; only the pairwise-interaction terms
    # of WAF can detect the pattern. The 6 emotions collectively cover all
    # 6 C(4,2) pairs.
    regimes = {
        "multi": ("multi", mappings["multi"]),
        "xor":   ("xor",   mappings["xor"]),
    }
    variants = {"full": True, "main_only": False}

    rows = []
    for regime_name, (regime, mapping) in regimes.items():
        # Generate synthetic data once per regime.
        ds_path = os.path.join(args.output_dir, f"seab_{regime_name}.csv")
        print(f"\n=== Generating {regime_name} regime ===")
        set_seed()
        dataset = generate_dataset(mapping, ds_path, regime=regime, k=args.k)

        # Compute MI once per regime.
        mi_df = _attribute_error_mi(dataset, mapping)

        # Traditional metrics once per regime.
        trad_df = _trad_metrics(dataset)

        for variant_name, include_int in variants.items():
            print(f"--- WAF variant: {variant_name}  regime={regime_name} ---")
            waf_df, fit_summary = _waf_scores_linear(
                dataset, include_interactions=include_int
            )
            corr, merged = _merge_and_correlate(mi_df, waf_df, trad_df)
            merged_out = os.path.join(
                args.output_dir, f"merged_{regime_name}_{variant_name}.csv"
            )
            merged.to_csv(merged_out, index=False)

            # Save the full parameter table (including interactions) for
            # inspection.
            tbl_out = os.path.join(
                args.output_dir, f"waf_params_{regime_name}_{variant_name}.csv"
            )
            fit_summary["table"].to_csv(tbl_out, index=False)

            for metric_name, stats in corr.items():
                rows.append({
                    "regime": regime_name,
                    "waf_variant": variant_name,
                    "metric_vs_mi": metric_name,
                    "pearson_r": stats["pearson_r"],
                    "pearson_p": stats["pearson_p"],
                    "spearman_r": stats["spearman_r"],
                    "spearman_p": stats["spearman_p"],
                    "mean_r2": fit_summary["mean_r2"],
                })

    summary = pd.DataFrame(rows)
    # Wide-format table mirroring Table 5.1/5.2 of the paper
    p_pivot = summary.pivot_table(
        index=["regime", "waf_variant"], columns="metric_vs_mi", values="pearson_r"
    ).add_prefix("Pearson_")
    s_pivot = summary.pivot_table(
        index=["regime", "waf_variant"], columns="metric_vs_mi", values="spearman_r"
    ).add_prefix("Spearman_")
    r2_col = summary.groupby(["regime", "waf_variant"])["mean_r2"].first()
    pivot = pd.concat([p_pivot, s_pivot], axis=1)
    pivot["mean_r2"] = r2_col
    pivot_path = os.path.join(args.output_dir, "e2_correlation_summary.csv")
    pivot.to_csv(pivot_path)
    # Also save the raw long-format
    summary.to_csv(os.path.join(args.output_dir, "e2_correlation_long.csv"), index=False)
    print("\n=== E2 correlation summary (r of MI vs. metric; * = p>=0.05) ===")
    print(pivot)
    print(f"\nSaved: {pivot_path}")


if __name__ == "__main__":
    main()
