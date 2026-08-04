"""KernelSHAP and permutation importance for demographic-attribute
importance in the WAF surrogate.

Reads the cached WAF dataset produced by ``run_e1.py`` (same
demographic binarisation and top-``|PC1|``-loading speech dimensions
used by the main WAF fit) and applies two external interpretability
methods to a Ridge surrogate that predicts the same per-class BCE
target:

* KernelSHAP — mean absolute Shapley values per demographic attribute.
* Permutation importance — mean decrease in surrogate prediction
  accuracy under attribute value permutation.

Both rankings are compared with WAF's main-effect coefficient ranking
via top-1 agreement and Kendall's tau.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from datasets import load_from_disk
from scipy.stats import kendalltau
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

from databases.constants import label2id, labels
from fairness.waf.linear_waf import (bce_per_class_np, build_feature_matrix,
                                     fit_linear_waf, pairwise_interactions)
from utils import set_seed


DEM = ["AgeGroup", "Ethnicity", "Sex", "Race"]


def _fit_ridge_per_emotion(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0):
    return [Ridge(alpha=alpha).fit(X, Y[:, c]) for c in range(Y.shape[1])]


def kernel_shap_ranking(ridges, X, feature_names, demog_cols,
                        background_size=100, sample_size=200):
    """Compute mean-|SHAP| per demographic feature, per emotion class."""
    import shap
    rng = np.random.RandomState(42)
    idx_bg = rng.choice(len(X), min(background_size, len(X)), replace=False)
    idx_s = rng.choice(len(X), min(sample_size, len(X)), replace=False)
    X_bg = X[idx_bg]
    X_s = X[idx_s]
    demog_idx = [feature_names.index(d) for d in demog_cols]
    rows = []
    for c, r in enumerate(ridges):
        explainer = shap.KernelExplainer(
            r.predict, shap.sample(X_bg, min(50, len(X_bg)))
        )
        sv = explainer.shap_values(X_s, nsamples=100, silent=True)
        mean_abs = np.abs(sv).mean(axis=0)
        for j, d in enumerate(demog_cols):
            rows.append({"emotion": labels[c], "attribute": d,
                         "mean_abs_shap": float(mean_abs[demog_idx[j]])})
    return pd.DataFrame(rows)


def perm_importance_ranking(ridges, X, Y, feature_names, demog_cols,
                            n_repeats=20):
    rows = []
    for c, r in enumerate(ridges):
        pi = permutation_importance(r, X, Y[:, c], n_repeats=n_repeats,
                                    random_state=42, n_jobs=1)
        for d in demog_cols:
            j = feature_names.index(d)
            rows.append({
                "emotion": labels[c],
                "attribute": d,
                "perm_importance_mean": float(pi.importances_mean[j]),
                "perm_importance_std": float(pi.importances_std[j]),
            })
    return pd.DataFrame(rows)


def waf_main_effect_ranking(X_df, Y, demog_cols, speech_cols, interaction_cols):
    fit = fit_linear_waf(
        X=X_df, Y=Y, emotion_labels=labels,
        demographic_cols=demog_cols,
        interaction_cols=interaction_cols,
        speech_cols=speech_cols,
    )
    tbl = fit.table_iii()
    main = tbl[tbl["feature_type"] == "main"].copy()
    return main.rename(columns={"feature": "attribute", "coef": "waf"})[
        ["emotion", "attribute", "waf"]
    ]


def _rank_agreement(a: pd.Series, b: pd.Series) -> dict:
    if len(a) < 2 or len(b) < 2:
        return {"top1_match": np.nan, "kendall_tau": np.nan, "tau_p": np.nan}
    top_a, top_b = a.abs().idxmax(), b.abs().idxmax()
    tau, tp = kendalltau(a.abs().values, b.abs().values)
    return {"top1_match": int(top_a == top_b),
            "kendall_tau": float(tau), "tau_p": float(tp)}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, choices=["hubert", "wavlm"])
    ap.add_argument("--waf_dataset_dir", required=True,
                    help="Path to the cached WAF dataset "
                         "(e.g. content/waf_training/data/hubert/waf_dataset)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--skip_shap", action="store_true",
                    help="Skip KernelSHAP (still runs permutation).")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed()

    ds = load_from_disk(args.waf_dataset_dir)
    df = ds.to_pandas() if hasattr(ds, "to_pandas") else pd.DataFrame(
        {c: ds[c] for c in ds.column_names}
    )
    print(f"loaded WAF dataset: N={len(df)}")

    speech_cols = [c for c in df.columns if c.startswith("embedding_")]
    demog_cols = list(DEM)
    print(f"speech pool = {len(speech_cols)}, demog = {demog_cols}")

    X_df = build_feature_matrix(
        df, demographic_cols=demog_cols, speech_cols=speech_cols,
        include_interactions=True,
    )
    interaction_cols = list(pairwise_interactions(df, demog_cols).columns)

    y_idx = np.array([label2id[e] for e in df["emotion"]])
    y_onehot = np.eye(len(labels))[y_idx]
    scores = np.stack([np.asarray(s, dtype=np.float32) for s in df["scores"]])
    Y = bce_per_class_np(scores, y_onehot)

    os.makedirs(args.out_dir, exist_ok=True)

    # WAF main-effect coefficients as reference ranking
    waf_rank = waf_main_effect_ranking(
        X_df, Y, demog_cols, speech_cols, interaction_cols
    )
    waf_path = os.path.join(args.out_dir, f"{args.model_name}_waf_ranking.csv")
    waf_rank.to_csv(waf_path, index=False)
    print(f"saved {waf_path}")

    # ridge surrogate for SHAP / permutation
    X_np = X_df.values.astype(float)
    feature_names = list(X_df.columns)
    ridges = _fit_ridge_per_emotion(X_np, Y, alpha=1.0)

    perm = perm_importance_ranking(
        ridges, X_np, Y, feature_names, demog_cols, n_repeats=20
    )
    perm_path = os.path.join(args.out_dir, f"{args.model_name}_perm_importance.csv")
    perm.to_csv(perm_path, index=False)
    print(f"saved {perm_path}")

    shap_df = None
    if not args.skip_shap:
        shap_df = kernel_shap_ranking(
            ridges, X_np, feature_names, demog_cols,
            background_size=100, sample_size=200,
        )
        shap_path = os.path.join(args.out_dir, f"{args.model_name}_shap.csv")
        shap_df.to_csv(shap_path, index=False)
        print(f"saved {shap_path}")

    # rank agreement per emotion
    agree_rows = []
    for emo in labels:
        w = waf_rank[waf_rank["emotion"] == emo].set_index("attribute")["waf"]
        p = perm[perm["emotion"] == emo].set_index("attribute")["perm_importance_mean"]
        pw = _rank_agreement(w, p.reindex(w.index))
        row = {"emotion": emo, **{f"perm_{k}": v for k, v in pw.items()}}
        if shap_df is not None:
            s = shap_df[shap_df["emotion"] == emo].set_index("attribute")["mean_abs_shap"]
            sw = _rank_agreement(w, s.reindex(w.index))
            row.update({f"shap_{k}": v for k, v in sw.items()})
        agree_rows.append(row)
    agree_df = pd.DataFrame(agree_rows)
    agree_path = os.path.join(args.out_dir, f"{args.model_name}_rank_agreement.csv")
    agree_df.to_csv(agree_path, index=False)
    print(f"saved {agree_path}")
    print("\n--- rank agreement ---")
    print(agree_df.to_string(index=False))


if __name__ == "__main__":
    main()
