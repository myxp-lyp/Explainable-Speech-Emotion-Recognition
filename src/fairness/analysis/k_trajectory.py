"""Coefficient trajectory of the linear WAF as more speech features
are progressively added.

For a fixed WAF dataset (produced by ``run_e1.py``), refit the WAF
regression at ``k in {0, 10, 25, 50, 75, 100}`` speech features and
record the demographic coefficients at each ``k``. When ``k`` equals
the number of speech dimensions used by the main WAF fit, the
coefficients reproduce those in the main WAF result table (Table III).

Speech features are the top-``|PC1|``-loading original embedding
dimensions, in the same PC1-descending order as the main WAF fit.
"""
from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk

from databases.constants import label2id, labels
from fairness.waf.linear_waf import (bce_per_class_np, build_feature_matrix,
                                     fit_linear_waf, pairwise_interactions)
from utils import set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, choices=["hubert", "wavlm"])
    ap.add_argument("--waf_dataset_dir", required=True,
                    help="Path to the cached WAF dataset "
                         "(e.g. content/waf_training/data/hubert/waf_dataset)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[0, 10, 25, 50, 75, 100])
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed()

    ds = load_from_disk(args.waf_dataset_dir)
    df = ds.to_pandas() if hasattr(ds, "to_pandas") else pd.DataFrame(
        {c: ds[c] for c in ds.column_names}
    )
    print(f"loaded WAF dataset: N={len(df)}, cols={len(df.columns)}")

    # embedding columns are stored in PC1-descending order
    all_speech_cols = [c for c in df.columns if c.startswith("embedding_")]
    print(f"speech feature pool size = {len(all_speech_cols)} "
          f"(top-|PC1|-loading dims)")

    demog_cols = ["AgeGroup", "Ethnicity", "Sex", "Race"]

    # target: per-class BCE loss
    y_idx = np.array([label2id[e] for e in df["emotion"]])
    y_onehot = np.eye(len(labels))[y_idx]
    scores = np.stack([np.asarray(s, dtype=np.float32) for s in df["scores"]])
    Y = bce_per_class_np(scores, y_onehot)

    rows: List[pd.DataFrame] = []
    for k in args.ks:
        if k > len(all_speech_cols):
            print(f"skip k={k}, exceeds pool of {len(all_speech_cols)}")
            continue
        speech_cols = all_speech_cols[:k]
        X = build_feature_matrix(
            df,
            demographic_cols=demog_cols,
            speech_cols=speech_cols,
            include_interactions=True,
        )
        interaction_cols = list(pairwise_interactions(df, demog_cols).columns)
        fit = fit_linear_waf(
            X=X, Y=Y, emotion_labels=labels,
            demographic_cols=demog_cols,
            interaction_cols=interaction_cols,
            speech_cols=speech_cols,
        )
        tbl = fit.table_iii()
        keep = tbl[tbl["feature_type"].isin(["main", "interaction"])].copy()
        keep["k"] = k
        keep["mean_r2"] = float(np.mean([f.result.rsquared for f in fit.fits]))
        keep["mean_mse"] = float(np.mean([f.result.mse_resid for f in fit.fits]))
        rows.append(keep)
        print(f"[k={k:3d}] mean_r2 = {keep['mean_r2'].iloc[0]:.4f}, "
              f"mean_mse = {keep['mean_mse'].iloc[0]:.4f}")

    long = pd.concat(rows, ignore_index=True)
    os.makedirs(args.out_dir, exist_ok=True)
    long_csv = os.path.join(args.out_dir, f"{args.model_name}_trajectory.csv")
    long.to_csv(long_csv, index=False)
    print(f"saved {long_csv}")

    # plot: coefficient trajectories per emotion
    main_only = long[long["feature_type"] == "main"].copy()
    emos = main_only["emotion"].unique()
    attrs = main_only["feature"].unique()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    for i, emo in enumerate(emos):
        ax = axes[i]
        sub = main_only[main_only["emotion"] == emo]
        for a in attrs:
            s = sub[sub["feature"] == a].sort_values("k")
            ax.plot(s["k"], s["coef"], marker="o", label=a)
        ax.set_title(f"Emotion: {emo}")
        ax.set_xlabel("k (top-|PC1| speech dims)")
        ax.set_ylabel("WAF coef")
        ax.axhline(0, color="k", linestyle=":", linewidth=0.5)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"{args.model_name.upper()} — demographic WAF coefficient vs k")
    fig.tight_layout()
    fig1 = os.path.join(args.out_dir, f"{args.model_name}_trajectory.png")
    fig.savefig(fig1, dpi=150)
    print(f"saved {fig1}")

    # plot: mean MSE vs k
    mse_by_k = long.groupby("k")["mean_mse"].first().reset_index()
    fig2, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mse_by_k["k"], mse_by_k["mean_mse"], marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("mean MSE")
    ax.set_title(f"{args.model_name.upper()} — mean per-emotion MSE vs k")
    fig2.tight_layout()
    fig2_path = os.path.join(args.out_dir, f"{args.model_name}_mse_vs_k.png")
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path}")


if __name__ == "__main__":
    main()
