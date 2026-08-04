"""E1 runner: linear (ridge) WAF with two-way demographic interactions.

Usage
-----
python -m fairness.waf.run_linear_waf \\
    --model_name hubert \\
    --ser_model_ckpt content/ser_training/models/hubert/final \\
    --ser_result_dataset content/ser_training/data/hubert \\
    --waf_config src/fairness/example/waf_config.yaml \\
    --waf_model_outdir content/waf_training/models/hubert \\
    --waf_dataset_outdir content/waf_training/data/hubert \\
    --table_out content/waf_training/tables/hubert_tableIII.csv
"""
from __future__ import annotations

import argparse
import os
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import Wav2Vec2FeatureExtractor
from yaml import safe_load

from databases.constants import labels
from fairness.constants import privileged_groups
from fairness.waf.config import WAFConfig
from fairness.waf.dataset import create_waf_dataset
from fairness.waf.linear_waf import (bce_per_class_np, build_feature_matrix,
                                     fit_linear_waf, pairwise_interactions)
from ser_models.hubert import HubertForSpeechClassification
from ser_models.wavlm import WavLMForSpeechClassification
from utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="E1 -- Linear WAF with interactions")
    p.add_argument("--model_name", required=True, choices=["hubert", "wavlm"])
    p.add_argument("--ser_model_ckpt", required=True)
    p.add_argument("--ser_result_dataset", required=True)
    p.add_argument("--waf_config", required=True)
    p.add_argument("--waf_model_outdir", required=False)
    p.add_argument("--waf_dataset_outdir", required=False)
    p.add_argument("--table_out", required=True, help="CSV output for Table III")
    p.add_argument("--main_effects_only", action="store_true",
                   help="Ablation: drop interaction terms")
    p.add_argument("--xds_override", default=None,
                   help="Comma-separated list to override config.xds "
                        "(e.g. 'Sex' for IEMOCAP)")
    p.add_argument("--ridge_alpha", type=float, default=0.0,
                   help="L2 penalty; 0 disables ridge (pure OLS)")
    p.add_argument("--cluster_col", default=None,
                   help="Column name in SER dataset to use as cluster ID "
                        "(e.g. 'ActorID' for cluster-robust SE)")
    p.add_argument("--dataset", default="cremad",
                   choices=["cremad", "iemocap"],
                   help="Selects the label vocabulary for BCE-per-class.")
    return p.parse_args()


def load_config_from_yaml(path: str) -> WAFConfig:
    with open(path, "r") as f:
        data = safe_load(f)
    return WAFConfig(**data)


def build_waf_dataset(model_name, ser_model_ckpt, ser_result_dataset, config,
                      label2id_map=None):
    """Reuse the existing WAF dataset builder to get demographic binarisation
    and PCA-top-k speech features consistent with the rest of the pipeline."""
    set_seed()

    if model_name == "hubert":
        ser_model = HubertForSpeechClassification.from_pretrained(ser_model_ckpt)
    elif model_name == "wavlm":
        ser_model = WavLMForSpeechClassification.from_pretrained(ser_model_ckpt)
    else:
        raise ValueError(model_name)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(ser_model_ckpt)
    device = get_device()
    ser_model = ser_model.to(device).eval()

    ser_dataset = load_from_disk(ser_result_dataset)
    dataset, config = create_waf_dataset(
        ser_dataset=ser_dataset,
        config=config,
        model=ser_model,
        processor=processor,
        label2id_map=label2id_map,
    )
    return dataset, config


def main():
    args = parse_args()
    set_seed()

    # Select label vocabulary
    if args.dataset == "iemocap":
        from databases.iemocap import (IEMOCAP_LABEL2ID as _label2id,
                                       IEMOCAP_LABELS as _labels)
    else:
        from databases.constants import label2id as _label2id
        _labels = labels

    config = load_config_from_yaml(args.waf_config)
    if args.xds_override:
        config.xds = [x.strip() for x in args.xds_override.split(",") if x.strip()]
    if args.waf_model_outdir:
        config.model_cache_dir = args.waf_model_outdir
    if args.waf_dataset_outdir:
        config.data_cache_dir = args.waf_dataset_outdir

    dataset, config = build_waf_dataset(
        args.model_name, args.ser_model_ckpt, args.ser_result_dataset, config,
        label2id_map=_label2id,
    )

    df = dataset.to_pandas() if hasattr(dataset, "to_pandas") else pd.DataFrame(
        {k: dataset[k] for k in dataset.column_names}
    )

    # Feature construction
    demographic_cols = list(config.xds)
    speech_cols = list(config.xcs)

    # Build design matrix
    include_int = (not args.main_effects_only) and (len(demographic_cols) >= 2)
    X = build_feature_matrix(
        df,
        demographic_cols=demographic_cols,
        speech_cols=speech_cols,
        include_interactions=include_int,
    )
    interaction_cols = (
        list(pairwise_interactions(df, demographic_cols).columns) if include_int else []
    )

    # Target: BCE per class (must match ``fairness.waf.dataset.bce_per_class``)
    y_true_idx = np.array([_label2id[e] for e in df["emotion"]])
    y_onehot = np.eye(len(_labels))[y_true_idx]
    scores = np.stack([np.asarray(s, dtype=np.float32) for s in df["scores"]], axis=0)
    Y = bce_per_class_np(scores, y_onehot)

    # Cluster IDs for robust SE
    cluster_groups = None
    if args.cluster_col is not None:
        assert args.cluster_col in df.columns, \
            f"cluster_col {args.cluster_col} not found in dataset columns {df.columns.tolist()}"
        cluster_groups = df[args.cluster_col].values

    print(f"[E1] N={len(df)}, P={X.shape[1]} "
          f"(main={len(demographic_cols)}, interaction={len(interaction_cols)}, "
          f"speech={len(speech_cols)})")

    fit = fit_linear_waf(
        X=X,
        Y=Y,
        emotion_labels=_labels,
        demographic_cols=demographic_cols,
        interaction_cols=interaction_cols,
        speech_cols=speech_cols,
        l2_alpha=args.ridge_alpha if args.ridge_alpha > 0 else None,
        cluster_groups=cluster_groups,
    )

    table = fit.table_iii()
    os.makedirs(os.path.dirname(args.table_out) or ".", exist_ok=True)
    table.to_csv(args.table_out, index=False)
    print(f"[E1] Wrote Table III to {args.table_out}")

    # Also print a condensed summary to stdout
    main_eff = fit.main_effects().sort_values(["emotion", "feature"])
    print("\n--- Main effects ---")
    print(main_eff.to_string(index=False))

    if include_int:
        sig = fit.significant_interactions(alpha=0.05)
        print(f"\n--- Significant interactions (BH-adjusted, alpha=0.05, N={len(sig)}) ---")
        print(sig.to_string(index=False) if len(sig) else "(none)")


if __name__ == "__main__":
    main()
