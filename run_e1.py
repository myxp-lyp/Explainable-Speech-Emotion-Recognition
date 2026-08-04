"""Runner for E1 on both HuBERT and WavLM.

Fits a linear (OLS) WAF with:
  * 4 demographic main effects
  * 6 pairwise interaction terms (Age×Sex, Age×Race, Age×Eth,
    Sex×Race, Sex×Eth, Race×Eth)
  * PCA-100 speech features (from top-|PC1|-loading dims, matching the
    existing pipeline)

Reports Table III per model with coef / std_err / Wald p-value /
Benjamini-Hochberg-adjusted significance for main and interaction terms.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["PYTHONPATH"] = os.path.join(REPO, "src") + ":" + os.environ.get("PYTHONPATH", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["hubert", "wavlm"])
    ap.add_argument("--main_effects_only", action="store_true",
                    help="Ablation: drop the 6 pairwise interaction terms.")
    ap.add_argument("--cluster_col", default=None,
                    help="Column to use for cluster-robust SE (e.g. ActorID).")
    args = ap.parse_args()

    tables_dir = os.path.join(REPO, "content", "waf_training", "tables")
    os.makedirs(tables_dir, exist_ok=True)

    for model in args.models:
        ckpt = os.path.join(REPO, "content/ser_training/models", model, "final")
        result_ds = os.path.join(REPO, "content/ser_training/data", model)
        cfg = os.path.join(REPO, "src/fairness/example/waf_config.yaml")
        model_out = os.path.join(REPO, "content/waf_training/models", model)
        data_out = os.path.join(REPO, "content/waf_training/data", model)
        table_out = os.path.join(
            tables_dir,
            f"{model}_tableIII{'_mainonly' if args.main_effects_only else ''}.csv",
        )
        cmd = [
            sys.executable, "-m", "fairness.waf.run_linear_waf",
            "--model_name", model,
            "--ser_model_ckpt", ckpt,
            "--ser_result_dataset", result_ds,
            "--waf_config", cfg,
            "--waf_model_outdir", model_out,
            "--waf_dataset_outdir", data_out,
            "--table_out", table_out,
        ]
        if args.main_effects_only:
            cmd.append("--main_effects_only")
        if args.cluster_col:
            cmd += ["--cluster_col", args.cluster_col]

        print(f"\n=== E1 on {model} -> {table_out} ===", flush=True)
        subprocess.run(cmd, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
