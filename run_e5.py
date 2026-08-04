"""Run E5 (trajectory + SHAP/permutation + cluster-robust SE) on
HuBERT and WavLM.

Assumes ``run_e1.py`` has already produced the cached WAF datasets at
``content/waf_training/data/{model}/waf_dataset``.
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
    ap.add_argument("--skip_shap", action="store_true",
                    help="Skip KernelSHAP for a faster pass "
                         "(permutation + trajectory + cluster-robust are cheap).")
    args = ap.parse_args()

    for model in args.models:
        waf_ds = os.path.join(
            REPO, "content/waf_training/data", model, "waf_dataset"
        )
        ser_ckpt = os.path.join(REPO, "content/ser_training/models", model, "final")
        ser_ds = os.path.join(REPO, "content/ser_training/data", model)
        cfg = os.path.join(REPO, "src/fairness/example/waf_config.yaml")
        out_dir = os.path.join(REPO, "content/e5", model)
        os.makedirs(out_dir, exist_ok=True)

        # coefficient trajectory
        print(f"\n### trajectory: {model} ###", flush=True)
        subprocess.run([
            sys.executable, "-m", "fairness.analysis.k_trajectory",
            "--model_name", model,
            "--waf_dataset_dir", waf_ds,
            "--out_dir", os.path.join(out_dir, "trajectory"),
        ], check=True, env=os.environ.copy())

        # SHAP + permutation
        print(f"\n### SHAP + permutation: {model} ###", flush=True)
        cmd = [
            sys.executable, "-m", "fairness.analysis.shap_and_permutation",
            "--model_name", model,
            "--waf_dataset_dir", waf_ds,
            "--out_dir", os.path.join(out_dir, "shap_perm"),
        ]
        if args.skip_shap:
            cmd.append("--skip_shap")
        subprocess.run(cmd, check=True, env=os.environ.copy())

        # cluster-robust SE (via E1 CLI with --cluster_col)
        print(f"\n### cluster-robust SE: {model} ###", flush=True)
        subprocess.run([
            sys.executable, "-m", "fairness.waf.run_linear_waf",
            "--model_name", model,
            "--ser_model_ckpt", ser_ckpt,
            "--ser_result_dataset", ser_ds,
            "--waf_config", cfg,
            "--table_out", os.path.join(out_dir, f"{model}_tableIII_cluster.csv"),
            "--cluster_col", "ActorID",
        ], check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
