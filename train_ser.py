"""Train HuBERT-large and WavLM-large on CREMA-D or IEMOCAP.

Runs sequentially (T4 has only 15GB VRAM). Each model:
  1. finetune on the specified dataset
  2. save best model + processor to <ckpt_dir>/final
  3. evaluate on test set, save {predicted, scores, speech} to <result_dir>

For CREMA-D (default): outputs go under content/ser_training/{models,data}/<model>/
For IEMOCAP: outputs go under content/ser_training_iemocap/{models,data}/<model>/
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = os.path.join(REPO, "src") + ":" + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")

CREMA_D_ROOT = os.path.join(REPO, "db", "CREMA-D")
IEMOCAP_ROOT = os.path.join(REPO, "db", "IEMOCAP_full_release")
MODELS = ["hubert", "wavlm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--dataset", default="cremad", choices=["cremad", "iemocap"])
    args = ap.parse_args()

    if args.dataset == "cremad":
        base_ckpt = os.path.join(REPO, "content/ser_training/models")
        base_res = os.path.join(REPO, "content/ser_training/data")
    else:
        base_ckpt = os.path.join(REPO, "content/ser_training_iemocap/models")
        base_res = os.path.join(REPO, "content/ser_training_iemocap/data")

    for model_name in args.models:
        ckpt_dir = os.path.join(base_ckpt, model_name)
        res_dir = os.path.join(base_res, model_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)

        print(f"\n==============================================")
        print(f" Training {model_name}  ({args.dataset})")
        print(f"   ckpt -> {ckpt_dir}")
        print(f"   result -> {res_dir}")
        print(f"==============================================\n", flush=True)
        t0 = time.time()

        cmd = [
            sys.executable, "-m", "ser_models.main",
            "--model_name", model_name,
            "--dataset", args.dataset,
            "--model_output_dir", ckpt_dir,
            "--result_output_dir", res_dir,
        ]
        if args.dataset == "cremad":
            cmd += ["--cremad_root", CREMA_D_ROOT]
        else:
            cmd += ["--iemocap_root", IEMOCAP_ROOT]

        subprocess.run(cmd, check=True, env=os.environ.copy())

        dt = (time.time() - t0) / 60
        print(f"\n[{model_name}] done in {dt:.1f} min")


if __name__ == "__main__":
    main()
