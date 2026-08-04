"""Full pipeline: E1 + E5 on CREMA-D, then IEMOCAP SER training + E4.

Runs sequentially, logs to logs/pipeline_full.log. Safe to nohup.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = os.path.join(REPO, "src") + ":" + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def run(cmd, name):
    print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env=os.environ.copy())
    dt = (time.time() - t0) / 60
    status = "OK" if r.returncode == 0 else f"FAIL rc={r.returncode}"
    print(f"[{name}] {status}  ({dt:.1f} min)", flush=True)
    return r.returncode == 0


def main():
    py = sys.executable

    # --- E1 on both models (Table III) ---
    run([py, "run_e1.py", "--models", "hubert", "wavlm"], "E1 CREMA-D")

    # --- E5 (a,b,c,d) on both models ---
    run([py, "run_e5.py", "--models", "hubert", "wavlm"], "E5 CREMA-D")

    # --- E4: train HuBERT & WavLM on IEMOCAP (3 epochs each) ---
    run([py, "train_ser.py", "--dataset", "iemocap", "--models", "hubert", "wavlm"],
        "E4 SER training on IEMOCAP")

    # --- E4: WAF sex-only on IEMOCAP ---
    print(f"\n{'='*60}\nE4 WAF sex-only on IEMOCAP\n{'='*60}", flush=True)
    for m in ["hubert", "wavlm"]:
        ckpt = os.path.join(REPO, "content/ser_training_iemocap/models", m, "final")
        res = os.path.join(REPO, "content/ser_training_iemocap/data", m)
        table = os.path.join(REPO, "content/waf_training/tables",
                             f"iemocap_{m}_tableIII_sex.csv")
        run([py, "-m", "fairness.waf.run_linear_waf",
             "--model_name", m,
             "--ser_model_ckpt", ckpt,
             "--ser_result_dataset", res,
             "--waf_config", os.path.join(REPO, "src/fairness/example/waf_config.yaml"),
             "--waf_model_outdir", os.path.join(REPO, "content/waf_training/models",
                                                f"iemocap_{m}"),
             "--waf_dataset_outdir", os.path.join(REPO, "content/waf_training/data",
                                                  f"iemocap_{m}"),
             "--table_out", table,
             "--xds_override", "Sex",
             "--cluster_col", "SpeakerID",
             ], f"E4 WAF {m}")

    print("\nAll experiments finished.", flush=True)


if __name__ == "__main__":
    main()
