"""Re-run only the E4 WAF sex-only step after fixing the label-map bug."""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = os.path.join(REPO, "src") + ":" + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    for m in ["hubert", "wavlm"]:
        ckpt = os.path.join(REPO, "content/ser_training_iemocap/models", m, "final")
        res = os.path.join(REPO, "content/ser_training_iemocap/data", m)
        table = os.path.join(REPO, "content/waf_training/tables",
                             f"iemocap_{m}_tableIII_sex.csv")
        cmd = [
            sys.executable, "-m", "fairness.waf.run_linear_waf",
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
            "--dataset", "iemocap",
        ]
        print(f"\n=== E4 WAF {m} ===", flush=True)
        subprocess.run(cmd, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
