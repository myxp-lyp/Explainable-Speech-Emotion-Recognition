

import os
import subprocess

os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.path.join(os.getcwd(), "src")

CREMA_D_ROOT = "db/CREMA-D/"

models = {
    "hubert": {
        "model_ckpt_dir": "content/ser_training/models/hubert",
        "result_outdir": "content/ser_training/data/hubert",
        "waf_model_outdir": "content/waf_training/models/hubert",
        "waf_data_outdir": "content/waf_training/models/hubert",
        "trad_metrics_outdir": "content/trad_metrics/hubert",
    },
    "wavlm": {
        "model_ckpt_dir": "content/ser_training/models/wavlm",
        "result_outdir": "content/ser_training/data/wavlm",
        "waf_model_outdir": "content/waf_training/models/wavlm",
        "waf_data_outdir": "content/waf_training/models/wavlm",
         "trad_metrics_outdir": "content/trad_metrics/wavlm",
    },
}

for model_name, attrs in models.items():

    subprocess.run(
        [
            "python",
            "-m",
            "ser_models.main",
            "--model_name",
            model_name,
            "--cremad_root",
            CREMA_D_ROOT,
            "--model_output_dir",
            attrs['model_ckpt_dir'],
            "--result_output_dir",
            attrs['result_outdir'],
        ],
        check=True,
    )

    best_model_paths = os.listdir(attrs['model_ckpt_dir'])

    # select one with highest number in name
    best_checkpoint = max(
        best_model_paths,
        key=lambda x: int(x.split('-')[1]) if '-' in x else 0,
    )

    best_model_path = os.path.join(attrs['model_ckpt_dir'], best_checkpoint)
    print(f"Best model for {model_name}: {best_model_path}")
    

    subprocess.run(
        [
            "python",
            "-m",
            "fairness.waf.main",
            "--model_name",
            model_name,
            "--ser_model_ckpt",
           best_model_path,
            "--ser_result_dataset",
            attrs['result_outdir'],
            "--waf_config",
            "./src/fairness/example/waf_config.yaml",
            "--waf_model_outdir",
            attrs['waf_model_outdir'],
            "--waf_dataset_outdir",
            attrs['waf_data_outdir'],
        ],
        check=True,
    )

    subprocess.run(
        [
            "python",
            "-m",
            "fairness.trad_metrics.main",
            "--ser_result_dataset",
            attrs['result_outdir'],
            "--output_csv",
            attrs['trad_metrics_outdir'],
        ],
        check=True,
    )
    print(f"Finished training and evaluating model: {model_name}")
    print(f"Results saved to: {attrs['result_outdir']}")
    print(f"WAF model saved to: {attrs['waf_model_outdir']}")
    print(f"WAF dataset saved to: {attrs['waf_data_outdir']}")
    print(f"Traditional metrics saved to: {attrs['trad_metrics_outdir']}")
    print("-" * 50)