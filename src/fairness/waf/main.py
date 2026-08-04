import argparse
import os

import datasets
from IPython.display import display
from transformers import Wav2Vec2FeatureExtractor
from yaml import safe_load

from databases.constants import labels
from fairness.waf.config import WAFConfig
from fairness.waf.dataset import collator_fn, create_waf_dataset
from fairness.waf.evaluator import evaluate_waf
from fairness.waf.mean_regressor import evaluate_mean_regressor
from fairness.waf.model import WAFModel
from fairness.waf.parser import parse_waf_metrics
from fairness.waf.trainer import WAFTrainer
from ser_models.hubert import HubertForSpeechClassification
from ser_models.wavlm import WavLMForSpeechClassification
from utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train WAF models from SER models")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to train (e.g., 'hubert' or 'wavlm')",
    )
    parser.add_argument(
        "--ser_model_ckpt", type=str, required=True, help="Path to SER model checkpoint"
    )
    parser.add_argument(
        "--ser_result_dataset",
        type=str,
        required=True,
        help="Path to SER model dataset",
    )
    parser.add_argument("--waf_config", type=str, required=True)
    parser.add_argument(
        "--waf_model_outdir",
        type=str,
        required=False,
        help="Path to WAF model output dir",
    )
    parser.add_argument(
        "--waf_dataset_outdir",
        type=str,
        required=False,
        help="Path to WAF dataset output dir",
    )

    return parser.parse_args()


def load_config_from_yaml(path: str) -> WAFConfig:
    with open(path, "r") as f:
        data = safe_load(f)
    return WAFConfig(**data)


supported_models = ["hubert", "wavlm"]


def main(
    model_name,
    ser_model_ckpt,
    ser_result_dataset,
    config: WAFConfig,
):

    set_seed()

    if model_name == "hubert":
        ser_model = HubertForSpeechClassification.from_pretrained(ser_model_ckpt)
    elif model_name == "wavlm":
        ser_model = WavLMForSpeechClassification.from_pretrained(ser_model_ckpt)
    elif model_name == "synthetic":
        ser_model = None
        config.use_embeddings = False
    else:
        raise ValueError(
            f"Unsupported model name: {model_name}. Supported models are: {list(supported_models.keys())}"
        )

    # ``ser_model_ckpt`` is a directory; from_pretrained needs the directory,
    # not the preprocessor_config.json path.
    processor = Wav2Vec2FeatureExtractor.from_pretrained(ser_model_ckpt)
    print(f"Loading results dataset from {ser_result_dataset}")
    dataset = datasets.load_from_disk(ser_result_dataset)
    device = get_device()
    print("\n Creating WAF dataset from SER outputs \n")

    dataset, config = create_waf_dataset(
        ser_dataset=dataset,
        config=config,
        model=ser_model.to(device),
        processor=processor,
    )

    print("\n Training WAF model \n")
    input_dim = len(config.xds) + len(config.xcs)
    output_dim = len(labels)
    waf_model = WAFModel(input_dim, output_dim).to(device)

    trainer = WAFTrainer(waf_model, config, dataset, collator_fn)
    trainer.train()

    # print("\n Evaluating WAF model \n")
    mse, r2 = evaluate_waf(
        model=waf_model,
        dataset=dataset,
    )
    print(f"[{model_name}] Best MSE: {mse:.4f}, Best R2: {r2:.4f}")

    # Evalute baseline - mean regressor
    mse_dummy, r2_dummy = evaluate_mean_regressor(dataset)
    print(f"[Dummy mean regressor] Best MSE: {mse_dummy:.4f}, Best R2: {r2_dummy:.4f}")

    # parse waf metrics
    print(f"WAF metrics for provided model {model_name}")
    metrics_df = parse_waf_metrics(waf_model, config)
    display(metrics_df)


if __name__ == "__main__":
    args = parse_args()
    # load config from yml
    model_name = args.model_name
    ser_model_ckpt = args.ser_model_ckpt
    ser_results_dataset = args.ser_result_dataset
    waf_config_path = args.waf_config

    waf_config = load_config_from_yaml(waf_config_path)

    if args.waf_model_outdir:
        waf_config.model_cache_dir = args.waf_model_outdir

    if args.waf_dataset_outdir:
        waf_config.data_cache_dir = args.waf_dataset_outdir

    main(
        model_name=model_name,
        ser_model_ckpt=ser_model_ckpt,
        ser_result_dataset=ser_results_dataset,
        config=waf_config,
    )
