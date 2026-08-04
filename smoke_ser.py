"""Quick smoke test: train HuBERT-base on a tiny subset (60 samples, 5 steps).

Purpose: catch obvious bugs (import errors, tensor shape issues, HF trainer
config problems) in ~2 minutes before committing to the 3-6h large-model run.
Uses base checkpoints to keep memory low.
"""
from __future__ import annotations

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))

import torch
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import AutoConfig, TrainingArguments, Wav2Vec2FeatureExtractor

from databases.cremad import extract_train_test_sets, load_db
from databases.constants import labels
from ser_models.evaluator import evaluate
from ser_models.hubert import HubertForSpeechClassification
from ser_models.speech_processing import preprocess_function
from ser_models.trainer import ClassificationTrainer, DataCollatorWithPadding
from utils import get_device, set_seed


def main():
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    db = load_db(os.path.join(REPO, "db/CREMA-D"))
    train_df, test_df = extract_train_test_sets(db)
    # Tiny subset
    train_df = train_df.head(60).reset_index(drop=True)
    test_df = test_df.head(20).reset_index(drop=True)
    train_ds = Dataset.from_pandas(train_df).map(
        preprocess_function, batch_size=20, batched=True, num_proc=2
    )
    test_ds = Dataset.from_pandas(test_df).map(
        preprocess_function, batch_size=20, batched=True, num_proc=2
    )

    model_path = "facebook/hubert-large-ll60k"   # already cached under hf_cache/
    config = AutoConfig.from_pretrained(
        model_path, num_labels=len(labels),
        label2id={l: i for i, l in enumerate(labels)},
        id2label={i: l for i, l in enumerate(labels)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, "pooling_mode", "mean")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = HubertForSpeechClassification.from_pretrained(model_path, config=config)
    model.to(device)
    model.freeze_feature_extractor()

    with tempfile.TemporaryDirectory() as td:
        args = TrainingArguments(
            output_dir=td,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            max_steps=5,
            save_steps=100,
            eval_steps=100,
            logging_steps=1,
            learning_rate=1e-4,
            remove_unused_columns=False,
            fp16=True,
            report_to="none",
        )
        collator = DataCollatorWithPadding(processor=processor, padding=True)
        trainer = ClassificationTrainer(
            model=model, data_collator=collator, args=args,
            train_dataset=train_ds, eval_dataset=test_ds,
            processing_class=processor,
        )
        trainer.train()
        print("Train OK. Evaluating...")

    model.eval()
    result = evaluate(model, processor, test_ds)
    print("Eval columns:", result.column_names)
    print("Predicted (5):", result["predicted"][:5])
    print("Scores[0]:", result["scores"][0])
    print("SMOKE OK")


if __name__ == "__main__":
    main()
