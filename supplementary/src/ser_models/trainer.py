from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from transformers import (AutoConfig, EvalPrediction, Trainer,
                          TrainingArguments, Wav2Vec2FeatureExtractor)

from databases.constants import labels


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


class ClassificationTrainer(Trainer):

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=1,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        label_dtype = torch.long if isinstance(labels[0], int) else torch.float

        batch = self.processor(
            input_values,
            sampling_rate=self.processor.sampling_rate,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=label_dtype)

        return batch


def get_trainer(model, processor, output_dir, train_dataset, test_dataset):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=1.0,
        save_steps=10,
        eval_steps=10,
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=1e-5,
        save_total_limit=2,
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorWithPadding(processor=processor, padding=True)

    trainer = ClassificationTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=processor,
    )

    return trainer


def get_model(model_name_or_path, model_cls):
    pooling_mode = "mean"
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        label2id={label: i for i, label in enumerate(labels)},
        id2label={i: label for i, label in enumerate(labels)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, "pooling_mode", pooling_mode)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    target_sampling_rate = processor.sampling_rate
    print(f"The target sampling rate: {target_sampling_rate}")

    model = model_cls.from_pretrained(
        model_name_or_path,
        config=config,
    )

    return config, processor, model
