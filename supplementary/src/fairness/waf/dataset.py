import os
from copy import deepcopy
from typing import Tuple

import datasets
import torch
import torch.nn.functional as F

from databases.constants import label2id
from fairness.waf.config import WAFConfig
from fairness.waf.embeddings import add_embeddings, get_embeddings
from fairness.waf.pca import get_top_k_embedding_dims
from src.fairness.waf.utils import (binarise_protected_attribute,
                                    fit_one_hot_encoder,
                                    onehot_protected_attribute)


def bce_per_class(pred, true, factor=1):
    """
    pred - predicted scores (logits) for each class
    true - true labels (one-hot encoded)
    """
    log_probs = torch.log(pred + 1e-10)
    loss = -(true * log_probs) - factor * ((1 - true) * torch.log(1 - pred + 1e-10))
    return loss


# Add loss to test dataset
def to_waf_dataset(batch, xcs, xds, label_fn, combo_features=None):

    # encode other features
    xcs_encoded = torch.tensor([(batch[xc]) for xc in xcs], dtype=torch.float32).T
    xds_encoded = torch.tensor([(batch[xd]) for xd in xds], dtype=torch.float32).T

    input_features = torch.cat(
        [xcs_encoded, xds_encoded], dim=1
    )  # Stack along the 2nd dimension\

    if combo_features is not None:
        combo_encoded = torch.tensor(
            [(batch[cf]) for cf in combo_features], dtype=torch.float32
        ).T
        input_features = torch.cat([input_features, combo_encoded], dim=1)

    # set waf_features to combined features
    batch["waf_features"] = input_features

    # set waf_labels to loss
    y_trues = torch.tensor([label2id[item] for item in batch["emotion"]])
    y_trues = F.one_hot(y_trues, num_classes=len(label2id)).float()
    y_preds = torch.tensor(batch["scores"], dtype=torch.float32)

    batch["waf_label"] = label_fn(y_preds, y_trues)

    del batch["speech"]
    return batch


def collator_fn(batch):
    # Custom collate function to handle the batch
    return {
        "waf_features": torch.tensor([item["waf_features"] for item in batch]),
        "waf_label": torch.tensor([item["waf_label"] for item in batch]),
    }


def create_waf_dataset(
    ser_dataset, config: WAFConfig, model=None, processor=None
) -> Tuple[datasets.Dataset, WAFConfig]:
    """
    - ser_dataset: The dataset used to train the SER model containing 'emotion', 'scores' and 'speech' columns
    """

    # cache_file = os.path.join(config.data_cache_dir, f"waf_dataset")

    # if os.path.exists(cache_file) and config.use_data_cache:
    #     print(f" Found cached file for loading... ")
    #     waf_dataset = datasets.load_from_disk(cache_file)
    #     config.xcs = waf_dataset['xcs']
    #     config.xds = waf_dataset['xds']
    #     return waf_dataset, config

    print(f"Creating WAF dataset...")

    if config.use_embeddings:
        assert (
            model is not None and processor is not None
        ), "Model and Processor must be provided for embedding extraction."
        embeddings_cache_dir = (
            os.path.join(config.data_cache_dir, "embeddings")
            if config.use_embedding_cache
            else None
        )
        embeddings = get_embeddings(
            ser_dataset["speech"],
            ser_dataset["path"],
            processor,
            model,
            cache_dir=embeddings_cache_dir,
        )

        # Get top k embedding dimensions
        pca_cache_file = (
            os.path.join(config.data_cache_dir, "pca_embeddings")
            if config.use_embedding_cache
            else None
        )
        embedding_cols = get_top_k_embedding_dims(
            embeddings, config.embedding_k, pca_cache_file
        )

        # Add embeddings to input features
        ser_dataset = ser_dataset.map(
            add_embeddings,
            fn_kwargs={
                "model": model,
                "processor": processor,
                "cache_dir": embeddings_cache_dir,
                "dims": embedding_cols,
            },
            batched=True,
            batch_size=8,
        )

        # add embedding columns as speech features
        config.xcs += [f"embedding_{i}" for i in embedding_cols]

    # Binarise xds into privileged and unprivileged groups
    waf_dataset = deepcopy(ser_dataset)
    if config.use_one_hot:
        xds = []
        for xd in config.xds:
            one_hot_encoder = fit_one_hot_encoder(waf_dataset, [xd])
            waf_dataset = waf_dataset.map(
                onehot_protected_attribute,
                fn_kwargs={"attribute": xd, "encoder": one_hot_encoder},
                batched=True,
                batch_size=8,
            )
            xds.extend(one_hot_encoder.get_feature_names_out([xd]).tolist())
    # skip for now
    # elif config.use_combination:
    #     xds_encoded = encode_fairness_binary(batch, xds)

    else:
        xds = config.xds
        for xd in config.xds:
            waf_dataset = waf_dataset.map(
                binarise_protected_attribute,
                fn_kwargs={"attribute": xd},
                batched=True,
                batch_size=8,
            )

    waf_dataset = waf_dataset.map(
        to_waf_dataset,
        fn_kwargs={"xcs": config.xcs, "xds": config.xds, "label_fn": bce_per_class},
        batched=True,
        batch_size=8,
    )

    dataset_cache_path = os.path.join(config.data_cache_dir, "waf_dataset")
    os.makedirs(dataset_cache_path, exist_ok=True)
    waf_dataset.save_to_disk(dataset_cache_path)
    print(f"WAF dataset saved to {dataset_cache_path}")
    return waf_dataset, config
