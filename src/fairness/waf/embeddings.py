import os

import numpy as np
import torch

from utils import get_device


def get_embeddings(audios, paths, processor, model, cache_dir=None):

    embeddings = []
    ids = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    embedding_cache_paths = [
        os.path.join(cache_dir, id + ".npy") for id in ids
    ] if cache_dir is not None else []

    if cache_dir is not None and all(
        os.path.exists(path) for path in embedding_cache_paths
    ):
        print("Loading embeddings from cache...")
        embeddings = []
        for path in embedding_cache_paths:
            # np.load returns np.ndarray; wrap in torch tensor to match the
            # else-branch which returns a torch.Tensor.
            emb = torch.from_numpy(np.load(path))
            embeddings.append(emb)
        embeddings = torch.stack(embeddings, dim=0)

    else:
        print("Computing embeddings...")

        device = get_device()
        # batch_size=4 keeps peak memory manageable on T4 15GB even for
        # IEMOCAP utterances (which are much longer than CREMA-D's).
        batch_size = 4
        embeddings_list = []

        model = model.to(device)
        model.eval()

        for i in range(0, len(audios), batch_size):
            batch = audios[i:i + batch_size]
             # Preprocess audios
            inputs = processor(
                batch,
                sampling_rate=processor.sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Extract embeddings
            with torch.no_grad():
                outputs = model(
                    **inputs, output_hidden_states=True
                ) 
                batch_embeddings = outputs.hidden_states[0].mean(dim=1).detach().cpu()
                embeddings_list.append(batch_embeddings)
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        embeddings = torch.cat(embeddings_list, dim=0)

        # Cache all embeddings
        os.makedirs(cache_dir, exist_ok=True)
        for speech_path, emb in zip(paths, embeddings):
            file_name = os.path.splitext(os.path.basename(speech_path))[0]
            emb_path = os.path.join(cache_dir, f"{file_name}.npy")
            np.save(emb_path, emb)

    return embeddings


def add_embeddings(batch, model, processor, cache_dir=None, dims=None):

    embeddings = get_embeddings(
        batch["speech"], batch["path"], processor, model, cache_dir=cache_dir
    )
    _, embedding_dim = embeddings.shape

    # Add embeddings to dataset
    dims_to_use = dims if dims is not None else range(embedding_dim)

    # Ensure dims_to_use is a list or iterable of integers within the valid range
    assert all(
        0 <= dim < embedding_dim for dim in dims_to_use
    ), f"All elements in dims must be integers in the range [0, {embedding_dim - 1}]"

    for dim in dims_to_use:
        # Convert to numpy and add to batch
        batch[f"embedding_{dim}"] = embeddings[:, dim].numpy().tolist()

    return batch
