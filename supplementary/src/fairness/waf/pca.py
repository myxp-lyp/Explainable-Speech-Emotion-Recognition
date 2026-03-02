import os
import pickle

import numpy as np
from sklearn.decomposition import PCA


def get_principal_components(embeddings):
    pca = PCA(n_components=1)
    pca.fit(embeddings)
    return pca


def get_or_load_pca(embeddings, cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print("Loading PCA from cache...")
            return pickle.load(f)
    else:
        print("Computing PCA...")
        pca = get_principal_components(embeddings)
        with open(cache_path, "wb") as f:
            pickle.dump(pca, f)
        return pca


def get_top_k_embedding_dims(embeddings, k, cache_path=None):
    # Get the top k dimensions based on variance explained
    pca = get_or_load_pca(embeddings, cache_path=cache_path)
    pc1 = pca.components_[0]
    top_dims = np.argsort(np.abs(pc1))[::-1][:k]
    return top_dims
