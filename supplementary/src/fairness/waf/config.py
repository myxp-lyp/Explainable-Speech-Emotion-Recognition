from dataclasses import dataclass, field

from databases.constants import demographic_columns


@dataclass
class WAFConfig:
    xds: list = field(default_factory=lambda: demographic_columns)
    xcs: list = field(default_factory=lambda: [])
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    embedding_k: int = 100
    device: str = "cuda"
    model_cache_dir: str = "./models/waf"
    data_cache_dir: str = "./data/waf"
    use_model_cache: bool = True
    use_data_cache: bool = True
    # flags for experimental purposes
    use_embeddings: bool = False
    use_combination: bool = False
    use_one_hot: bool = False
    use_embedding_cache: bool = True
