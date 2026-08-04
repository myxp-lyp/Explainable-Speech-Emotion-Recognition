import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For CUDA 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For performance consistency
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'