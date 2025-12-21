import os
import random

import numpy as np
import torch


def seed_torch(seed=1029):
    """
    > It sets the seed for the random number generator in Python, NumPy, and PyTorch

    :param seed: the random seed, defaults to 1029 (optional)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
