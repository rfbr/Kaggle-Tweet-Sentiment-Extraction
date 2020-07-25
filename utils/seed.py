import os
import random

import numpy as np
import torch


def set_seed(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Parameters:
        seed {int}: number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
