from torch.autograd import Variable
import numpy as np
import torch

def to_var(obj):
    if obj is not None:
        if isinstance(obj, np.ndarray):
            obj = torch.from_numpy(obj)
        return Variable(obj) if type(obj) != Variable else obj
    return None