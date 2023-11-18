import os
import numpy as np
import torch.utils.data as data

def mkdir_p(path):
    '''make dir if not exist'''
    if not os.path.exists(path):
        os.makedirs(path)
