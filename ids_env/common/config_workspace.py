#########################################################
# Configuration functions (torch device, seed)          #
#########################################################

import random
import numpy as np
import torch
import os
import stable_baselines3 as sb3

def config_device(device_name="cpu"):
    '''
    Config torch device (cpu, mps or cuda)
    '''
    device = torch.device("cpu")

    if device_name=="mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        
    elif "cuda" in device_name and torch.cuda.is_available():
        device = torch.device("cuda")
    
    return device

def config_seed(seed=0):
    '''
    Set random seed for deterministic experiments
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sb3.common.utils.set_random_seed(seed,using_cuda=True)