#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:32:54 2022

@author: negin
"""

import torch
import random
import numpy as np

# =============================================================================
# Towards result reproducability
# Codes and justification from:
#     https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
#     https://pytorch.org/docs/stable/notes/randomness.html
# =============================================================================

def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)    