"""
Description:    Implementation of Longformer self-attention following the paper: https://arxiv.org/abs/2004.05150
Project:        Longformer
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

class LongformerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass