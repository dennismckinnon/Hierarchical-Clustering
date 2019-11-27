# common.py

# This file contains various bits of frequently used code that honestly
# would be too much effort as of right now to place in a proper module
# and yet is used only by the modules in this folder so why not put 
# them all here

import torch
import torch.nn as nn
import torch.nn.functional as F

class Adam(nn.Module):
    def __init__(self, param, betas=(0.9, 0.999), eps=1e-8):
        super(Adam, self).__init__()
        self.register_buffer('beta1', torch.tensor(betas[0]))
        self.register_buffer('beta2', torch.tensor(betas[1]))
        self.register_buffer('eps', torch.tensor(eps))
        
        self.register_buffer('m', torch.zeros_like(param))
        self.register_buffer('v', torch.zeros_like(param))
        self.register_buffer('t', torch.tensor(0.0))
        
    def forward(self, g):
        self.m = self.beta1 * self.m + (1-self.beta1) * g
        self.v = self.beta2 * self.v + (1-self.beta2) * g**2
        self.t += 1
        
        m_hat = self.m/(1 - self.beta1**self.t)
        v_hat = self.v/(1 - self.beta2**self.t)
        
        return m_hat / (torch.sqrt(v_hat) + self.eps)

def cleanlr(lr, l):
    if (not isinstance(lr, (list, tuple))):
        lr = [lr]*l

    if (len(lr) != l):
        raise ValueError("Learning Rates must be length {}".format(l))
        
    return lr