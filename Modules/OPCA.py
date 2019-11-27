# OPCA.py

# This file contains the implementation of the online PCA module
# developed from the search over orthogonal matrices.

# This module should be tested more thoroughly as it seems prone
# to some form of numerical instability in some circumstances
# possibly if the covariance matrix is not full rank?

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Modules.conv import Convolutor
from Modules.common import *

def RandomSON(n):
    M = np.random.randn(n,n)
    Q,R = np.linalg.qr(M)
    return Q
    

class OPCA(nn.Module):
    def __init__(self, inputs, outputs, eta=1e-3, gamma=0):
        super().__init__()
        
        if (inputs < outputs):
            raise ValueError("Inputs must be >= outputs. Got {} < {}".format(inputs, outputs))
        
        self.dummy = nn.Parameter(torch.zeros(1))
        
        # OPCA
        self.register_buffer('T', torch.tensor(RandomSON(inputs)[:,:outputs]).float())
        self.register_buffer('N', torch.diag(torch.arange(outputs,0,-1).float()))
        self.register_buffer('eta', torch.tensor(eta))
        
        # Centering
        self.register_buffer('mu', torch.zeros(inputs))
        self.register_buffer('gamma', torch.tensor(gamma))
        
    def forward(self, xin):
        
        # Centering
        x = xin-self.mu
        xt = torch.einsum('bi,ij->bj', [x, self.T])
 
        if (self.training):
            self.learn(xin)
        
        return xt
    
    def learn(self, xin):
        
        # Centering
        x = xin-self.mu
        
        # EMA
        self.mu *= (1-self.gamma)
        self.mu += self.gamma*(x.mean(0))
        
        # Batched OPCA
        xt = torch.einsum('bi,ij->bj', [x, self.T])
        X = torch.einsum('bk,bl->bkl',[xt, xt]).mean(0)
        C = torch.einsum('ik,kl->il', [self.N, X])
        Tgrad = -self.T@(C-C.transpose(0,1))
        
        self.T += self.eta*Tgrad
    
    def params(self):
        return self.T
    
class COPCA(Convolutor):
    def __init__(self, in_size, out_channels, kernel_size, stride=1, eta=1e-3, gamma=0):
        super().__init__(in_size, out_channels, kernel_size, stride)
        
        self.Process = OPCA(self.D, out_channels, eta, gamma)

    def forward(self, x):
        (bs,c,h,w) = x.shape
        x1 = self.unfold(x)
        yuf = self.Process(x1)
        y = self.fold_forward(yuf, bs)
        return (y,None)
    
    def learn(self, x):
        x1 = self.unfold(x)
        self.Process.learn(x1)
    
    def params(self):
        return self.Process.params()
    
    def loss(self, x):
        return torch.tensor(0)