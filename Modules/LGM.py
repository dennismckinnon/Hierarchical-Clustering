import torch
import torch.nn as nn
import torch.nn.functional as F

from Modules.conv import Convolutor
from Modules.GM import *

##########################################################
#            Linear Gaussian Mixture Module
##########################################################    
# This module is nearly identical to the first GM module
# The only difference is that it attempts to learn a 
# linear layer transformation.
# This hasn't been thoroughly tested...
# Why is it in its own file? Honestly, this was an idea
# I played with but didn't follow up on so cluttering the
# GM file with this code seems to detract from readability

class LGM(GM):
    def __init__(self, inputs, hiddens, gaussians, lr=1e-3, min_sig=0.01,
                 Loss=GMMLoss, FeedForward=MixtureLogProb):
        super().__init__(hiddens, gaussians, lr=1e-3, min_sig=0.01,
                 Loss=Loss, FeedForward=FeedForward)
        
        self.W = nn.Parameter(torch.randn((inputs, hiddens)))
        self.b = nn.Parameter(torch.zeros(hiddens))
        
        self.W_adam = Adam(self.W)
        self.b_adam = Adam(self.b)
        

    def forward(self, x):
        x1 = torch.einsum('ij,bi->bj', [self.W, x]) + self.b
        (y, dydx1) = super().forward(x1)
        dydx = torch.einsum('ij,bj->bi', [self.W, dydx1])
        return (y, dydx)
    
    def learn(self, x, lr=None):
    
        if (lr is None):
            lr = self.lr
            
        self.W.grad = torch.zeros_like(self.W)
        self.b.grad = torch.zeros_like(self.b)
        
        x1 = torch.einsum('ij,bi->bj', [self.W, x]) + self.b
        
        super().learn(x1, lr)
        
        self.W.data -= lr*self.W_adam(self.W.grad)
        self.b.data -= lr*self.b_adam(self.b.grad)
        
    def loss(self, x):
        x1 = torch.einsum('ij,bi->bj', [self.W, x]) + self.b
        return super().loss(x1)
    

####################################################################
#                        Convolutional LGM                         #
####################################################################
    
class CLGM(Convolutor):
    def __init__(self, in_size, hidden, out_channels, kernel_size, stride=1, lr=1e-3,
                Loss=GMMLoss, FeedForward=MixtureLogProb, min_sig=0.01):
        super().__init__(in_size, out_channels, kernel_size, stride)
        self.Process = LGMN(self.D, hidden, out_channels, lr=lr, Loss=Loss, FeedForward=FeedForward, min_sig=min_sig)
     
    def forward(self, x):
        (bs,c,h,w) = x.shape
        x1 = self.unfold(x)
        (yuf, dx1) = self.Process(x1)
        y = self.fold_forward(yuf, bs)
        dx = self.fold_backward(dx1, bs)
        return (y,dx)
    
    def learn(self, x):
        x1 = self.unfold(x)
        self.Process.learn(x1)
    
    def params(self, reshape=False):
        (pi, sig, mu) =  self.Process.params()
        mss = sig.shape
        if (reshape):
            pass
        return (pi, sig.view(mss), mu.view(mss))
    
    def loss(self, x):
        x1 = self.unfold(x)
        return self.Process.loss(x1)