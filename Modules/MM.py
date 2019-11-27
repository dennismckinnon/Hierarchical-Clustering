import torch
import torch.nn as nn
import torch.nn.functional as F

from Modules.conv import Convolutor
from Modules.common import *


def MMMLoss(x, pi, mu, reduce=True):
    x = x.unsqueeze(1)
    # Negative Log Liklihood of the multinoulis
    logpi = torch.log(pi)
    logmu = torch.log(mu)
    logommu = torch.log(1-mu)

    mix_log_probs = (logpi + (x*logmu + (1-x)*logommu).sum(-1))
    max_log_probs = torch.max(mix_log_probs, dim=-1, keepdim=True)[0]
    mix_log_probs = mix_log_probs - max_log_probs

    probs = torch.exp(mix_log_probs).sum(-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob


def dummy(x, pi, mu):
    return x
    
    
class MMN(nn.Module):
    def __init__(self, inputs, noulis, lr=1e-3, Loss=MMMLoss,
                 FeedForward=dummy, pre=None):
        super(MMN, self).__init__()
        self.inputs = inputs
        self.noulis = noulis
        
        self.Pis = nn.Parameter(torch.randn((1, noulis)))
        self.Mus = nn.Parameter(torch.randn((1, noulis, inputs)))
    
        self.Pi_adam = Adam(self.Pis)
        self.Mu_adam = Adam(self.Mus)
        
        self.lr = lr
        
        self.FF = FeedForward
        self._loss = Loss
        self.pre = pre if pre is not None else lambda x:x
    
    def params(self):  
        Pis = F.softmax(self.Pis, dim=1)
        Mus = torch.sigmoid(self.Mus)
        
        return (Pis, Mus)
    
    def forward(self, x, lr=None):
        # This returns both the p vector to higher levels and the dP/dx for feed back
        # If in training mode it will also perform update of this layer's parameters
        
        (pi, mu) = self.params()
        
        xt = torch.zeros_like(x, requires_grad=True)
        xt.data = x
        x = self.pre(xt)
        
        # Do parameter learning
#         if (self.training):
#             self.learn(x, lr)
            
        p = self.FF(x, pi, mu)
        
        # Calculate backwards vector
        loss = self._loss(x, pi, mu)
        loss.backward()
        dpdx = xt.grad
        
        return (p.detach(), dpdx.detach())
    
    def learn(self, x, lr=None):
        
        x = self.pre(x)
        
        (pi, mu) = self.params()
        
        if (lr is None):
            lr = self.lr
            
        self.Pis.grad = torch.zeros_like(self.Pis)
        self.Mus.grad = torch.zeros_like(self.Mus) 
            
        loss = self._loss(x, pi, mu)
        loss.backward()
        
        self.Pis.data  -= lr*self.Pi_adam(self.Pis.grad)
        self.Mus.data  -= lr*self.Mu_adam(self.Mus.grad)
        
        
    def loss(self, x):
        x = self.pre(x)
        (pi, mu) = self.params()
        return self._loss(x, pi, mu)