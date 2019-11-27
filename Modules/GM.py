import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from Modules.conv import Convolutor
from Modules.common import *


################################################################
#                     BASIC FUNCTIONS
################################################################

def GMMLoss(batch, pi, sigmas, mus, reduce=True):
    logpi = pi.log()
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob

def GMMProbs(batch, pi, sigmas, mus, reduce=True, log=False, stable=False, normalize=False):
    logpi = pi.log()
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    
    if (log):
        return g_log_probs
    
    if (stable):
        max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
        g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    
    if (normalize):
        g_probs /= g_probs.sum(-1, keepdim=True)
        
    return g_probs

def MixtureLogProb(x, pi, sig, mu):
    if (len(x.shape)==2):
        x = x.unsqueeze(1)
        
    mal_dist = ((x-mu)**2/sig).sum(-1)
    log_prob = torch.log(pi) + mal_dist
    
    return log_prob


##########################################################
#                Gaussian Mixture Module
##########################################################

class GM(nn.Module):
    def __init__(self, inputs, gaussians, lr=1e-3, min_sig=0.01, start_sig=None,
                 Loss=GMMLoss, FeedForward=MixtureLogProb, pre=None):
        super().__init__()
        self.inputs = inputs
        self.gaussians = gaussians
        
        if (start_sig is not None):
            start_sig = torch.log(torch.tensor(start_sig).float())
            self.Sigs = nn.Parameter(start_sig*torch.ones(1,gaussians, inputs))
        else:
            self.Sigs = nn.Parameter(torch.randn((1, gaussians, inputs)))
        
        self.Pis = nn.Parameter(torch.randn((1, gaussians)))
        self.Mus = nn.Parameter(torch.randn((1, gaussians, inputs)))
    
        self.Pi_adam = Adam(self.Pis)
        self.Sig_adam = Adam(self.Sigs)
        self.Mu_adam = Adam(self.Mus)
        
        self.lr = cleanlr(lr, 3)
        self.min_sig = torch.log(torch.tensor(min_sig).float())
        
        self.FF = FeedForward
        self._loss = Loss
        self.pre = pre if pre is not None else lambda x:x
    
    def params(self):  
        Pis = F.softmax(self.Pis, dim=1)
        Sigs = self.Sigs.exp()
        Mus =  self.Mus
        
        return (Pis, Sigs, Mus)
    
    def forward(self, x, lr=None):
        # This returns both the p vector to higher levels and the dP/dx for feed back
        # If in training mode it will also perform update of this layer's parameters
        
        (pi, sig, mu) = self.params()
        
        xt = torch.zeros_like(x, requires_grad=True)
        xt.data = x
        x = self.pre(xt)
        
        # Do parameter learning
        if (self.training):
            self.learn(x, lr)
            
        p = self.FF(x, pi, sig, mu)
        
        # Calculate backwards vector
        loss = self._loss(x, pi, sig, mu)
        loss.backward()
        dpdx = xt.grad
        
        return (p.detach(), dpdx.detach())
    
    def learn(self, x, lr=None):
        
        x = self.pre(x)
        
        (pi, sig, mu) = self.params()
        
        if (lr is None):
            lr = self.lr
        lr = cleanlr(lr, 3)
            
        self.Pis.grad = torch.zeros_like(self.Pis)
        self.Sigs.grad = torch.zeros_like(self.Sigs)
        self.Mus.grad = torch.zeros_like(self.Mus) 
            
        loss = self._loss(x, pi, sig, mu)
        loss.backward()
        
        self.Pis.data  -= lr[0]*self.Pi_adam(self.Pis.grad)
        self.Sigs.data -= lr[1]*self.Sig_adam(self.Sigs.grad)#.clamp(0.01)
        self.Mus.data  -= lr[2]*self.Mu_adam(self.Mus.grad)
        
        # Range bounding
        self.Sigs.data = torch.clamp(self.Sigs, min=self.min_sig)
        
    def loss(self, x):
        x = self.pre(x)
        (pi, sig, mu) = self.params()
        return self._loss(x, pi, sig, mu)


    
####################################################################
#                        Convolutional GMN                         #
####################################################################

class CGM(Convolutor):
    def __init__(self, in_size, out_channels, kernel_size, stride=1, lr=1e-3,
                Loss=GMMLoss, FeedForward=MixtureLogProb, pre=None, min_sig=0.01, start_sig=None):
        super().__init__(in_size, out_channels, kernel_size, stride)
        self.Process = GMN(self.D, out_channels, lr=lr, Loss=Loss, FeedForward=FeedForward,
                           pre=pre, min_sig=min_sig, start_sig=start_sig)
        
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
            mss = (self.G, self.C)+self.K
        return (pi, sig.view(mss), mu.view(mss))
    
    def loss(self, x):
        x1 = self.unfold(x)
        return self.Process.loss(x1)
    
    
    

        
