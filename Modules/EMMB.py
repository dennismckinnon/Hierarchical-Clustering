import torch
import torch.nn as nn
import torch.nn.functional as F

from Modules.conv import Convolutor
from Modules.common import *


class EMMB(nn.Module):
    def __init__(self, D, M, eta=0.95, FF=None):
        super().__init__()
        self.register_buffer('D', torch.tensor(D))
        self.register_buffer('M', torch.tensor(M))
        
        self.pi = nn.Parameter(torch.ones((1,M))/M, requires_grad=False)
        self.mu = nn.Parameter(torch.rand((1,M,D)), requires_grad=False)
        
        self.FF = nn.Softmax(dim=-1) if FF is None else FF
        
    def responsibility(self, x, raw=False, verbose=False):
        with silence(not verbose):
            print ('x', torch.isnan(x).sum()>0)
            print ('pi', torch.isnan(self.pi).sum()>0)
            print ('mu', torch.isnan(self.mu).sum()>0)
            # Negative Log Liklihood of the multinoulis
            logpi = torch.log(self.pi+1e-18)
            print ('logpi',torch.isnan(logpi).sum()>0)
            logmu = torch.log(self.mu+1e-18)
            print ('logmu',torch.isnan(logmu).sum()>0)
            logommu = torch.log(1-self.mu+1e-18)
            print ('logommu',torch.isnan(logommu).sum()>0)
            mix_log_probs = (logpi + (x*logmu + (1-x)*logommu).sum(-1))
            print ('mix_log_probs', torch.isnan(mix_log_probs).sum()>0)
            resp = F.softmax(mix_log_probs, dim=-1)
            print ('resp', torch.isnan(resp).sum()>0)

        if (raw):
            return mix_log_probs
        return resp
        
    def forward(self, x):
        x = x.view(-1,1,self.D)
        y = self.responsibility(x, raw=True)
        return self.FF(y)
        
    def learn(self, x):
        x = x.view(-1,1,self.D)
        
        # This learns via EM algorithm
        # E step: calculates the responsibility of each mode for each point
        # M step: calculates the pis and mus
        rik = self.responsibility(x)
        if (torch.isnan(rik).sum()>0):
            self.responsibility(x, verbose=True)
            raise ValueError("NaN Found")
        rk = rik.sum(0, keepdim=True)
        # rik:  (BS, M) -> (BS, M, 1)
        newmu = (rik.unsqueeze(-1)*x).sum(0, keepdim=True)/(rk.unsqueeze(-1)+1e-18)
        newpi = rk/x.shape[0]
        
        # The M update (Online Version)
        self.pi.data = eta*self.pi.data + (1-eta)*newpi
        self.mu.data = eta*self.mu.data + (1-eta)*newmu
        
    def params(self, sort=True):
        pi = self.pi.squeeze()
        mu = self.mu.squeeze()
        if (sort):
            pi, pin = pi.sort(descending=True)
            mu = mu[pin]
        return (pi, mu)
    
class CEMMB(Convolutor):
    def __init__(self, in_size, filters, kernel_size, stride=1, eta=0.95, FF=None):
        super().__init__(in_size, filters, kernel_size, stride=stride)
        self.Process = EMMB(self.D, filters, eta=eta, FF=FF)
        
        self.register_buffer('M', torch.tensor(filters))
        
    def forward(self, x):
        (bs,c,h,w) = x.shape
        x1 = self.unfold_input(x)
        yuf = self.Process(x1)
        y = self.fold_forward(yuf, bs)
        return y
    
    def learn(self, x):
        x1 = self.unfold_input(x)
        self.Process.learn(x1)
    
    def params(self, reshape=False, sort=True):
        (pi, mu) =  self.Process.params(sort)
        mss = mu.shape
        if (reshape):
            mss = (self.F, self.C)+self.K
        return (pi, mu.view(mss))