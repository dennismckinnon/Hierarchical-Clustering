import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Convolutor(nn.Module):
    def __init__(self, in_size, out_channels, kernel_size, stride=1):
        super(Convolutor, self).__init__()
        
        (k0, k1) = kernel_size
        (c,h,w) = in_size
        
        # D1 = product of kernel dims and input channels
        D = c * k0 * k1
        hp = (h-k0)//stride+1
        wp = (w-k1)//stride+1
        L = hp*wp
        
        self.L = L
        self.D = D
        self.C = c
        self.K = kernel_size
        self.G = out_channels
        
        self.unfold = nn.Unfold(kernel_size, stride=stride)
        self.fold_forward  = nn.Fold((hp,wp), (1,1), stride=1)
        self.Process = None
        
    def forward(self, x):
        (bs,c,h,w) = x.shape
        x1 = self.unfold(x).transpose(1,2).contiguous().view(-1, self.D)
        yuf = self.Process(x1)
        y = self.fold_forward(yuf.view(bs,self.L,self.G).transpose(1,2))
        return (y,None)
    
    def learn(self, x):
        x1 = self.unfold(x).view(-1, self.D)
        self.Process.learn(x1)
    
    def params(self):
        return self.Process.params()
    
    def loss(self, x):
        return torch.tensor(0)
    
    
    

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
        