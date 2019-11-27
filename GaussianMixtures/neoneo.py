import torch
import torch.nn as nn
import torch.nn.functional as F

from GMM import *

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

class GMN(nn.Module):
    def __init__(self, inputs, gaussians, lr=1e-3, min_sig=0.01, start_sig=None,
                 Loss=GMMLoss, FeedForward=MixtureLogProb, pre=None):
        super(GMN, self).__init__()
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
#         if (self.training):
#             self.learn(x, lr)
            
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
    
class LGMN(GMN):
    def __init__(self, inputs, hiddens, gaussians, lr=1e-3, min_sig=0.01,
                 Loss=GMMLoss, FeedForward=MixtureLogProb):
        super(LGMN, self).__init__(hiddens, gaussians, lr=1e-3, min_sig=0.01,
                 Loss=GMMLoss, FeedForward=MixtureLogProb)
        
        self.W = nn.Parameter(torch.randn((inputs, hiddens)))
        self.b = nn.Parameter(torch.zeros(hiddens))
        
        self.W_adam = Adam(self.W)
        self.b_adam = Adam(self.b)
        

    def forward(self, x):
        x1 = torch.einsum('ij,bi->bj', [self.W, x]) + self.b
        (y, dydx1) = super(LGMN, self).forward(x1)
        dydx = torch.einsum('ij,bj->bi', [self.W, dydx1])
        return (y, dydx)
    
    def learn(self, x, lr=None):
    
        if (lr is None):
            lr = self.lr
            
        self.W.grad = torch.zeros_like(self.W)
        self.b.grad = torch.zeros_like(self.b)
        
        x1 = torch.einsum('ij,bi->bj', [self.W, x]) + self.b
        
        super(LGMN, self).learn(x1, lr)
        
        self.W.data -= lr*self.W_adam(self.W.grad)
        self.b.data -= lr*self.b_adam(self.b.grad)
        
    def loss(self, x):
        x1 = torch.einsum('ij,bi->bj', [self.W, x]) + self.b
        return super(LGMN, self).loss(x1)
        
        

####################################################################
#                        Convolutor                                #
####################################################################

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
        self.fold_backward = nn.Fold((h,w), kernel_size, stride=stride)
        self.Process = None
        
        temp = torch.zeros((1,D,L))
        self.normalizer = nn.Parameter(self.fold_backward(temp), requires_grad=False)
        
    def forward(self, x):
        (bs,c,h,w) = x.shape
        x1 = self.unfold(x).transpose(1,2).contiguous().view(-1, self.D)
        (yuf, dx1) = self.Process(x1)
        y = self.fold_forward(yuf.view(bs,self.L,self.G).transpose(1,2))
        dx = self.fold_backward(dx1.view(bs,self.L,self.D).transpose(1,2))/self.normalizer
        return (y, dx)
    
    def learn(self, x):
        x1 = self.unfold(x).view(-1, self.D)
        self.Process.learn(x1)
        
    def params(self):
        return self.Process.params()
    
    def loss(self, x):
        x1 = self.unfold(x).view(-1, self.D)
        return self.Process.loss(x1)
        



####################################################################
#                        Convolutional GMN                         #
####################################################################

class CGMN(Convolutor):
    def __init__(self, in_size, out_channels, kernel_size, stride=1, lr=1e-3,
                Loss=GMMLoss, FeedForward=MixtureLogProb, pre=None, min_sig=0.01, start_sig=None):
        super(CGMN, self).__init__(in_size, out_channels, kernel_size, stride)
        self.Process = GMN(self.D, out_channels, lr=lr, Loss=Loss, FeedForward=FeedForward,
                           pre=pre, min_sig=min_sig, start_sig=start_sig)
        
    def params(self, reshape=False):
        (pi, sig, mu) =  self.Process.params()
        mss = sig.shape
        if (reshape):
            mss = (self.G, self.C)+self.K
        return (pi, sig.view(mss), mu.view(mss))
        
class CLGMN(Convolutor):
    def __init__(self, in_size, hidden, out_channels, kernel_size, stride=1, lr=1e-3,
                Loss=GMMLoss, FeedForward=MixtureLogProb, min_sig=0.01):
        super(CLGMN, self).__init__(in_size, out_channels, kernel_size, stride)
        self.Process = LGMN(self.D, hidden, out_channels, lr=lr, Loss=Loss, FeedForward=FeedForward, min_sig=min_sig)
        
    def params(self, reshape=False):
        (pi, sig, mu) =  self.Process.params()
        mss = sig.shape
        if (reshape):
            pass
#             mss = (self.G, self.C)+self.K
        return (pi, sig.view(mss), mu.view(mss))