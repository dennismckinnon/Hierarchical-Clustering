import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal, Categorical
from torch.nn.utils.clip_grad import clip_grad_value_, clip_grad_norm_

import matplotlib
import matplotlib.pyplot as plt

class MDN(nn.Module):
    def __init__(self, inputs, outputs, gaussians):
        super(MDN, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.gaussians = gaussians
        
        self.As = nn.Linear(inputs, gaussians)
        self.Sigs = nn.Linear(inputs, outputs*gaussians)
        self.Mus = nn.Linear(inputs, outputs*gaussians)
        
    def forward(self, x):
        inshape = x.shape 
        outshape = inshape[:-1] + (self.gaussians, self.outputs)
        As = F.softmax(self.As(x), dim=-1)
        Sigs = self.Sigs(x).exp().view(*outshape)
        Mus =  self.Mus(x).view(*outshape)
        
        return (As, Sigs, Mus)
    
class DMDN(nn.Module):
    def __init__(self, inputs, outputs, gaussians):
        super(DMDN, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.gaussians = gaussians
        
        self.As = nn.Parameter(torch.randn((1, gaussians)))
        self.Sigs = nn.Parameter(torch.randn((1, gaussians, outputs)))
        self.Mus = nn.Parameter(torch.randn((1, gaussians, outputs)))
        
        self.register_buffer('node_mask', torch.ones(gaussians).byte())
        
    def forward(self, x):
        As = F.softmax(self.As[:, self.node_mask], dim=1)
        Sigs = self.Sigs[:, self.node_mask,:].exp()
        Mus =  self.Mus[:, self.node_mask, :]
        
        return (As, Sigs, Mus)
    
    def prune(self, N, thresh=0.0001):
        for i in range(N):
            As = F.softmax(self.As, dim=1) + 100*(1-self.node_mask.float()) # take masked nodes out of running
            minNode = torch.argmin(As, dim=1).item()
            if (As[0, minNode] < thresh):
                self._prune_node(minNode)
            else:
                break
        
    def _prune_node(self, node):
        if (self.node_mask[node] != 0):
            self.node_mask[node] = 0
            self.gaussians -= 1
        
    def create_node(self, As=None, Mu=None, Sig=None):
        print ("Not Implemented")
        return

class SMDN(nn.Module):
    def __init__(self, inputs, outputs, gaussians):
        super(SMDN, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.gaussians = gaussians
        
        self.As = nn.Linear(inputs, gaussians)
        self.Sigs = nn.Parameter(torch.randn((1, outputs*gaussians)))
        self.Mus = nn.Parameter(torch.randn((1, outputs*gaussians)))
        
    def forward(self, x):
        As = F.softmax(self.As(x), dim=-1)
        Sigs = self.Sigs.exp().view(-1, self.gaussians, self.outputs)
        Mus =  self.Mus.view(-1, self.gaussians, self.outputs)
        
        return (As, Sigs, Mus)

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

    
def sampleGMM(pi, sig, mu, samples=1, shape=None):
    mode_dist = Categorical(pi)
    mode = mode_dist.sample((samples,)).squeeze()
    normal_dist = Normal(mu[0,mode], sig[0,mode])
    
    sampled = normal_dist.sample()
    if (samples == 1):
        sampled = sampled.unsqueeze(0)
    return sampled

def plotSet(set, dims=[0,1]):
    fig = plt.figure(figsize=(8,8))
    colours = ['r', 'g', 'b']
    for (data, label) in set:
        label = label[:,0].numpy()
        plt.scatter(data[:,dims[0]], data[:,dims[1]], c=label, cmap=matplotlib.colors.ListedColormap(colours))
    plt.show()
        
def plotGMM(pi, sig, mu, samples=5000, classes=None, dims=[0,1], show=True):
    fig = plt.figure(figsize=(8,8))
    colours = ['r', 'g', 'b']
    data = sampleGMM(pi, sig, mu, samples=samples)
    if (classes is not None):
        label = data[:,classes].round().int()
        plt.scatter(data[:,dims[0]], data[:,dims[1]], c=label, cmap=matplotlib.colors.ListedColormap(colours))
    else:
        plt.scatter(data[:, dims[0]], data[:,dims[1]], c='b')
        
    plt.scatter(mu[0,:,dims[0]].data, mu[0,:,dims[1]].data, c='k')
    if show:
        plt.show()
