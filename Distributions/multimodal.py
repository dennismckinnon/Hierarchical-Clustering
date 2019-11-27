import torch
from torch.utils.data import Dataset

# I have written two pytorch dataset classes which support multimodal classification tasks
# (also unimodal as a special case, set modes to 1). They both are essentially the same
# the JIT version will generate samples on the fly the non-JIT version pre-generates them all
# I don't know if memory is a concern or if one is faster than the other.

# Arguments:
# classes: The number of classes in the classification task
# modes: How many clusters are associated with each class
# dim: the dimension of the input space
# inverses: If you set this to True, the number of modes is effectively doubles
#           It will include a cluster mirrored across 0,0 (that way the mean of each class is 0,0)
# noise: the sigma of the gaussian noise added to the mode archetypes
# samples: how many samples for each mode, class pair to generate
#          if you have 3 classes, 2 modes each and 100 samples the whole dataset will be
#          3*2*100 = 600 samples
# transform: is for optional transformation pipelineing the pytorch Datasets support

# How it works:
# A number of centers or "archetypes" are generated and assigned to classes. the total number 
# of archetypes is classes * modes. From each archetype some gaussian noise is added with 
# variance given by the noise parameter. If the noise is 0 it will return the archetypes exactly
# (so there isn't a point setting samples > 1) You can think of it as a mixture of gaussians
# with variance given by noise centered at each archetype with a class label given to each.

# Note:
# There are no guarantees that the distributions don't overlap :S oh well this should
# be a less significant problem in higher dimensions theoretically

# Its intended to be used with pytorch's DataLoader. Examples are below. If you don't use it
# with the DataLoader, you would have to manually shuffle (Though not with the JIT version)

class GuassianClassification(Dataset):
    def __init__(self, classes, modes, dim, inverses=False, noise=0, samples=1, transform=None):
        self.classes = classes
        self.modes = modes
        self.inverses = inverses
        self.noise = noise
        self.samples = samples
        self.transform = transform
        
        self.archetypes = torch.randn((classes*modes, dim))
        
        if (inverses):
            self.modes *= 2
            self.archetypes = torch.cat((self.archetypes, -self.archetypes), 0)
            
        self.data = torch.zeros(((self.classes*self.modes*samples, dim)))
        self.labels = torch.zeros((self.classes*self.modes*samples, 1))
        
        for i in range(self.archetypes.shape[0]):
            self.labels[i*samples:(i+1)*samples] = i % classes
            sampleNoise = torch.randn((samples, dim))
            self.data[i*samples:(i+1)*samples] = self.archetypes[i] + noise*sampleNoise
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        sample = (self.data[idx,:], self.labels[idx,:])
        if (self.transform):
            sample = self.transform(sample)
        return sample
    
class GuassianClassificationJIT(Dataset):
    def __init__(self, classes, modes, dim, inverses=False, noise=0, samples=1, transform=None):
        self.classes = classes
        self.modes = modes
        self.dim = dim
        self.inverses = inverses
        self.noise = noise
        self.samples = samples
        self.transform = transform
        
        self.archetypes = torch.randn((classes*modes, dim))
        
        if (inverses):
            self.modes *= 2
            self.archetypes = torch.cat((self.archetypes, -self.archetypes), 0)
        
    def __len__(self):
        return self.classes*self.modes*self.samples
    
    def __getitem__(self, idx):
        # Pick archetype
        a = torch.randint(self.archetypes.shape[0], (1,))
        c = a % self.classes
        sampleNoise = torch.randn((self.dim))
        sample = (self.archetypes[a[0],:] + self.noise*sampleNoise, c)
        if (self.transform):
            sample = self.transform(sample)
        return sample
    
    
#############################################################################
#                      MultiModal Multinouli Datasets
#############################################################################

class MultinouliClassification(Dataset):
    def __init__(self, classes, modes, dim, noise=None, samples=1, transform=None):
        self.classes = classes
        self.modes = modes
        self.noise = noise
        self.samples = samples
        self.transform = transform
        
        # If noise is provided, instead of each position having a probability
        # of being 0 or 1 individually, bitstring centers are constructed with
        # a uniform (across the bits) probabitity of a flip is provided
        if (noise is None):
            self.archetypes = torch.rand((classes*modes, dim))
        else:
            centers = torch.randint(0, 2, (classes*modes, dim), dtype=torch.float)
            self.archetypes = self.noise*centers + (1-self.noise)*(1-centers)
            
        self.data = torch.zeros(((self.classes*self.modes*samples, dim)))
        self.labels = torch.zeros((self.classes*self.modes*samples, 1), dtype=torch.long)
        
        for i in range(self.archetypes.shape[0]):
            self.labels[i*samples:(i+1)*samples] = i % classes
            self.data[i*samples:(i+1)*samples] = torch.bernoulli(self.archetypes[i].expand(samples, -1))
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        sample = (self.data[idx,:], self.labels[idx,:])
        if (self.transform):
            sample = self.transform(sample)
        return sample
    
class MultinouliClassificationJIT(Dataset):
    def __init__(self, classes, modes, dim, noise=None, samples=1, transform=None):
        self.classes = classes
        self.modes = modes
        self.dim = dim
        self.noise = noise
        self.samples = samples
        self.transform = transform
        
        # If noise is provided, instead of each position having a probability
        # of being 0 or 1 individually, bitstring centers are constructed with
        # a uniform (across the bits) probabitity of a flip is provided
        if (noise is None):
            self.archetypes = torch.rand((classes*modes, dim))
        else:
            centers = torch.randint(0, 2, (classes*modes, dim), dtype=torch.float)
            self.archetypes = self.noise*centers + (1-self.noise)*(1-centers)
        
    def __len__(self):
        return self.classes*self.modes*self.samples
    
    def __getitem__(self, idx):
        # Pick archetype
        a = torch.randint(self.archetypes.shape[0], (1,))
        c = a % self.classes
        sample = (torch.bernoulli(self.archetypes[a[0],:]), c)
        if (self.transform):
            sample = self.transform(sample)
        return sample