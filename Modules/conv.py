# conv.py

# This file contains the Convolutor class which is supposed to be
# inherited by various modules in order to simplify the process
# of turning a basic module into a convolutional one

# The class manages all the nitty gritty details of folding and
# unfolding inputs and outputs. This version makes no assumpyions
# about what functions the module that inherits it will provide
# It only assumes that it will need the 4D (BS, C, W, H) format
# to be converted into flattened windows 2D (BSxL, WxH)


import torch
import torch.nn as nn

class Convolutor(nn.Module):
    def __init__(self, in_size, filters, kernel_size, stride=1):
        super().__init__()
        
        assert isinstance(stride, int), 'Stride must be an int got {}'.format(stride)
        
        (k0, k1) = kernel_size
        (c,h,w) = in_size
        
        # D1 = product of kernel dims and input channels
        D = int(c * k0 * k1)
        hp = int((h-k0)//stride+1)
        wp = int((w-k1)//stride+1)
        L = int(hp*wp)
        
        self.L = L
        self.Hp = hp
        self.Wp = wp
        self.D = D
        self.C = c
        self.K = kernel_size
        self.F = filters
        self.input_shape = in_size
        self.output_shape = (filters, hp, wp)
        
        
        self._unfold = nn.Unfold(kernel_size, stride=stride)
        self._fold_forward  = nn.Fold((hp,wp), (1,1), stride=1)
        self._fold_backward = nn.Fold((h,w), kernel_size, stride=stride)
        
        temp = torch.zeros((1,D,L))
        self._normalizer = nn.Parameter(self._fold_backward(temp), requires_grad=False)

    def fold_forward(self, x, bs):
        return self._fold_forward(x.view(bs, self.L, self.F).transpose(1,2))

    def fold_backward(self, x, bs, normalize=True):
        folded = self._fold_backward(x.view(bs, self.L, self.D).transpose(1,2))
        return (folded/self.normalizer if normalize else folded)
    
    def unfold_input(self, x):
        return self._unfold(x).transpose(1,2).contiguous().view(-1, self.D)
    