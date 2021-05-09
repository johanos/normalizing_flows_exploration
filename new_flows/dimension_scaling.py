# Let's just implement a simple dimension 
import numpy as np
import torch 
from torch.nn import functional as F
from nflows.transforms.base import Transform
from nflows.utils import torchutils


class ScalingTransform(Transform):
    """
    A simple scaling layer, easy to work with etc...
    """

    def __init__(self, dim:int , negative: bool):
        super(ScalingTransform,self).__init__()
        self.dim = dim
        self.negative = negative
        self.weights = torch.nn.Parameter(torch.randn(dim))


    def forward(self, inputs: torch.Tensor, context=None):
        """ 
            define the forward pass of this normalizing flow layer
        """

        batch_size = inputs.shape[0]

        sign = -1 if self.negative else 1

        ones_vec =  torch.ones(batch_size, 1)

        exp_scale = sign * torch.exp(self.weights)
        final_scaling = ones_vec * exp_scale

        z = final_scaling * inputs

        logabsdet = torchutils.logabsdet(torch.diag(exp_scale))
        logabsdet = logabsdet * z.new_ones(batch_size)

        return z, logabsdet
    
    def inverse(self, z, context=None):
        """
            Get the bacwards (inverse) flow of this layer.
        """
        batch_size = z.shape[0]
        
        sign = -1 if self.negative else 1
        ones_vec =  torch.ones(batch_size, 1)

        exp_scale = sign * torch.exp(self.weights)
        x = (1.0 / exp_scale) * z

        logabsdet = torchutils.logabsdet(torch.diag(exp_scale))
        logabsdet = logabsdet * x.new_ones(batch_size)
        return x, logabsdet