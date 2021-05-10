# Let's just implement a simple dimension 
import numpy as np
import torch 
from torch.nn import functional as F
from nflows.transforms.base import Transform
from nflows.utils import torchutils


class TranslatingTransform(Transform):
    """
    A simple scaling layer, easy to work with etc...
    """

    def __init__(self, dim:int):
        super(TranslatingTransform,self).__init__()
        self.dim = dim
        self.t = torch.nn.Parameter(torch.randn(dim))


    def forward(self, inputs: torch.Tensor, context=None):
        """ 
            define the forward pass of this normalizing flow layer
        """

        # inputs are batch_size x 2 

        batch_size = inputs.shape[0]
        #print(translation.shape)
        
        z = inputs + self.t 


        #logabsdet = 0 because log(1) # because translations are metric transformations 
        logabsdet = 0 * inputs.new_ones(batch_size)

        return z, logabsdet
    
    def inverse(self, z, context=None):
        """
            Get the bacwards (inverse) flow of this layer.
        """
        batch_size = z.shape[0]
        
        x = z - self.t
        #logabsdet = 1
        logabsdet = 0 * z.new_ones(batch_size)
        return x, logabsdet