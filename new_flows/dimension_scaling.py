# Let's just implement a simple dimension 
import numpy as np
import torch 
from torch.nn import functional as F
from nflows.transforms.base import Transform

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

        
        sign_vec = torch.ones(batch_size, 1)
        # if self.negative: 
        #     sign_vec = -1 * sign_vec

        exp_scale = torch.exp(self.weights)
        final_scaling = sign_vec * exp_scale

        z = sign_vec * final_scaling * inputs

        log_det = float(torch.sum(self.weights))
        logabsdet = torch.full((batch_size,), log_det)
        return z, logabsdet
    
    def inverse(self, z, context=None):
        """
            Get the bacwards (inverse) flow of this layer.
        """
        batch_size = z.shape[0]
        
        sign_vec = torch.ones(z.shape[0], 1)
        
        # if self.negative:
        #     sign_vec = -1 * sign_vec
        
        scale_vec = torch.exp(self.weights)
        x = sign_vec * (1.0 / scale_vec)

        log_det = float(torch.sum(-self.weights))

        logabsdet = torch.full((batch_size,), log_det)

        return x, log_det