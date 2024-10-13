#layer_normalization.py

import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)

class LayerNormalization(nn.Module):
    def __init__(self,ephsilon=10**(-5),hidden_dims=568):
        self.hidden_dims=hidden_dims
        super(LayerNormalization,self).__init__()
        self.ephsilon=ephsilon
        self.gamma=nn.Parameter(torch.rand(self.hidden_dims))
        self.bias=nn.Parameter(torch.rand(self.hidden_dims))
    def forward(self,input_after_attention):
        input_mean=torch.mean(input_after_attention,dim=-1,keepdim=True)
        input_std_dev=torch.std(input_after_attention,dim=-1,keepdim=True)
        normalised_layer=(
            (input_after_attention-input_mean)/(input_std_dev+self.ephsilon) * self.gamma + self.bias
        )
        return normalised_layer