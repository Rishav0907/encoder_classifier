# add_normalize.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.nn as nn
from Modules.layer_normalization import LayerNormalization
torch.set_default_dtype(torch.float32)

class AddAndNormalize(nn.Module):
    def __init__(self,hidden_dims):
        super(AddAndNormalize,self).__init__()
        self.hidden_dims=hidden_dims
        self.layer_norm=LayerNormalization(hidden_dims=self.hidden_dims)
    def forward(self,original_input,transformed_input):
        new_added_input=torch.add(original_input,transformed_input)
        normalized_output=self.layer_norm(new_added_input)
        return normalized_output