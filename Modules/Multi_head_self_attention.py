#multi-head_attention.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modules.self_attention import SelfAttention
import torch.nn as nn
import torch
torch.set_default_dtype(torch.float32)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,hidden_dims,num_heads):
        super(MultiHeadSelfAttention,self).__init__()
        self.attention_heads=nn.ModuleList(
            [SelfAttention(hidden_dim=hidden_dims,num_heads=num_heads) for head in range(num_heads)]
            )
        # print(self.attention_heads)
    
    def forward(self,input_token_matrix):
        attention_outputs=[head(input_token_matrix) for head in self.attention_heads]
        concatenated_heads=torch.cat(attention_outputs,dim=-1)
        # print(concatenated_heads.shape)
        return concatenated_heads

# head=MultiHeadSelfAttention(hidden_dims=512,num_heads=8)
# test=torch.rand(32,5,512)
# head(test)
