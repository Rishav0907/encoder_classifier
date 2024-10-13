#self-attention.py

import torch
import torch.nn as nn
import numpy as np
torch.set_default_dtype(torch.float32)

class SelfAttention(nn.Module):
    def __init__(self,hidden_dim,num_heads) -> None:
        super(SelfAttention,self).__init__()
        self.hidden_dims=hidden_dim
        self.softmax=nn.Softmax(dim=-1)
        self.num_heads=num_heads
        self.head_dims=self.hidden_dims//self.num_heads # we will later on concatenate and get a vector with dim same as hidden_dims
        self.W_Q=nn.Linear(in_features=self.hidden_dims,out_features=self.head_dims,bias=False).float()
        self.W_K=nn.Linear(in_features=self.hidden_dims,out_features=self.head_dims,bias=False).float()
        self.W_V=nn.Linear(in_features=self.hidden_dims,out_features=self.head_dims,bias=False).float()
        # print(self.W_Q.weight)
        # self.W_V=nn.Linear(self.head_dims,self.head_dims,bias=False)


    def forward(self,input_token_matrix,attn_mask=None):
        Q=self.W_Q(input_token_matrix)
        K=self.W_K(input_token_matrix)
        V=self.W_V(input_token_matrix)
        print(Q.shape)
        print(K.transpose(-1,-2).shape)
        # #self attention scores
        
        Q_x_K=torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.hidden_dims)
        # Q_x_K=Q_x_K/np.sqrt(self.hidden_dims)
        self.attention_scores=torch.matmul(self.softmax(Q_x_K),V)
        print(self.attention_scores.shape)
        return self.attention_scores
    
# head=SelfAttention(hidden_dim=512,num_heads=8)
# test=torch.rand(32,5,512)
# head(test)