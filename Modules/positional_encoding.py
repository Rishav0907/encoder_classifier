# positional_encoding.py

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super(PositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, num_tokens, batch_size):
        # Create a tensor with shape (num_tokens, hidden_dim) for position indices
        position = torch.arange(0, num_tokens).unsqueeze(1).float()
        
        # Create a tensor with shape (hidden_dim,) for dimension indices
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.hidden_dim))
        
        # Calculate positional encoding matrix with shape (num_tokens, hidden_dim)
        pe = torch.zeros(num_tokens, self.hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Expand the positional encoding matrix to shape (batch_size, num_tokens, hidden_dim)
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return pe