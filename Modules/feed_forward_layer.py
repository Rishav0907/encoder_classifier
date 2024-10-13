#feed_forward.py
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)

class FeedForward(nn.Module):
    def __init__(self,input_dims,hidden_layer_dims,dropout):
        super(FeedForward,self).__init__()
        self.dropout=dropout
        self.ffl=nn.Sequential(
            nn.Linear(input_dims,hidden_layer_dims),
            nn.SELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_layer_dims,input_dims)
        )
    def forward(self,input_transformed_data):
        input_transformed_data=self.ffl(input_transformed_data)
        return input_transformed_data
