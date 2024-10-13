#encoder_block.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
import torch.nn as nn
from Modules.positional_encoding import PositionalEncoding
from Modules.feed_forward_layer import FeedForward
from Modules.Multi_head_self_attention import MultiHeadSelfAttention
from Modules.add_and_normalize import AddAndNormalize
from config.config import config
torch.set_default_dtype(torch.float32)

class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock,self).__init__()
        self.num_heads=config['HEADS']
        self.hidden_dimension=config['HIDDEN_DIMS']
        self.positional_encoding_instance=PositionalEncoding(hidden_dim=self.hidden_dimension)
        self.multi_head_self_attention=MultiHeadSelfAttention(hidden_dims=self.hidden_dimension,num_heads=self.num_heads)
        self.add_and_normalize=AddAndNormalize(hidden_dims=self.hidden_dimension)
        self.feed_forward_network=FeedForward(input_dims=self.hidden_dimension,hidden_layer_dims=4*self.hidden_dimension,dropout=0.05)
    def forward(self,input_token_matrix):
        
        positional_encoding=self.positional_encoding_instance(input_token_matrix.shape[1],input_token_matrix.shape[0])
        # print(positional_encoding.shape)

        position_encoded_input=torch.add(input_token_matrix,positional_encoding)
        # print(position_encoded_input)
        attention_outputs=self.multi_head_self_attention(position_encoded_input)
        # print(attention_outputs.shape)
        add_and_normalized_output=self.add_and_normalize(input_token_matrix,attention_outputs)
        feed_forward_output=self.feed_forward_network(add_and_normalized_output)
        # print(feed_forward_output.shape)
        add_and_normalized_output=self.add_and_normalize(add_and_normalized_output,feed_forward_output)
        return add_and_normalized_output

enc=EncoderBlock()
a=torch.rand(32,5,568)
# print(a.shape)
enc(a)