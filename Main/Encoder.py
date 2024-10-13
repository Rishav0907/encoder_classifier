#encoder.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SubModules.Encoder_block import EncoderBlock
import torch
import torch.nn as nn
from config.config import config

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.num_encoder_layers=config["NUM_ENCODER_BLOCKS"]
        self.encoders=nn.ModuleList([
            EncoderBlock() for _ in range(self.num_encoder_layers)
        ])
    def forward(self,input_token_matrix):
        for encoder in self.encoders:
            input_tensor=encoder(input_token_matrix)
        # print(input_tensor.shape)
            # break
        return input_tensor

enc=Encoder()
# # print(enc)
a=torch.rand(32,8,568)
# print(a)
enc(a)