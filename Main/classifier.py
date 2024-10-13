import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Main.Encoder import Encoder
from Modules.self_attention import SelfAttention
from config.config import config
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,vocab_size,embedding_dims):
        super(Classifier,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dims)
        self.encoder=Encoder()
        self.mean_pooling=nn.AdaptiveAvgPool1d(1)
        self.fc_layer=nn.Sequential(
            nn.Linear(config["HIDDEN_DIMS"],config["HIDDEN_DIMS"]*2),
            nn.ReLU(),
            nn.Linear(config["HIDDEN_DIMS"]*2,config["HIDDEN_DIMS"]*2),
            nn.ReLU(),
            nn.Linear(config["HIDDEN_DIMS"]*2,config["NUM_CLASSES"])
        )
        self.softmax=nn.Softmax(dim=1)

        # self.fc_layer=nn.Sequential()
    def forward(self,input_token_matrix):
        embedding=self.embedding(input_token_matrix)
        output=self.encoder(embedding)
        # output=self.attention(output)
        output=self.mean_pooling(output.transpose(1,2)).squeeze(2)
        output=self.fc_layer(output)
        probabilities=self.softmax(output)
        return probabilities


# clas=Classifier()
# # print(enc)
# a=torch.rand(32,8,568)
# # print(a)
# clas(a)