import math
import pickle
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

class MetaES(nn.Module):
    def __init__(self, dim_input=4,dim_hidden1=1024,dim_hidden2=32,dim_output=1):
        super(MetaES, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden1 = dim_hidden1
        self.dim_hidden2 = dim_hidden2
        self.dim_output = dim_output
        self.policy= nn.Sequential(
            nn.Linear(self.dim_input, self.dim_hidden1),
            nn.Linear(self.dim_hidden1, self.dim_hidden2),
            nn.Linear(self.dim_hidden2, self.dim_output),
            nn.Sigmoid()
        )
        self.apply(init_weight)

    def forward(self, x):

        output = self.policy(x)
        sigma = output.data.numpy()[0]*0.06+0.02
        return sigma

    def es_params(self):
        return [(k, v) for k, v in zip(self.state_dict().keys(), self.state_dict().values())]

class MetaES_LSTM(nn.Module):
    def __init__(self, dim_input=200,dim_hidden1=1024,dim_hidden2=32,dim_output=1):
        super(MetaES_LSTM, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden1 = dim_hidden1
        self.dim_hidden2 = dim_hidden2
        self.dim_output = dim_output
        self.lstm = nn.LSTM(dim_input, dim_hidden1)
        self.policy= nn.Sequential(
            nn.Linear(self.dim_hidden1, self.dim_hidden2),
            nn.Linear(self.dim_hidden2, self.dim_output),
            nn.Sigmoid()
        )
        self.hidden = (torch.randn(1, 1, dim_hidden1),
          torch.randn(1, 1, dim_hidden1))
        self.apply(init_weight)

    def forward(self, inputs):
        for i in inputs:
            out, self.hidden = self.lstm(i.view(1, 1, -1), self.hidden)
        inputs = torch.cat(inputs).view(len(inputs), 1, -1)
        (h_n, c_n) = (torch.randn(1, 1, self.dim_hidden1), torch.randn(1, 1, self.dim_hidden1))  # clean out hidden state
        out, (h_n, c_n) = self.lstm(inputs, self.hidden)
        output = self.policy(h_n)
        sigma = output.data.numpy()[0][0][0]*0.06+0.02
        return sigma

    def es_params(self):
        return [(k, v) for k, v in zip(self.state_dict().keys(), self.state_dict().values())]
