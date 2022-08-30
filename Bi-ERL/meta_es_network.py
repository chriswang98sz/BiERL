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
    def __init__(self, dim_input=4,dim_hidden=128,dim_output=1):
        super(MetaES, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.policy= nn.Sequential(
            nn.Linear(self.dim_input, self.dim_hidden),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.Linear(self.dim_hidden, self.dim_output),
            nn.Sigmoid()
        )
        self.apply(init_weight)

    def forward(self, x):

        output = self.policy(x)
        sigma = output.data.numpy()[0]/4+0.01
        return sigma

    def es_params(self):
        return [(k, v) for k, v in zip(self.state_dict().keys(), self.state_dict().values())]

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))