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


class ESContinuous(nn.Module):
    def __init__(self,env,
                 dim_hidden0=64,
                 dim_hidden1=64,
                 log_std=-3.5,
                 ):
        super(ESContinuous, self).__init__()
        self.dim_state = env.observation_space.shape[0]
        self.dim_hidden0 = dim_hidden0
        self.dim_hidden1 = dim_hidden1
        self.dim_action = env.action_space.shape[0]
        self.common1 = nn.Linear(self.dim_state, self.dim_hidden0)
        self.common2 = nn.Tanh()
        self.common3 = nn.Linear(self.dim_hidden0, self.dim_hidden1)
        self.common4 = nn.Tanh()
        self.policy =  nn.Linear(self.dim_hidden1, self.dim_action)
        self.log_std = nn.Parameter(torch.ones(1, self.dim_action) * log_std, requires_grad=True)


    def forward(self, x):
        x = self.common1(x)
        x = self.common2(x)
        x = self.common3(x)
        x = self.common4(x)
        mean = self.policy(x)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        return dist

    def es_params(self):
        return [(k, v) for k, v in zip(self.state_dict().keys(), self.state_dict().values())]

'''
class ESDiscrete(nn.Module):
    def __init__(self, env, dim_hidden0=32,dim_hidden1=512):
        super(ESDiscrete, self).__init__()
        self.dim_state = env.observation_space.shape[0]
        self.dim_hidden0 = dim_hidden0
        self.dim_hidden1 = dim_hidden1
        self.dim_action = env.action_space.n
        self.policy = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_hidden0),
            nn.LeakyReLU(),
            nn.Linear(self.dim_hidden0, self.dim_hidden1),
            nn.LeakyReLU(),
            nn.Linear(self.dim_hidden1, self.dim_hidden0),
            nn.LeakyReLU(),
            nn.Linear(self.dim_hidden0, self.dim_action),
            nn.Softmax(dim=-1),
        )

        self.apply(init_weight)

    def forward(self, x):

        action_probs = self.policy(x.to(torch.float32))
        dist = Categorical(action_probs)
        entropy = dist.entropy()
        return action_probs,entropy

    def es_params(self):
        return [(k, v) for k, v in zip(self.state_dict().keys(), self.state_dict().values())]

'''