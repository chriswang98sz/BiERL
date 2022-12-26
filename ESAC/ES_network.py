import math
import pickle
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.apply(init_weight)


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

    def sample(self, state):
        state = self.common1(state)
        state = self.common2(state)
        state = self.common3(state)
        state = self.common4(state)
        mean = self.policy(state)
        std = torch.exp(self.log_std)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean
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
class QNetwork(nn.Module):
    def __init__(self, env, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.num_inputs = env.observation_space.shape[0]

        self.num_actions = env.action_space.shape[0]
        # Q1 architecture
        self.linear1 = nn.Linear(self.num_inputs + self.num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(self.num_inputs + self.num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(init_weight)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1.clone())

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2.clone())

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, env, hidden_dim=64, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.num_inputs = env.observation_space.shape[0]

        self.num_actions = env.action_space.shape[0]
        self.linear1 = nn.Linear(self.num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, self.num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, self.num_actions)

        self.apply(init_weight)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-8)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
