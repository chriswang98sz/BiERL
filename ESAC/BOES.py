from __future__ import absolute_import, division, print_function
import os
import math
import numpy as np
from bayes_opt import BayesianOptimization
import torch

import torch.multiprocessing as mp
from torch.autograd import Variable

from ES_network import ESContinuous
from Meta_network import MetaES
import time

def do_rollouts_unperturbed(args, model,env):
    state = env.reset()
    state = torch.from_numpy(state)
    this_model_return = 0
    for step in range(args.max_episode_length):
        state = state.float()
        dist = model.forward(state)
        action = dist.sample()
        if type(action) == torch.Tensor:
            action = action.data.numpy()[0]
        next_state, reward, done, _ = env.step(action)
        if type(reward) == torch.Tensor:
            reward = reward.data.numpy()[0]
        state = next_state
        this_model_return += reward
        if done:
            break
        state = torch.from_numpy(state)
    return this_model_return

def black_box_function(sigma, model,env,args):
    positive_model = ESContinuous(env)
    negative_model = ESContinuous(env)
    positive_model.load_state_dict(model.state_dict())
    negative_model.load_state_dict(model.state_dict())
    np.random.seed()
    for (positive_k, positive_v), (negative_k, negative_v) in zip(positive_model.es_params(),
                                                                  negative_model.es_params()):
        eps = np.random.normal(0, 1, positive_v.size())
        positive_v += torch.from_numpy(sigma * eps).float()
        negative_v += torch.from_numpy(sigma * -eps).float()
    ret_pos = do_rollouts_unperturbed(args, positive_model,env)
    ret_neg = do_rollouts_unperturbed(args, negative_model,env)
    ret = (ret_neg+ret_pos)/2
    return ret

def bo_train(args, synced_model, env):
    #mp.set_start_method("spawn")
    print("============================================================================================")
    print("Generating New Sigma...")
    print("============================================================================================")
    np.random.seed()
    pbounds = {'sigma': (0.02, 0.08)}
    optimizer = BayesianOptimization(
        f=lambda sigma: black_box_function(sigma=sigma,model=synced_model,env=env,args=args),
        pbounds=pbounds,
        random_state=1)
    optimizer.maximize(init_points=10,n_iter=20)
    sigma = optimizer.max['params']["sigma"]
    return sigma
