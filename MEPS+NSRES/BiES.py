from __future__ import absolute_import, division, print_function
import os
import math
import numpy as np

import torch

import torch.multiprocessing as mp
from torch.autograd import Variable

from ES_network import ESContinuous
from Meta_network import MetaES
import time


def gradient_update_sigma(args, meta_synced_model, returns, random_seeds, neg_list):
    def fitness_shaping():
        sorted_returns_backwards = sorted(returns)[::-1]
        lamb = len(returns)
        shaped_returns = []
        denom = 0
        for r in returns:
            num = max(0, math.log(lamb / 2 + 1, 2) - math.log(sorted_returns_backwards.index(r) + 1, 2))
            denom += num
            shaped_returns.append(num)
        shaped_returns = np.array(shaped_returns)
        shaped_returns = list(shaped_returns / denom - 1 / lamb)
        return shaped_returns

    batch_size = len(returns)
    assert batch_size == args.m
    assert len(random_seeds) == batch_size
    shaped_returns = fitness_shaping()
    for i in range(args.m):
        np.random.seed(random_seeds[i])
        multiplier = -1 if neg_list[i] else 1
        reward = shaped_returns[i]

        for k, v in meta_synced_model.es_params():
            eps = np.random.normal(0, 1, v.size())
            v += torch.from_numpy(args.lr_meta / (40 * args.sigma_meta) *
                                  (reward * multiplier * eps)).float()
    return meta_synced_model


def do_rollouts(args, model, random_seeds, return_queue, env, is_negative):
    state = env.reset()
    state = torch.from_numpy(state)
    this_model_return = 0
    for step in range(args.max_episode_length):
        state = state.float()
        dist = model.forward(state)
        action = dist.sample()
        if type(action)==torch.Tensor:
            action=action.data.numpy()[0]
        next_state, reward, done, _ = env.step(action)
        if type(reward)==torch.Tensor:
            reward=reward.data.numpy()[0]
        state = next_state
        this_model_return += reward
        if done:
            break
        state = torch.from_numpy(state)
    return_queue.put((random_seeds, this_model_return,is_negative))


def perturb_model_single(sigma, model, random_seed, env):
    new_model = ESContinuous(env)
    new_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (k, v)in new_model.es_params():
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(sigma * eps).float()
    return [new_model]


def perturb_meta_model(args, meta_model, random_seed_meta):
    positive_meta_model = MetaES()
    negative_meta_model = MetaES()
    positive_meta_model.load_state_dict(meta_model.state_dict())
    negative_meta_model.load_state_dict(meta_model.state_dict())
    np.random.seed(random_seed_meta)
    for (positive_k, positive_v), (negative_k, negative_v) in zip(positive_meta_model.es_params(),
                                                                  negative_meta_model.es_params()):

        eps = np.random.normal(0, 1, positive_v.size())
        positive_v += torch.from_numpy(args.sigma_meta * eps).float()
        negative_v += torch.from_numpy(args.sigma_meta * -eps).float()
    return [positive_meta_model, negative_meta_model]

def meta_train(args, synced_model, meta_synced_model, env,input):
    #mp.set_start_method("spawn")
    def flatten(raw_results, index):
        notflat_results = [result[index] for result in raw_results]
        return notflat_results
    np.random.seed()
    print("============================================================================================")
    print("Training Meta Learner...")
    print("============================================================================================")
    processes = []
    manager = mp.Manager()
    return_queue = manager.Queue()
    all_seeds, all_models = [], []
    for j in range(int(args.m / 2)):
        random_seed = np.random.randint(2 ** 30)
        random_seed_meta = np.random.randint(2 ** 30)
        positive_meta_model,negative_meta_model = perturb_meta_model(args, meta_synced_model
                                                                     ,random_seed_meta)
        positive_sigma= positive_meta_model(input)
        negative_sigma= negative_meta_model(input)
        positive_model = perturb_model_single(positive_sigma, synced_model, random_seed, env)
        all_seeds.append(random_seed_meta)
        all_models += positive_model
        negative_model = perturb_model_single(negative_sigma, synced_model, random_seed, env)
        all_seeds.append(random_seed_meta)
        all_models += negative_model
        assert len(all_seeds) == len(all_models)
    is_negative = True
    while all_models:
        perturbed_model = all_models.pop()
        seed = all_seeds.pop()
        p = mp.Process(target=do_rollouts, args=(args, perturbed_model, seed, return_queue, env, is_negative))
        p.start()
        processes.append(p)
        is_negative = not is_negative
    assert len(all_seeds) == 0
    for p in processes:
        p.join()
    raw_results = [return_queue.get() for p in processes]
    seeds, results, neg_list = [flatten(raw_results, index) for index in [0, 1, 2]]
    meta_synced_model = gradient_update_sigma(args,meta_synced_model, results, seeds, neg_list)
    return meta_synced_model

