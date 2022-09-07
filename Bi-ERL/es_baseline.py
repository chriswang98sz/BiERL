from __future__ import absolute_import, division, print_function
import os
import math
import numpy as np

import torch

import torch.multiprocessing as mp
from torch.autograd import Variable

from es_network import ESContinuous, ESDiscrete
from meta_es_network import MetaES
import time


def gradient_update(args, synced_model, returns, random_seeds,
                    num_eps, num_frames, unperturbed_results, env):
    def fitness_shaping():
        sorted_returns_backwards = sorted(returns)[::-1]
        lamb = len(returns)
        shaped_returns = []
        denom = 0
        for r in returns:
            num = max(0, math.log(lamb + 1, 2) - math.log(sorted_returns_backwards.index(r) + 1, 2))
            denom += num
            shaped_returns.append(num)
        shaped_returns = np.array(shaped_returns)
        shaped_returns = list(shaped_returns / denom - 1 / lamb)
        return shaped_returns

    def unperturbed_rank(returns, unperturbed_results):
        nth_place = 1
        for r in returns:
            if r > unperturbed_results:
                nth_place += 1
        rank_diag = ('%d out of %d ' % (nth_place, len(returns) + 1))
        return rank_diag, nth_place

    rank_diag, rank = unperturbed_rank(returns, unperturbed_results)

    batch_size = len(returns)
    assert batch_size == args.n
    assert len(random_seeds) == batch_size
    shaped_returns = fitness_shaping()
    print('Episode num: %d\n'
          'Average reward: %f\n'
          'Standard Deviation: %f\n'
          'Max reward: %f\n'
          'Min reward: %f\n'
          'Sigma: %f\n'
          'Total num frames seen: %d\n'
          'Unperturbed rank: %s\n'
          'Unperturbed reward: %f' %
          (num_eps, np.mean(returns), np.std(returns), max(returns), min(returns),
           args.sigma, num_frames, rank_diag,unperturbed_results))
    for i in range(args.n):
        np.random.seed(random_seeds[i])
        reward = shaped_returns[i]

        for k, v in synced_model.es_params():
            eps = np.random.normal(0, 1, v.size())
            v += torch.from_numpy(args.lr / (40 * args.sigma) *
                                  (reward * eps)).float()
    return synced_model, rank


def do_rollouts(args, model, random_seeds, return_queue, env):
    state = env.reset()
    state = torch.from_numpy(state)
    this_model_return = 0
    this_model_num_frames = 0
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
        this_model_num_frames += 1
        if done:
            break
        state = torch.from_numpy(state)
    return_queue.put((random_seeds, this_model_return,this_model_num_frames))
def do_rollouts_unperturbed(args, model,env):
    state = env.reset()
    state = torch.from_numpy(state)
    this_model_return = 0
    for step in range(args.max_episode_length):
        if args.render=='True':
            try:
                env.render()
            except:
                pass
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

def perturb_model(sigma, model, random_seed, env):
    new_model = ESContinuous(env)
    new_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (k, v)in new_model.es_params():
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(sigma * eps).float()
    return [new_model]

def train_loop_es_baseline(args, synced_model, env):
    def flatten(raw_results, index):
        notflat_results = [result[index] for result in raw_results]
        return notflat_results

    if args.save == 'True':
        synced_model.save(args.save_path)
    if args.load == 'True':
        synced_model.load(args.save_path + "/" + args.load_name)

    print("============================================================================================")
    print("Training Continuous Env...")
    print("Temperature Factor:{},".format(args.alpha))
    print("Learning Rate of Network:{},\nSigma:{},".format(args.lr, args.sigma))
    print("Batch Size of Network:{},\nBatch Size of Sigma:{},".format(args.n, args.m))
    print("Total Interations:{},\nUpdate Frequency of Sigma:{},".format(args.T, args.t))
    print("============================================================================================")
    np.random.seed()
    num_eps = 0
    total_num_frames = 0
    start_time = time.time()
    for gradient_updates in range(args.T):
        processes = []
        manager = mp.Manager()
        return_queue = manager.Queue()
        all_seeds, all_models = [], []
        for i in range(args.n):
            random_seed = np.random.randint(2 ** 30)
            model = perturb_model(args.sigma, synced_model, random_seed, env)
            all_seeds.append(random_seed)
            all_models += model
        assert len(all_seeds) == len(all_models)
        while all_models:
            perturbed_model = all_models.pop()
            seed = all_seeds.pop()
            p = mp.Process(target=do_rollouts, args=(args, perturbed_model, seed, return_queue, env))
            p.start()
            processes.append(p)
        assert len(all_seeds) == 0
        for p in processes:
            p.join()
        raw_results = [return_queue.get() for p in processes]
        seeds, results, num_frames= [flatten(raw_results, index) for index in [0, 1, 2]]

        unperturbed_results = do_rollouts_unperturbed(args, synced_model, env)
        total_num_frames += sum(num_frames)
        num_eps += len(results)
        synced_model, rank = gradient_update(args, synced_model,seeds,
                                                   num_eps, total_num_frames,
                                                  unperturbed_results, env)
        print('Time: %.1f' % (time.time() - start_time))