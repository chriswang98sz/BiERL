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
from BiES import meta_train
from BiES_LSTM import meta_lstm_train
from BOES import bo_train

def gradient_update(args, synced_model, returns, returns_with_entropy, random_seeds, neg_list,
                    num_eps, unperturbed_results, env, sigma):
    def fitness_shaping():
        sorted_returns_backwards = sorted(returns_with_entropy)[::-1]
        lamb = len(returns_with_entropy)
        shaped_returns = []
        denom = 0
        for r in returns_with_entropy:
            num = max(0, math.log(lamb / 2 + 1, 2) - math.log(sorted_returns_backwards.index(r) + 1, 2))
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
    print('Iteration num: %d\n'
          'Average reward: %f\n'
          'Standard Deviation: %f\n'
          'Max reward: %f\n'
          'Min reward: %f\n'
          'Sigma: %f\n'
          'Unperturbed rank: %s\n'
          'Unperturbed reward: %f' %
          (num_eps+1, np.mean(returns), np.std(returns), max(returns), min(returns),
           sigma, rank_diag,unperturbed_results))
    for i in range(args.n):
        np.random.seed(random_seeds[i])
        multiplier = -1 if neg_list[i] else 1
        reward = shaped_returns[i]

        for k, v in synced_model.es_params():
            eps = np.random.normal(0, 1, v.size())
            v += torch.from_numpy(args.lr / (40 * sigma) *
                                  (reward * multiplier * eps)).float()
    return synced_model, rank



def do_rollouts(args, model, random_seeds, return_queue, env, is_negative):
    state = env.reset()
    state = torch.from_numpy(state)
    this_model_return = 0
    this_model_return_with_entropy = 0
    for step in range(args.max_episode_length):
        state = state.float()
        dist = model.forward(state)
        action = dist.sample()
        if type(action)==torch.Tensor:
            action=action.data.numpy()[0]
        entropy = sum(dist.entropy().data.numpy()[0])*args.alpha
        next_state, reward, done, _ = env.step(action)
        if type(reward)==torch.Tensor:
            reward=reward.data.numpy()[0]
        state = next_state
        this_model_return += reward
        this_model_return_with_entropy += reward
        this_model_return_with_entropy += entropy
        if done:
            break
        state = torch.from_numpy(state)
    return_queue.put((random_seeds, this_model_return, this_model_return_with_entropy,
                      is_negative))
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


def perturb_model(sigma, model, random_seed, env):
    positive_model = ESContinuous(env)
    negative_model = ESContinuous(env)
    positive_model.load_state_dict(model.state_dict())
    negative_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (positive_k, positive_v), (negative_k, negative_v) in zip(positive_model.es_params(),
                                                                  negative_model.es_params()):
        eps = np.random.normal(0, 1, positive_v.size())
        positive_v += torch.from_numpy(sigma * eps).float()
        negative_v += torch.from_numpy(sigma * -eps).float()
    return [positive_model, negative_model]

def train_loop_ESAC(args, synced_model,meta_synced_model, sac_model,memory,env):
    #mp.set_start_method("spawn")
    def flatten(raw_results, index):
        notflat_results = [result[index] for result in raw_results]
        return notflat_results
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:1' if use_cuda else 'cpu')
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 1200
    updates = 0
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    print("============================================================================================")
    print("Training Continuous Env...")
    print("Base Model:ESAC")
    print("Temperature Factor:{},".format(args.alpha))
    print("Learning Rate of Network:{},\nLearning Rate of Sigma:{},\nLearning Rate of SAC:{},".format(args.lr, args.lr_meta, args.lr_sac))
    print("Batch Size of Network:{},\nBatch Size of Sigma:{},".format(args.n, args.m))
    print("Total Interations:{},\nUpdate Frequency of Sigma:{}.".format(args.T, args.t))
    print("============================================================================================")
    np.random.seed()
    start_time = time.time()
    if args.use_meta == 1 and args.meta_model != 2:
        if args.meta_model == 0:
            input = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
        elif args.meta_model == 1:
            input = [torch.zeros(1, args.m) for _ in range(args.t)]  # make a sequence of length 10
        sigma = meta_synced_model.forward(input)
    else:
        sigma = args.sigma
    for gradient_updates in range(args.T):
        processes = []
        manager = mp.Manager()
        return_queue = manager.Queue()
        all_seeds, all_models = [], []
        for i in range(int(args.n / 2)):
            random_seed = np.random.randint(2 ** 30)
            two_models = perturb_model(sigma, synced_model, random_seed, env)
            all_seeds.append(random_seed)
            all_seeds.append(random_seed)
            all_models += two_models
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
        seeds, results, results_with_entropy,neg_list = [flatten(raw_results, index)
                                                                      for index in [0, 1, 2, 3]]

        unperturbed_results = do_rollouts_unperturbed(args, synced_model, env)
        synced_model, rank = gradient_update(args, synced_model, results, results_with_entropy, seeds,
                                                  neg_list, gradient_updates,
                                                  unperturbed_results, env, sigma)
        if gradient_updates % 5 == 4 and np.random.random() < epsilon_by_frame(gradient_updates):
            print("============================================================================================")
            print("SAC Training...")
            print("============================================================================================")
            for i_episode in range(5):
                episode_reward = 0
                episode_steps = 0
                done = False
                state = env.reset()
                state = torch.from_numpy(state).to(device)
                while not done:
                    if gradient_updates < 10:
                        state = state.float()
                        action = env.action_space.sample()
                    else:
                        state = state.float()
                        dist = sac_model.policy.forward(state)
                        action = dist.sample()  # Sample action from policy
                        if type(action) == torch.Tensor:
                            action = action.cpu()
                            action = action.data.numpy()[0]

                    if len(memory) > args.batch_size:
                        # Number of updates per step in environment
                        for i in range(1):
                            # Update parameters of all the networks
                            critic_loss, qf_2_loss, policy_loss, ent_loss, alpha = sac_model.update_parameters(memory,
                                                                                                               args.batch_size,
                                                                                                               updates)
                            updates += 1
                    next_state, reward, done, _ = env.step(action)  # Step
                    if type(reward) == torch.Tensor:
                        reward = reward.data.numpy()[0]
                    episode_reward += reward
                    episode_steps += 1
                    next_state = torch.from_numpy(next_state).to(device)
                    # Ignore the "done" signal if it comes from hitting the time horizon.
                    # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                    mask = 1.0 if episode_steps == env._max_episode_steps else float(not done)
                    memory.push(state.cpu(), action, reward, next_state.cpu(), mask)  # Append transition to memory
                    state = next_state
            print("============================================================================================")
            print("SAC Training Over.")
            print("============================================================================================")
            synced_model.load_state_dict(sac_model.policy.state_dict())
        if args.use_meta == 1:
            if args.meta_model == 0:
                input = torch.tensor([[(rank - 1) / (args.n),
                                       (unperturbed_results - np.mean(results)) / (np.std(results) + 1e-8),
                                       (max(results) - np.mean(results)) / (np.std(results) + 1e-8),
                                       (min(results) - np.mean(results)) / (np.std(results) + 1e-8),
                                       ]], dtype=torch.float32)
                if gradient_updates % args.t == (args.t - 1):
                    meta_synced_model = meta_train(args, synced_model, meta_synced_model, env, input)
                sigma = meta_synced_model(input)
            elif args.meta_model == 1:
                results = sorted(results)
                meta_state = torch.from_numpy(np.array([(results - np.mean(results)) / (np.std(results) + 1e-8)])).to(
                    torch.float32)
                input[0:-1] = input[1:]
                input[-1] = meta_state
                if gradient_updates % args.t == (args.t - 1):
                    meta_synced_model = meta_lstm_train(args, synced_model, meta_synced_model, env, input)
                sigma = meta_synced_model(input)
            else:
                if gradient_updates % args.t == (args.t - 1):
                    sigma = bo_train(args, synced_model, env)
        else:
            if gradient_updates % args.t == (args.t - 1):
                results = torch.tensor(results)
                std_loss = torch.nn.functional.smooth_l1_loss(torch.max(results), torch.mean(results))
                sigma = sigma + torch.clamp((args.lr / (40 * sigma)) * std_loss, 0, 1e-4)
                sigma = sigma.item()
        print('Time: %.1f' % (time.time() - start_time))

