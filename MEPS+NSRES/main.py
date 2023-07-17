from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings("ignore")
import os
import gym
import argparse
import  time
import torch
import numpy as np
from Warm_start import warm_up
from env import get_env_space, get_env_info
from ES_network import ESContinuous
from Meta_network import MetaES,MetaES_LSTM
from MEPS import  train_loop_MEPS
from VanillaES import train_loop_VanillaES
from NSES import train_loop_NSES
from NSRES import train_loop_NSRES
from NSRAES import train_loop_NSRAES
parser = argparse.ArgumentParser(description='ES')
parser.add_argument('--env_name', default='Walker2d-v2',
                    metavar='ENV', help='environment')
parser.add_argument('--gamma', type=float, default=1,
                     help='gamma')
parser.add_argument('--sigma_meta', type=float, default=0.05, metavar='MSD',
                    help='initial metanoise standard deviation')
parser.add_argument('--sigma', type=float, default=0.05, metavar='MSD',
                    help='initial noise standard deviation, not need in meat-es')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr_meta', type=float, default=0.03, metavar='LR',
                    help='learning rate')
parser.add_argument('--m', type=int, default=200, metavar='M',
                    help='meta batch size')
parser.add_argument('--n', type=int, default=200, metavar='N',
                    help='batch size')
parser.add_argument('--k', type=int, default=10, metavar='M',
                    help='kNN,only need in NS-ES')
parser.add_argument('--max-episode-length', type=int, default=1000,
                    metavar='MEL', help='maximum length of an episode')
parser.add_argument('--alpha', type=float, default=0.1, metavar='TEM',
                    help='temperature factor')
parser.add_argument('--T', type=int, default=1000,
                    metavar='T', help='maximum number of iteration')
parser.add_argument('--t', type=int, default=10,
                    metavar='T', help='iterations update mata')
parser.add_argument('--base_methods', type=int, default=0,
                     help='the lower methods:{0:VanillaES,1:With entropy,2:NSES,3:NSRES}')
parser.add_argument('--use_meta', type=int, default=1,
                     help='use_meta:{0:False,1:True}')
parser.add_argument('--meta_model', type=int, default=0,
                     help='use-history:{0:Without LSTM,1:With LSTM,2:Non-parametric model(BO}')
parser.add_argument('--seed', type=int, default=0,
                     help='the random seed')

if __name__ == '__main__':
    print("============================================================================================")
    # set device to cpu or cuda
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        device = torch.device('cpu')
        print("Device set to : cpu")

    args = parser.parse_args()
    torch.set_num_threads(100) 
    env, env_continuous, num_states, num_actions = get_env_info(args.env_name)
    print("Env:{},Env_continuous:{},num_states:{},num_actions:{}".format(args.env_name,
                                                                         env_continuous,
                                                                         num_states,
                                                                         num_actions))
    print("============================================================================================")
    assert args.n % 2 == 0
    synced_model = ESContinuous(env)
    for param in synced_model.parameters():
        param.requires_grad = False
    assert args.m % 2 == 0
    if args.meta_model != 1:
        meta_synced_model = MetaES()
        for param in meta_synced_model.parameters():
            param.requires_grad = False
    else:
        meta_synced_model = MetaES_LSTM(dim_input=args.m)
        meta_synced_model = warm_up(args,meta_synced_model)
        for param in meta_synced_model.parameters():
            param.requires_grad = False


    if args.base_methods == 0:
        train_loop_VanillaES(args, synced_model,meta_synced_model, env)
    elif args.base_methods == 1:
        train_loop_MEPS(args, synced_model, meta_synced_model, env)
    elif args.base_methods == 2:
        train_loop_NSES(args,meta_synced_model, env)
    elif args.base_methods == 3:
        train_loop_NSRES(args,meta_synced_model, env)
    elif args.base_methods == 4:
        train_loop_NSRAES(args,meta_synced_model, env)
