import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam, SGD

import numpy

import argparse
import glob


#from matplotlib import pyplot as plot

import copy
from time import sleep

import gym

from utils import Buffer, collect_one_episode, normalize_obs, copy_params, avg_params
from pong import Player, Value

def main(args):

    device=args.device

    env = gym.make(args.env)
    # env = gym.make('Assault-ram-v0')

    n_frames = args.n_frames
    max_len = args.max_len

    # create a policy
    player = Player(n_in=128 * n_frames, n_hid=args.n_hid, n_out=6).to(device)

    print('Reloading from {}'.format(args.saveto))
    checkpoint = torch.load(args.saveto, map_location=device)
    player.load_state_dict(checkpoint['player'])

    player.eval()

    _, _, _, _, _, ret_ = collect_one_episode(env, player, max_len=max_len, 
                                              deterministic=True, 
                                              n_frames=n_frames,
                                              rendering=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-max-len', type=int, default=100000)
    parser.add_argument('-n-frames', type=int, default=1)
    parser.add_argument('-env', type=str, default='Pong-ram-v0')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-n-hid', type=int, default=256)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args)
