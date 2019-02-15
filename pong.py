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

device='cuda'

class ResLinear(nn.Module):
    def __init__(self, n_in, n_out, act=nn.ReLU()):
        super(ResLinear, self).__init__()
        self.act = act
        self.linear = nn.Linear(n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        
        assert(n_in == n_out)
    
    def forward(self, x):
        h = self.act(self.bn(self.linear(x)))
        return h + x

class Player(nn.Module):
    def __init__(self, n_in=128, n_hid=100, n_out=6):
        super(Player, self).__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, n_hid),
                                    nn.BatchNorm1d(n_hid),
                                    nn.ReLU(),
                                    ResLinear(n_hid, n_hid, nn.ReLU()),
                                    nn.Linear(n_hid, n_out))
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, obs, normalized=False):
        if normalized:
            return self.softmax(self.layers(obs))
        else:
            return self.layers(obs)

class Value(nn.Module):
    def __init__(self, n_in=128, n_hid=100):
        super(Value, self).__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, n_hid),
                                    nn.BatchNorm1d(n_hid),
                                    nn.ReLU(),
                                    ResLinear(n_hid, n_hid, nn.ReLU()),
                                    nn.Linear(n_hid, 1))
    
    def forward(self, obs):
        return self.layers(obs)

def main(args):

    env = gym.make(args.env)
    # env = gym.make('Assault-ram-v0')

    n_frames = args.n_frames

    # initialize replay buffer
    replay_buffer = Buffer(max_items=args.buffer_size, n_frames=n_frames)

    n_iter = args.n_iter
    init_collect = args.init_collect
    n_collect = args.n_collect
    n_value = args.n_value
    n_policy = args.n_policy

    disp_iter = args.disp_iter
    val_iter = args.val_iter
    save_iter = args.save_iter

    max_len = args.max_len
    batch_size = args.batch_size

    ent_coeff = args.ent_coeff
    discount_factor = args.discount_factor

    value_loss = -numpy.Inf
    ret = -numpy.Inf
    entropy = -numpy.Inf
    valid_ret = -numpy.Inf

    offset = 0

    return_history = []

    # create a policy
    player = Player(n_in=128 * n_frames, n_hid=args.n_hid, n_out=6).to(device)

    # create a value estimator
    value = Value(n_in=128 * n_frames, n_hid=args.n_hid).to(device)
    value_old = Value(n_in=128 * n_frames, n_hid=args.n_hid).to(device)

    if args.cont:
        files = glob.glob("{}*th".format(args.saveto))
        iterations = [int(".".join(f.split('.')[:-1]).split('_')[-1].strip()) for f in files]
        last_iter = numpy.max(iterations)
        offset = last_iter-1
        print('Reloading from {}_{}.th'.format(args.saveto, last_iter))
        checkpoint = torch.load("{}_{}.th".format(args.saveto, last_iter))
        player.load_state_dict(checkpoint['player'])
        value.load_state_dict(checkpoint['value'])
        return_history = checkpoint['return_history']

    copy_params(value, value_old)

    # initialize optimizers
    opt_player = Adam(player.parameters(), lr=0.0001)
    opt_value = Adam(value.parameters(), lr=0.0001)

    for ni in range(n_iter):
        if numpy.mod(ni, save_iter) == 0:
            torch.save({
                'n_iter': n_iter,
                'n_collect': n_collect,
                'n_value': n_value,
                'n_policy': n_policy,
                'max_len': max_len,
                'n_hid': n_hid,
                'batch_size': batch_size,
                'player': player.state_dict(),
                'value': value.state_dict(),
                'return_history': return_history, 
                }, '{}_{}.th'.format(args.saveto,ni+offset+1))

        player.eval()

        if numpy.mod(ni, val_iter) == 0:
            _, _, _, _, _, ret_ = collect_one_episode(env, player, max_len=max_len, deterministic=True, n_frames=n_frames)
            return_history.append(ret_)
            if valid_ret == -numpy.Inf:
                valid_ret = ret_
            else:
                valid_ret = 0.9 * valid_ret + 0.1 * ret_
            print('Valid run', ret_, valid_ret)

        # collect some episodes using the current policy
        # and push (obs,a,r,p(a)) tuples to the replay buffer.
        nc = n_collect
        if ni == 0:
            nc = init_collect
        for ci in range(nc):
            o_, r_, c_, a_, ap_, ret_ = collect_one_episode(env, player, max_len=max_len, discount_factor=discount_factor, n_frames=n_frames)
            replay_buffer.add(o_, r_, c_, a_, ap_)
            if ret == -numpy.Inf:
                ret = ret_
            else:
                ret = 0.9 * ret + 0.1 * ret_
        
        # fit a value function
        # TD(1)
        value.train()
        for vi in range(n_value):
            opt_player.zero_grad()
            opt_value.zero_grad()
            
            batch = replay_buffer.sample(batch_size)

            batch_x = torch.from_numpy(numpy.stack([ex['current']['obs'] for ex in batch]).astype('float32')).to(device)
            batch_r = torch.from_numpy(numpy.stack([ex['current']['rew'] for ex in batch]).astype('float32')).to(device)
            batch_xn = torch.from_numpy(numpy.stack([ex['next']['obs'] for ex in batch]).astype('float32')).to(device)
            pred_y = value(batch_x).squeeze()
            pred_next = value_old(batch_xn).squeeze().clone().detach()
            loss_ = ((batch_r + discount_factor * pred_next - pred_y) ** 2)
            
            batch_a = torch.from_numpy(numpy.stack([ex['current']['act'] for ex in batch]).astype('float32')[:,None]).to(device)
            batch_pi = player(batch_x, normalized=True)
            batch_q = torch.from_numpy(numpy.stack([ex['current']['prob'] for ex in batch]).astype('float32')).to(device)
            logp = torch.log(batch_pi.gather(1, batch_a.long())+1e-8)

            # (clipped) importance weight: 
            # because the policy may have changed since the tuple was collected.
            iw = torch.exp((logp.clone().detach() - torch.log(batch_q+1e-8)).clamp(max=0.))
        
            loss = iw * loss_
            
            loss = loss.mean()
            
            loss.backward()
            opt_value.step()
        
        copy_params(value, value_old)
            
        if value_loss < 0.:
            value_loss = loss_.mean().item()
        else:
            value_loss = 0.9 * value_loss + 0.1 * loss_.mean().item()
        
        if numpy.mod(ni, disp_iter) == 0:
            print('# plays', (ni+1) * n_collect, 'return', ret, 'value_loss', value_loss, 'entropy', -entropy)
        
        # fit a policy
        value.eval()
        player.train()
        for pi in range(n_policy):
            opt_player.zero_grad()
            opt_value.zero_grad()
            
            batch = replay_buffer.sample(batch_size)
            
            batch_x = torch.from_numpy(numpy.stack([ex['current']['obs'] for ex in batch]).astype('float32')).to(device)
            batch_xn = torch.from_numpy(numpy.stack([ex['next']['obs'] for ex in batch]).astype('float32')).to(device)
            batch_r = torch.from_numpy(numpy.stack([ex['current']['rew'] for ex in batch]).astype('float32')[:,None]).to(device)
            
            batch_v = value(batch_x).clone().detach()
            batch_vn = value(batch_xn).clone().detach()
            
            batch_a = torch.from_numpy(numpy.stack([ex['current']['act'] for ex in batch]).astype('float32')[:,None]).to(device)
            batch_q = torch.from_numpy(numpy.stack([ex['current']['prob'] for ex in batch]).astype('float32')).to(device)

            batch_pi = player(batch_x, normalized=True)
            
            logp = torch.log(batch_pi.gather(1, batch_a.long())+1e-8)
            
            # advantage: r(s,a) + \gamma * V(s') - V(s)
            adv = batch_r + discount_factor * batch_vn - batch_v
            #adv = adv / adv.abs().max().clamp(min=1.)
            
            loss = -(adv * logp)
            
            # (clipped) importance weight: 
            # because the policy may have changed since the tuple was collected.
            iw = torch.exp((logp.clone().detach() - torch.log(batch_q+1e-8)).clamp(max=0.))
        
            loss = iw * loss
            
            # entropy regularization: though, it doesn't look necessary in this specific case.
            ent = (batch_pi * torch.log(batch_pi+1e-8)).sum(1)
            
            if entropy == -numpy.Inf:
                entropy = ent.mean().item()
            else:
                entropy = 0.9 * entropy + 0.1 * ent.mean().item()
            
            loss = (loss + ent_coeff * ent).mean()
            
            loss.backward()
            opt_player.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n-iter', type=int, default=1000)
    parser.add_argument('-n-collect', type=int, default=1)
    parser.add_argument('-init-collect', type=int, default=100)
    parser.add_argument('-n-value', type=int, default=150)
    parser.add_argument('-n-policy', type=int, default=150)
    parser.add_argument('-disp-iter', type=int, default=1)
    parser.add_argument('-val-iter', type=int, default=1)
    parser.add_argument('-save-iter', type=int, default=10)
    parser.add_argument('-max-len', type=int, default=1000)
    parser.add_argument('-batch-size', type=int, default=1000)
    parser.add_argument('-ent-coeff', type=float, default=0.)
    parser.add_argument('-discount-factor', type=float, default=0.95)
    parser.add_argument('-n-hid', type=int, default=256)
    parser.add_argument('-buffer-size', type=int, default=50000)
    parser.add_argument('-n-frames', type=int, default=1)
    parser.add_argument('-env', type=str, default='Pong-ram-v0')
    parser.add_argument('-cont', action="store_true", default=False)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args)
