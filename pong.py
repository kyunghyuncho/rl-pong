import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam, SGD

import numpy

from matplotlib import pyplot as plot

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

def main():

    env = gym.make('Pong-ram-v0')
    # env = gym.make('Assault-ram-v0')

    n_frames = 1

    # create a policy
    player = Player(n_in=128 * n_frames, n_hid=32, n_out=6).to(device)

    # create a value estimator
    value = Value(n_in=128 * n_frames, n_hid=32).to(device)
    value_old = Value(n_in=128 * n_frames, n_hid=32).to(device)
    copy_params(value, value_old)

    # initialize optimizers
    opt_player = Adam(player.parameters(), lr=0.0001)
    opt_value = Adam(value.parameters(), lr=0.0001)

    # initialize replay buffer
    replay_buffer = Buffer(max_items=50000, n_frames=n_frames)

    n_iter = 1000
    init_collect = 1
    n_collect = 1
    n_value = 150
    n_policy = 150
    disp_iter = 1
    val_iter = 1

    max_len = 1000
    batch_size = 1000

    ent_coeff = 0. #0.001
    discount_factor = .95

    value_loss = -numpy.Inf
    ret = -numpy.Inf
    entropy = -numpy.Inf
    valid_ret = -numpy.Inf

    return_history = []

    for ni in range(n_iter):
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
            logp = torch.log(batch_pi.gather(1, batch_a.long()))

            # (clipped) importance weight: 
            # because the policy may have changed since the tuple was collected.
            iw = torch.exp((logp.clone().detach() - torch.log(batch_q)).clamp(max=0.))
        
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
            
            logp = torch.log(batch_pi.gather(1, batch_a.long()))
            
            # advantage: r(s,a) + \gamma * V(s') - V(s)
            adv = batch_r + discount_factor * batch_vn - batch_v
            adv = adv / adv.abs().max().clamp(min=1.)
            
            loss = -(adv * logp)
            
            # (clipped) importance weight: 
            # because the policy may have changed since the tuple was collected.
            iw = torch.exp((logp.clone().detach() - torch.log(batch_q)).clamp(max=0.))
        
            loss = iw * loss
            
            # entropy regularization: though, it doesn't look necessary in this specific case.
            ent = (batch_pi * torch.log(batch_pi)).sum(1)
            
            if entropy == -numpy.Inf:
                entropy = ent.mean().item()
            else:
                entropy = 0.9 * entropy + 0.1 * ent.mean().item()
            
            loss = (loss + ent_coeff * ent).mean()
            
            loss.backward()
            opt_player.step()

    plot.figure()

    plot.plot(return_history)
    plot.grid(True)
    plot.xlabel('# of plays x {}'.format(n_collect))
    plot.ylabel('Return over the episode of length {}'.format(max_len))

    plot.show()
    plot.savefig('return_log.pdf', dpi=150)

    torch.save({
        'n_iter': n_iter,
        'n_collect': n_collect,
        'n_value': n_value,
        'n_policy': n_policy,
        'max_len': max_len,
        'batch_size': batch_size,
        'player': player.state_dict(),
        'value': value.state_dict(),
        'return_history': return_history, 
        }, 'saved_model.th')


if __name__ == '__main__':
    main()
