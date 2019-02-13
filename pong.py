import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

import copy
from time import sleep

import numpy

import gym

class Player(nn.Module):
    def __init__(self, n_in=128, n_hid=100, n_out=6):
        super(Player, self).__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, n_hid),
                                   nn.ReLU(),
                                   nn.Linear(n_hid, n_out))
        self.softmax = nn.Softmax()
    
    def forward(self, obs, normalized=False):
        if normalized:
            return self.softmax(self.layers(obs))
        else:
            return self.layers(obs)

class Value(nn.Module):
    def __init__(self, n_in=128, n_hid=100):
        super(Value, self).__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, n_hid),
                                   nn.ReLU(),
                                   nn.Linear(n_hid, 1))
    
    def forward(self, obs):
        return self.layers(obs)

def copy_params(from_, to_):
    for f_, t_ in zip(from_.parameters(), to_.parameters()):
        t_.data.copy_(t_.data)

def normalize_obs(obs):
    return obs.astype('float32') / 255.

# collect data
def collect_one_episode(env, player, max_len=50, discount_factor=0.9, deterministic=False, rendering=False):

    episode = []

    observations = []

    rewards = []
    crewards = []

    actions = []
    action_probs = []

    obs = env.reset()
    for ml in range(max_len):
        if rendering:
            env.render()
#             sleep(0.5)
    #     action = env.action_space.sample()
        obs = normalize_obs(obs)

        out_probs = player(torch.from_numpy(obs), normalized=True)
        if deterministic:
            action = numpy.argmax(out_probs.data.numpy())
        else:
            act_dist = Categorical(out_probs)
            action = act_dist.sample().item()
        action_prob = out_probs[action].item()

        observations.append(obs)
        actions.append(action)
        action_probs.append(action_prob)

        obs, reward, done, info = env.step(action)

        rewards.append(reward)

    rewards = numpy.array(rewards)

    for ri in range(len(rewards)):
        factors = (discount_factor ** numpy.arange(len(rewards)-ri))
        crewards.append(numpy.sum(rewards[ri:] * factors))

    return observations, crewards, actions, action_probs, rewards.sum()

class Buffer:
    def __init__(self, max_items=10000):
        self.max_items = max_items
        self.buffer = []
        
    def add(self, observations, crewards, actions, action_probs):
        new_n = len(observations)
        old_n = len(self.buffer)
        if new_n + old_n > self.max_items:
            del self.buffer[:new_n]
        for o, c, a, p in zip(observations, crewards, actions, action_probs):
            self.buffer.append((o, c, a, p))
            
    def sample(self, n=100):
        idxs = numpy.random.choice(len(self.buffer),n)
        return [self.buffer[ii] for ii in idxs]

    
    
def main():

    # create an environment
    env = gym.make('Pong-ram-v0')

    # create two models
    player = Player(n_in=128, n_hid=20, n_out=6)
    player_old = Player(n_in=128, n_hid=20, n_out=6)
    copy_params(player, player_old)

    # create a value estimator
    value = Value(n_in=128, n_hid=20)

    opt_player = Adam(player.parameters())
    opt_value = Adam(value.parameters())

    # initialize replay buffer
    replay_buffer = Buffer(max_items=100000)

    # o_, c_, a_, ap_, ret_ = collect_one_episode(env, player, max_len=10000, discount_factor=0.9, rendering=True)

n_iter = 1000 #1000
n_collect = 10 #100
n_value = 100 #100
n_policy = 5 #10
disp_iter = 1
val_iter = 1

max_len = 1000
batch_size = 1000

coeff = 0. #1.
ent_coeff = 0. #0.001
discount_factor = 0.9

value_loss = -numpy.Inf
ret = -numpy.Inf
entropy = -numpy.Inf

for ni in range(n_iter):
    if numpy.mod(ni, val_iter) == 0:
        _, _, _, _, ret_ = collect_one_episode(env, player, max_len=max_len, deterministic=True)
        print('Valid run', ret_)
    
    for ci in range(n_collect):
        o_, c_, a_, ap_, ret_ = collect_one_episode(env, player, max_len=max_len, discount_factor=discount_factor)
        replay_buffer.add(o_, c_, a_, ap_)
        if ret == -numpy.Inf:
            ret = ret_
        else:
            ret = 0.9 * ret + 0.1 * ret_
    
    # fit a value function
    for vi in range(n_value):
        opt_value.zero_grad()
        
        batch = replay_buffer.sample(batch_size)
        batch_x = torch.from_numpy(numpy.stack([ex[0] for ex in batch]).astype('float32'))
        batch_y = torch.from_numpy(numpy.stack([ex[1] for ex in batch]).astype('float32'))
        pred_y = value(batch_x).squeeze()
        loss = ((batch_y - pred_y) ** 2)
        
        batch_q = torch.from_numpy(numpy.stack([ex[3] for ex in batch]).astype('float32'))
        logp = torch.log(batch_pi.gather(1, batch_a.long()))

        iw = torch.exp((logp.detach() - torch.log(batch_q)).clamp(max=0.))
    
#         print('iw', iw.mean())
        
        loss = iw * loss
        
        loss = loss.mean()
        
        loss.backward()
        opt_value.step()
        
    if value_loss < 0.:
        value_loss = loss.item()
    else:
        value_loss = 0.9 * value_loss + 0.1 * loss.item()
    
    if numpy.mod(ni, disp_iter) == 0:
        print('# plays', (ni+1) * n_collect, 'return', ret, 'value_loss', value_loss, 'neg entropy', entropy)
    
    # fit a policy
    for pi in range(n_policy):
        opt_player.zero_grad()
        
        batch = replay_buffer.sample(batch_size)
        batch_x = torch.from_numpy(numpy.stack([ex[0] for ex in batch]).astype('float32'))
        batch_r = torch.from_numpy(numpy.stack([ex[1] for ex in batch]).astype('float32')[:,None])
        batch_v = value(batch_x)
        batch_a = torch.from_numpy(numpy.stack([ex[2] for ex in batch]).astype('float32')[:,None])
        batch_q = torch.from_numpy(numpy.stack([ex[3] for ex in batch]).astype('float32'))

        batch_pi = player(batch_x, normalized=True)
        batch_pi_old = player_old(batch_x, normalized=True)
        
        logp = torch.log(batch_pi.gather(1, batch_a.long()))
        logp_old = torch.log(batch_pi_old.gather(1, batch_a.long()))
        
        loss = -((batch_r - batch_v.detach()) * logp)
        
#         print('adv', (batch_r - batch_v).mean().item())

        iw = torch.exp((logp.detach() - torch.log(batch_q)).clamp(max=0.))
    
#         print('iw', iw.mean())
        
        loss = iw * loss
        
        kl = -(batch_pi_old * torch.log(batch_pi)).sum(1)
        ent = -(batch_pi * torch.log(batch_pi)).sum(1)
        
        if entropy == -numpy.Inf:
            entropy = ent.mean().item()
        else:
            entropy = 0.9 * entropy + 0.1 * ent.mean().item()
        
        loss = (loss + coeff * kl - ent_coeff * ent).mean()
        
        loss.backward()
        opt_player.step()
        
    copy_params(player, player_old)

if __name__ == "__main__":
    main()