import gym

import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam, SGD

import numpy

import copy
from time import sleep

def copy_params(from_, to_):
    for f_, t_ in zip(from_.parameters(), to_.parameters()):
        t_.data.copy_(f_.data)
        
def avg_params(from_, to_, coeff=0.95):
    for f_, t_ in zip(from_.parameters(), to_.parameters()):
        t_.data.copy_(coeff * t_.data + (1.-coeff) * f_.data)
        
def normalize_obs(obs):
    return obs.astype('float32') / 255.

# collect data
def collect_one_episode(env, player, max_len=50, discount_factor=0.9, 
                        deterministic=False, rendering=False, verbose=False, 
                        n_frames=1):
    episode = []

    observations = []

    rewards = []
    crewards = []

    actions = []
    action_probs = []
    
    prev_obs = [numpy.zeros(128).astype('float32')] * n_frames

    obs = env.reset()
    
    for ml in range(max_len):
        if rendering:
            env.render()
            sleep(0.05)
            
        obs = normalize_obs(obs)
        
        prev_obs.pop(0)
        prev_obs.append(obs)
        current_obs = numpy.concatenate(prev_obs)

        out_probs = player(torch.from_numpy(current_obs[None,:]).to(next(player.parameters()).device), normalized=True).squeeze()
        
        if deterministic:
            action = numpy.argmax(out_probs.to('cpu').data.numpy())
            if verbose:
                print(out_probs, action)
        else:
            act_dist = Categorical(out_probs.clamp(min=1e-8))
            action = act_dist.sample().item()
        action_prob = out_probs[action].item()

        observations.append(obs)
        actions.append(action)
        action_probs.append(action_prob)

        obs, reward, done, info = env.step(action)
        if deterministic and verbose:
            print(reward, done)
        
        rewards.append(reward)

    rewards = numpy.array(rewards)

    # it's probably not the best idea to compute the discounted cumulative returns here, but well..
    for ri in range(len(rewards)):
        factors = (discount_factor ** numpy.arange(len(rewards)-ri))
        crewards.append(numpy.sum(rewards[ri:] * factors))
        
    # discard the final 10%, because it really doesn't give me a good signal due to the unbounded horizon
    # this is only for training, not for computing the total return of the episode of the given length
    discard = max_len // 10
        
    return observations[:-discard], rewards[:-discard], crewards[:-discard], actions[:-discard], action_probs[:-discard], rewards.sum()

# simple implementation of FIFO-based replay buffer
class Buffer:
    def __init__(self, max_items=10000, n_frames=1):
        self.max_items = max_items
        self.n_frames = n_frames
        self.buffer = []
        
    def add(self, observations, rewards, crewards, actions, action_probs):
        new_n = len(observations)
        old_n = len(self.buffer)
        if new_n + old_n > self.max_items:
            del self.buffer[:new_n]
        for ii, (o, r, c, a, p, on, rn, cn, an, pn) in enumerate(zip(observations[:-1], rewards[:-1], 
                                                                     crewards[:-1], actions[:-1], action_probs[:-1],
                                             observations[1:], rewards[1:], crewards[1:], actions[1:], action_probs[1:])):
            act_obs = [numpy.zeros(o.shape).astype('float32')] * numpy.maximum(0,(self.n_frames-ii-1))
            act_obs = act_obs + observations[numpy.maximum(0, ii-self.n_frames+1):ii+1]
#             print(ii, len(act_obs))
            
            act_obs_next = [numpy.zeros(o.shape).astype('float32')] * numpy.maximum(0,(self.n_frames-ii-2))
            act_obs_next = act_obs_next + observations[numpy.maximum(0, ii+2-self.n_frames):ii+2]
            
            
            self.buffer.append({'current': {'obs': numpy.concatenate(act_obs), 
                                            'rew': r, 
                                            'crew': c, 
                                            'act': a, 
                                            'prob': p},
                                'next': {'obs': numpy.concatenate(act_obs_next), 
                                         'rew': rn, 
                                         'crew': cn, 
                                         'act': an, 
                                         'prob': pn}})

            
    def sample(self, n=100):
        idxs = numpy.random.choice(len(self.buffer),n)
        return [self.buffer[ii] for ii in idxs]
    
