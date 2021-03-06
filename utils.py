import gym

import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam, SGD

from skimage import transform

import numpy

import copy
from time import sleep

import heapq

def copy_params(from_, to_):
    for f_, t_ in zip(from_.parameters(), to_.parameters()):
        t_.data.copy_(f_.data)
    for f_, t_ in zip(from_.buffers(), to_.buffers()):
        t_.data.copy_(f_.data)
        
def avg_params(from_, to_, coeff=0.95):
    for f_, t_ in zip(from_.parameters(), to_.parameters()):
        t_.data.copy_(coeff * t_.data + (1.-coeff) * f_.data)
        
def normalize_obs(obs):
    if len(obs.shape) == 3:
        obs = obs.transpose(2, 0, 1)
    return obs.astype('float32') / 255.

# collect data
def collect_one_episode(env, player, max_len=50, discount_factor=0.9, 
                        deterministic=False, rendering=False, verbose=False, 
                        n_frames=1, queue=None, 
                        interval=10,
                        resize=None,
                        reload_interval=10,
                        reload_player=False, player_queue=None, reload_fun=None):
    episode = []

    observations = []

    rewards = []

    actions = []
    action_probs = []
    
    prev_obs = None

    obs = env.reset()
    
    for ml in range(max_len):
        if rendering:
            env.render()
            sleep(0.05)
            
        obs = normalize_obs(obs)
        if resize is not None:
            #print('before', obs.shape, type(obs), obs.dtype)
            obs = transform.resize(obs.transpose(1, 2, 0), resize).transpose(2, 0, 1)
            obs = obs.astype('float32')
            #print('after', obs.shape, type(obs), obs.dtype)

        if prev_obs == None:
            prev_obs = [obs * 0.] * n_frames
        
        prev_obs.pop(0)
        prev_obs.append(obs)
        if len(obs.shape) == 3:
            current_obs = numpy.concatenate(prev_obs)
        else:
            current_obs = numpy.concatenate(prev_obs)

        out_probs = player(torch.from_numpy(current_obs[None,:]).to(next(player.parameters()).device)).squeeze()
        
        if deterministic:
            action = numpy.argmax(out_probs.to('cpu').data.numpy())
            if verbose:
                print(out_probs, action)
        else:
            act_dist = Categorical(out_probs.clamp(min=1e-5,max=1.-1e-5))
            try:
                action = act_dist.sample().item()
            except Exception:
                print('!!!!!!!!!!!!!', out_probs)
        action_prob = out_probs[action].item()

        observations.append(obs)
        actions.append(action)
        action_probs.append(action_prob)

        obs, reward, done, info = env.step(action)
        if deterministic and verbose:
            print(reward, done)
        
        rewards.append(reward)

        if queue is not None:
            if numpy.mod(ml+1, interval) == 0:
                queue.put((observations, 
                                   rewards, 
                                   actions, 
                                   action_probs))
                observations = []
                rewards = []
                actions = []
                action_probs = []

            if numpy.mod(ml+1, reload_interval) == 0:
                if reload_player and player_queue is not None:
                    try:
                        player_state = player_queue.get_nowait()

                        if type(player_state) == str and player_state == "END":
                            break

                        for p, c in zip(player.parameters(), player_state[:n_params]):
                            p.data.copy_(torch.from_numpy(c))
                        for p, c in zip(player.buffers(), player_state[n_params:]):
                            p.data.copy_(torch.from_numpy(c))
                        if player_queue.qsize() > 0:
                            print('Simulator {} queue overflowing'.format(idx))
                    except queue.Empty:
                        pass


    return observations, \
            rewards, \
            actions, \
            action_probs, \
            numpy.sum(rewards)

# simple implementation of FIFO-based replay buffer
class Buffer:
    def __init__(self, max_items=10000, 
                 n_frames=1, priority_ratio=0.9, store_ratio=1.):
        self.max_items = max_items
        self.n_frames = n_frames
        self.priority_ratio = priority_ratio
        self.store_ratio = store_ratio

        self.buffer = []
        self.priority_buffer = []
        
    def add(self, observations, rewards, actions, action_probs):
        n_priority = 0
        n_rand = 0

        for ii, (o, r, a, p, on, rn, an, pn) in enumerate(zip(observations[:-1], rewards[:-1], 
                                                                     actions[:-1], action_probs[:-1],
                                             observations[1:], rewards[1:], actions[1:], action_probs[1:])):
            if numpy.random.rand() > self.store_ratio:
                continue

            act_obs = [numpy.zeros(o.shape).astype('float32')] * numpy.maximum(0,(self.n_frames-ii-1))
            act_obs = act_obs + observations[numpy.maximum(0, ii-self.n_frames+1):ii+1]
#             print(ii, len(act_obs))
            
            act_obs_next = [numpy.zeros(o.shape).astype('float32')] * numpy.maximum(0,(self.n_frames-ii-2))
            act_obs_next = act_obs_next + observations[numpy.maximum(0, ii+2-self.n_frames):ii+2]
            
            if numpy.random.rand() <= self.priority_ratio:
                n_priority = n_priority + 1
                heapq.heappush(self.priority_buffer, SAR({'obs': numpy.concatenate(act_obs), 
                                                'rew': r, 
                                                'act': a, 
                                                'prob': p},
                                    {'obs': numpy.concatenate(act_obs_next), 
                                             'rew': rn, 
                                             'act': an, 
                                             'prob': pn}))
            else:
                n_rand = n_rand + 1
                self.buffer.append(SAR({'obs': numpy.concatenate(act_obs), 
                                                'rew': r, 
                                                'act': a, 
                                                'prob': p},
                                    {'obs': numpy.concatenate(act_obs_next), 
                                             'rew': rn, 
                                             'act': an, 
                                             'prob': pn}))

        new_n = len(self.buffer) + len(self.priority_buffer)
        if new_n > self.max_items:
            new_n = new_n - self.max_items

            new_n_priority = int(numpy.round(new_n * self.priority_ratio))
            new_n_rand = new_n - new_n_priority

            n_removed_rand = numpy.minimum(new_n_rand, len(self.buffer))
            if n_removed_rand > 1:
                idxs = numpy.random.choice(len(self.buffer),n_removed_rand,replace=False)
                for index in sorted(idxs, reverse=True):
                    del self.buffer[index]

            n_removed_priority = numpy.minimum(new_n_priority, len(self.priority_buffer))
            if n_removed_priority > 1:
                for ni in range(n_removed_priority):
                    x = heapq.heappop(self.priority_buffer)


            
    def sample(self, n=100):
        n_samples_priority = int(numpy.round(n * self.priority_ratio))
        n_samples_rand = n - n_samples_priority

        idxs = numpy.random.choice(len(self.buffer),numpy.minimum(n_samples_rand, len(self.buffer)),replace=False)
        rand_samples = [self.buffer[ii] for ii in idxs]

        if n_samples_priority < 1:
            priority_samples = []
        else:
            idxs = numpy.random.choice(len(self.priority_buffer),numpy.minimum(n_samples_priority, len(self.priority_buffer)),replace=False)
            priority_samples = [self.priority_buffer[ii] for ii in idxs]

        return rand_samples + priority_samples

    
class SAR:
    def __init__(self, current_, next_):
        self.current_ = current_
        self.next_ = next_

    def __lt__(self, other):
        return self.current_['rew'] < other.current_['rew']

