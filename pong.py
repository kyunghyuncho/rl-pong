 torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam, SGD

import torch.multiprocessing as mp


from multiprocessing import Process, Queue
import queue

import numpy

import argparse
import glob


#from matplotlib import pyplot as plot

import copy
from time import sleep

import gym

from utils import Buffer, collect_one_episode, copy_params, avg_params

import ff
import conv

def simulator(idx, player_queue, episode_queue, args):

    print('Starting the simulator {}'.format(idx))

    env = gym.make(args.env)
    max_len = args.max_len
    discount_factor = args.discount_factor
    n_frames = args.n_frames
    device = 'cpu'

    if args.nn == "ff":
        player = ff.Player(n_in=128 * n_frames, n_hid=args.n_hid, n_out=6).to(device)
    elif args.nn == "conv":
        # create a policy
        player = conv.Player(n_frames=n_frames, n_hid=args.n_hid).to(device)
    else:
        raise Exception('Unknown type')

    while True:
        # first sync the player
        try:
            player_state = player_queue.get_nowait()
            player.load_state_dict(player_state)
        except queue.Empty:
            pass

        # run one episode
        player.eval()
        o_, r_, c_, a_, ap_, ret_ = collect_one_episode(env, player, max_len=max_len, discount_factor=discount_factor, n_frames=n_frames)
        episode_queue.put((o_, r_, c_, a_, ap_, ret_))

def main(args):

    # start simulators
    episode_q = Queue()
    player_qs = []
    simulators = []
    for si in range(args.n_simulators):
        player_qs.append(Queue())
        simulators.append(mp.Process(target=simulator, args=(si, player_qs[-1], episode_q, args,)))
        simulators[-1].start()

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
    n_hid = args.n_hid

    critic_aware = args.critic_aware

    update_every = args.update_every

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

    if args.nn == "ff":
        # create a policy
        player = ff.Player(n_in=128 * n_frames, n_hid=args.n_hid, n_out=6).to(args.device)

        # create a value estimator
        value = ff.Value(n_in=128 * n_frames, n_hid=args.n_hid).to(args.device)
        value_old = ff.Value(n_in=128 * n_frames, n_hid=args.n_hid).to(args.device)
    elif args.nn == "conv":
        # create a policy
        player = conv.Player(n_frames=n_frames, n_hid=args.n_hid).to(args.device)

        # create a value estimator
        value = conv.Value(n_frames, n_hid=args.n_hid).to(args.device)
        value_old = conv.Value(n_frames, n_hid=args.n_hid).to(args.device)
    else:
        raise Exception('Unknown type')

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

    # start simulators
    for si in range(args.n_simulators):
        player_qs[si].put(player.state_dict())

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
        #nc = n_collect
        #if ni == 0:
        #    nc = init_collect
        #for ci in range(nc):
        #    o_, r_, c_, a_, ap_, ret_ = collect_one_episode(env, player, max_len=max_len, discount_factor=discount_factor, n_frames=n_frames)
        #    replay_buffer.add(o_, r_, c_, a_, ap_)
        #    if ret == -numpy.Inf:
        #        ret = ret_
        #    else:
        #        ret = 0.9 * ret + 0.1 * ret_
        n_collected = 0
        while True:
            try:
                epi = episode_q.get_nowait()
                replay_buffer.add(epi[0], epi[1], epi[2], epi[3], epi[4])
                if ret == -numpy.Inf:
                    ret = ret_
                else:
                    ret = 0.9 * ret + 0.1 * epi[-1]
            except queue.Empty:
                if len(replay_buffer.buffer) > 0:
                    break
                else:
                    continue
            n_collected = n_collected + 1
            if numpy.mod(n_collected, 30) == 0 and len(replay_buffer.buffer) > 0:
                break
        print('Buffer length', len(replay_buffer.buffer))

        player.to('cpu')
        for si in range(args.n_simulators):
            player_qs[si].put(player.state_dict())
        player.to(args.device)
        
        # fit a value function
        # TD(1)
        value.train()
        for vi in range(n_value):
            if numpy.mod(vi, update_every) == 0:
                #print(vi, 'zeroing gradient')
                opt_player.zero_grad()
                opt_value.zero_grad()
            
            batch = replay_buffer.sample(batch_size)

            batch_x = torch.from_numpy(numpy.stack([ex['current']['obs'] for ex in batch]).astype('float32')).to(args.device)
            batch_r = torch.from_numpy(numpy.stack([ex['current']['rew'] for ex in batch]).astype('float32')).to(args.device)
            batch_xn = torch.from_numpy(numpy.stack([ex['next']['obs'] for ex in batch]).astype('float32')).to(args.device)
            pred_y = value(batch_x).squeeze()
            pred_next = value_old(batch_xn).squeeze().clone().detach()
            loss_ = ((batch_r + discount_factor * pred_next - pred_y) ** 2)
            
            batch_a = torch.from_numpy(numpy.stack([ex['current']['act'] for ex in batch]).astype('float32')[:,None]).to(args.device)
            batch_pi = player(batch_x, normalized=True)
            batch_q = torch.from_numpy(numpy.stack([ex['current']['prob'] for ex in batch]).astype('float32')).to(args.device)
            logp = torch.log(batch_pi.gather(1, batch_a.long())+1e-8)

            # (clipped) importance weight: 
            # because the policy may have changed since the tuple was collected.
            iw = torch.exp((logp.clone().detach() - torch.log(batch_q+1e-8)).clamp(max=0.))
        
            loss = iw * loss_
            
            loss = loss.mean()
            
            loss.backward()
            if numpy.mod(vi, update_every) == (update_every-1):
                #print(vi, 'making an update')
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
            if numpy.mod(pi, update_every) == 0:
                opt_player.zero_grad()
                opt_value.zero_grad()
            
            batch = replay_buffer.sample(batch_size)
            
            batch_x = torch.from_numpy(numpy.stack([ex['current']['obs'] for ex in batch]).astype('float32')).to(args.device)
            batch_xn = torch.from_numpy(numpy.stack([ex['next']['obs'] for ex in batch]).astype('float32')).to(args.device)
            batch_r = torch.from_numpy(numpy.stack([ex['current']['rew'] for ex in batch]).astype('float32')[:,None]).to(args.device)
            
            batch_v = value(batch_x).clone().detach()
            batch_vn = value(batch_xn).clone().detach()
            
            batch_a = torch.from_numpy(numpy.stack([ex['current']['act'] for ex in batch]).astype('float32')[:,None]).to(args.device)
            batch_q = torch.from_numpy(numpy.stack([ex['current']['prob'] for ex in batch]).astype('float32')).to(args.device)

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
            
            loss = (loss + ent_coeff * ent)
            
            if critic_aware:
                pred_y = value(batch_x).squeeze()
                pred_next = value_old(batch_xn).squeeze().clone().detach()
                critic_loss_ = -((batch_r + discount_factor * pred_next - pred_y) ** 2).clone().detach()
                critic_loss_ = torch.nn.Softmax(dim=0)(critic_loss_)

                loss = (loss * critic_loss_).sum()
            else:
                loss = loss.mean()
            
            loss.backward()
            if numpy.mod(pi, update_every) == (update_every-1):
                opt_player.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n-iter', type=int, default=1000)
    parser.add_argument('-n-collect', type=int, default=1)
    parser.add_argument('-init-collect', type=int, default=100)
    parser.add_argument('-n-value', type=int, default=150)
    parser.add_argument('-n-policy', type=int, default=150)
    parser.add_argument('-update-every', type=int, default=1)
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
    parser.add_argument('-nn', type=str, default='ff')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-cont', action="store_true", default=False)
    parser.add_argument('-critic-aware', action="store_true", default=False)
    parser.add_argument('-n-simulators', type=int, default=2)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args)
