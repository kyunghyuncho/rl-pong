import torch

from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam, SGD, ASGD

import torch.multiprocessing as mp

import os


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

    seed = 0
    for ii in range(idx):
        seed = numpy.random.randint(0, numpy.iinfo(int).max)

    torch.manual_seed(seed)

    print('Starting the simulator {}'.format(idx))

    device = 'cpu'
    torch.device('cpu')
    torch.set_num_threads(args.n_cores)

    env = gym.make(args.env)
    max_len = args.max_len
    discount_factor = args.discount_factor
    n_frames = args.n_frames

    if args.nn == "ff":
        player = ff.Player(n_in=128 * n_frames, n_hid=args.n_hid, n_out=6).to(device)
    elif args.nn == "conv":
        # create a policy
        player = conv.Player(n_frames=n_frames, n_hid=args.n_hid).to(device)
    else:
        raise Exception('Unknown type')

    n_params = len(list(player.parameters()))

    while True:
        # first sync the player if possible
        try:
            player_state = player_queue.get_nowait()
            for p, c in zip(player.parameters(), player_state[:n_params]):
                #p.data.copy_(c.data)
                p.data.copy_(torch.from_numpy(c))
            for p, c in zip(player.buffers(), player_state[n_params:]):
                #p.data.copy_(c.data)
                p.data.copy_(torch.from_numpy(c))
            if player_queue.qsize() > 0:
                print('Simulator {} queue overflowing'.format(idx))
        except queue.Empty:
            pass

        # run one episode
        player.eval()
        o_, r_, c_, a_, ap_, ret_ = collect_one_episode(env, 
                player, max_len=max_len, discount_factor=discount_factor, 
                n_frames=n_frames, 
                deterministic=numpy.random.rand() <= args.deterministic_ratio)
        episode_queue.put((o_, r_, c_, a_, ap_, ret_))
        #print('Simulator {} episode done'.format(idx))

def main(args):

    torch.manual_seed(args.seed)

    # start simulators
    mp.set_start_method('spawn')

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
    replay_buffer = Buffer(max_items=args.buffer_size, 
                           n_frames=n_frames,
                           priority_ratio=args.priority_ratio,
                           store_ratio=args.store_ratio)

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
    max_episodes = args.max_episodes

    clip_coeff = args.grad_clip
    ent_coeff = args.ent_coeff
    discount_factor = args.discount_factor

    value_loss = -numpy.Inf
    ret = -numpy.Inf
    entropy = -numpy.Inf
    valid_ret = -numpy.Inf
    ess = -numpy.Inf
    n_plays = 0

    offset = 0

    return_history = []

    if args.nn == "ff":
        # create a policy
        player = ff.Player(n_in=128 * n_frames, n_hid=args.n_hid, n_out=6).to(args.device)
        if args.player_coeff > 0.:
            player_old = ff.Player(n_in=128 * n_frames, n_hid=args.n_hid, n_out=6).to(args.device)
        player_copy = ff.Player(n_in=128 * n_frames, n_hid=args.n_hid, n_out=6).to('cpu')

        # create a value estimator
        value = ff.Value(n_in=128 * n_frames, n_hid=args.n_hid).to(args.device)
        value_old = ff.Value(n_in=128 * n_frames, n_hid=args.n_hid).to(args.device)
    elif args.nn == "conv":
        # create a policy
        player = conv.Player(n_frames=n_frames, n_hid=args.n_hid).to(args.device)
        if args.player_coeff > 0.:
            player_old = conv.Player(n_frames=n_frames, n_hid=args.n_hid).to(args.device)
        player_copy = conv.Player(n_frames=n_frames, n_hid=args.n_hid).to('cpu')

        # create a value estimator
        value = conv.Value(n_frames, n_hid=args.n_hid).to(args.device)
        value_old = conv.Value(n_frames, n_hid=args.n_hid).to(args.device)
    else:
        raise Exception('Unknown type')

    for m in player.parameters():
        m.data.normal_(0., 0.01)
    for m in value.parameters():
        m.data.normal_(0., 0.01)


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
        n_plays = checkpoint['n_plays']

    copy_params(value, value_old)

    # start simulators
    player.to('cpu')
    copy_params(player, player_copy)
    for si in range(args.n_simulators):
        #player_qs[si].put(copy.deepcopy(list(player_copy.parameters())))
        player_qs[si].put([copy.deepcopy(p.data.numpy()) for p in player_copy.parameters()]+
                          [copy.deepcopy(p.data.numpy()) for p in player_copy.buffers()])
    player.to(args.device)

    if args.device == 'cuda':
        torch.set_num_threads(1)

    for ni in range(n_iter):
        # re-initialize optimizers
        opt_player = eval(args.optimizer_player)(player.parameters(), 
                                                 lr=args.lr, weight_decay=args.l2)
        opt_value = eval(args.optimizer_value)(value.parameters(), 
                                               lr=args.lr, weight_decay=args.l2)

        lr = args.lr / (1 + ni * args.lr_factor)
        ent_coeff = args.ent_coeff / (1 + ni * args.ent_factor)
        print('lr', lr, 'ent_coeff', ent_coeff)

        for param_group in opt_player.param_groups:
            param_group['lr'] = lr
        for param_group in opt_value.param_groups:
            param_group['lr'] = lr

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
                'n_plays': n_plays, 
                }, '{}_{}.th'.format(args.saveto,ni+offset+1))

        player.eval()

        if numpy.mod(ni, val_iter) == 0:
            _, _, _, _, _, ret_ = collect_one_episode(env, player, 
                    max_len=max_len, 
                    deterministic=True, 
                    n_frames=n_frames)
            return_history.append(ret_)
            if valid_ret == -numpy.Inf:
                valid_ret = ret_
            else:
                valid_ret = 0.9 * valid_ret + 0.1 * ret_
            print('Valid run', ret_, valid_ret)

        player.to('cpu')
        copy_params(player, player_copy)
        for si in range(args.n_simulators):
            while True:
                try:
                    # empty the queue, as the new one has arrived
                    player_qs[si].get_nowait()
                except queue.Empty:
                    break
            
            player_qs[si].put([copy.deepcopy(p.data.numpy()) for p in player_copy.parameters()]+
                              [copy.deepcopy(p.data.numpy()) for p in player_copy.buffers()])
        player.to(args.device)
        
        n_collected = 0
        while True:
            try:
                epi = episode_q.get_nowait()
                replay_buffer.add(epi[0], epi[1], epi[2], epi[3], epi[4])
                n_plays = n_plays + 1
                if ret == -numpy.Inf:
                    ret = ret_
                else:
                    ret = 0.9 * ret + 0.1 * epi[-1]
            except queue.Empty:
                break
            n_collected = n_collected + 1
            if n_collected >= max_episodes \
                    and (len(replay_buffer.buffer) + len(replay_buffer.priority_buffer)) > 0:
                break
        #print('Buffer length', len(replay_buffer.buffer), len(replay_buffer.priority_buffer))

        if len(replay_buffer.buffer) + len(replay_buffer.priority_buffer) < 1:
            continue

        # fit a value function
        # TD(0)
        value.train()
        for vi in range(n_value):
            if numpy.mod(vi, update_every) == 0:
                #print(vi, 'zeroing gradient')
                opt_player.zero_grad()
                opt_value.zero_grad()
            
            batch = replay_buffer.sample(batch_size)

            batch_x = torch.from_numpy(numpy.stack([ex.current_['obs'] for ex in batch]).astype('float32')).to(args.device)
            batch_r = torch.from_numpy(numpy.stack([ex.current_['rew'] for ex in batch]).astype('float32')).to(args.device)
            batch_xn = torch.from_numpy(numpy.stack([ex.next_['obs'] for ex in batch]).astype('float32')).to(args.device)
            pred_y = value(batch_x)
            pred_next = value_old(batch_xn).clone().detach()
            batch_pi = player(batch_x)

            loss_ = ((batch_r + discount_factor * pred_next.squeeze() - pred_y.squeeze()) ** 2)
            
            batch_a = torch.from_numpy(numpy.stack([ex.current_['act'] for ex in batch]).astype('float32')[:,None]).to(args.device)
            batch_q = torch.from_numpy(numpy.stack([ex.current_['prob'] for ex in batch]).astype('float32')).to(args.device)
            logp = torch.log(batch_pi.gather(1, batch_a.long())+1e-8)

            # (clipped) importance weight: 
            # because the policy may have changed since the tuple was collected.
            log_iw = logp.squeeze().clone().detach() - torch.log(batch_q.squeeze()+1e-8)
            ess_ = torch.exp(-torch.logsumexp(2 * log_iw, dim=0)).item()
            iw = torch.exp(log_iw.clamp(max=0.))

            if args.iw:
                loss = iw * loss_
            else:
                loss = loss_

            loss = loss.mean()
            
            loss.backward()

            if numpy.mod(vi, update_every) == (update_every-1):
                #print(vi, 'making an update')
                if clip_coeff > 0.:
                    nn.utils.clip_grad_norm_(value.parameters(), clip_coeff)
                opt_value.step()
        
        copy_params(value, value_old)
            
        if value_loss < 0.:
            value_loss = loss_.mean().item()
        else:
            value_loss = 0.9 * value_loss + 0.1 * loss_.mean().item()
        
        if numpy.mod(ni, disp_iter) == 0:
            print('# plays', n_plays, 
                  'return', ret, 
                  'value_loss', value_loss, 
                  'entropy', -entropy,
                  'ess', ess)
        
        # fit a policy
        value.eval()
        player.train()
        if args.player_coeff > 0.:
            player_old.train()

        for pi in range(n_policy):
            if numpy.mod(pi, update_every) == 0:
                opt_player.zero_grad()
                opt_value.zero_grad()
            
            batch = replay_buffer.sample(batch_size)
            
            batch_x = torch.from_numpy(numpy.stack([ex.current_['obs'] for ex in batch]).astype('float32')).to(args.device)
            batch_xn = torch.from_numpy(numpy.stack([ex.next_['obs'] for ex in batch]).astype('float32')).to(args.device)
            batch_r = torch.from_numpy(numpy.stack([ex.current_['rew'] for ex in batch]).astype('float32')[:,None]).to(args.device)

            batch_v = value(batch_x).clone().detach()
            batch_vn = value(batch_xn).clone().detach()
            
            batch_a = torch.from_numpy(numpy.stack([ex.current_['act'] for ex in batch]).astype('float32')[:,None]).to(args.device)
            batch_q = torch.from_numpy(numpy.stack([ex.current_['prob'] for ex in batch]).astype('float32')).to(args.device)

            batch_pi = player(batch_x)
            logp = torch.log(batch_pi.gather(1, batch_a.long())+1e-8)

            if args.player_coeff > 0.:
                batch_pi_old = player_old(batch_x)
            
            # entropy regularization
            ent = -(batch_pi * torch.log(batch_pi+1e-8)).sum(1)
            if entropy == -numpy.Inf:
                entropy = ent.mean().item()
            else:
                entropy = 0.9 * entropy + 0.1 * ent.mean().item()
            
            
            # advantage: r(s,a) + \gamma * V(s') - V(s)
            adv = batch_r + discount_factor * batch_vn - batch_v
            #adv = adv / adv.abs().max().clamp(min=1.)
            
            loss = -(adv * logp).squeeze()

            loss = loss - ent_coeff * ent

            # (clipped) importance weight: 
            log_iw = logp.squeeze().clone().detach() - torch.log(batch_q+1e-8)
            iw = torch.exp(log_iw.clamp(max=0.))

            ess_ = torch.exp(-torch.logsumexp(2 * log_iw, dim=0)).item()
            if ess == -numpy.Inf:
                ess = ess_
            else:
                ess = 0.9 * ess + 0.1 * ess_
        
            if args.iw:
                loss = iw * loss
            else:
                loss = loss
            
            if critic_aware:
                pred_y = value(batch_x).squeeze()
                pred_next = value_old(batch_xn).squeeze()
                critic_loss_ = -((batch_r.squeeze() + discount_factor * pred_next - pred_y) ** 2).clone().detach()

                critic_loss_ = torch.exp(critic_loss_)
                loss = loss * critic_loss_

            loss = loss.mean()

            if args.player_coeff > 0.:
                loss_old = -(batch_pi_old * torch.log(batch_pi + 1e-8)).sum(1).mean()
                loss = loss + args.player_coeff * loss_old

            loss.backward()
            if numpy.mod(pi, update_every) == (update_every-1):
                if clip_coeff > 0.:
                    nn.utils.clip_grad_norm_(player.parameters(), clip_coeff)
                opt_player.step()

        if args.player_coeff > 0.:
            copy_params(player, player_old)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=1234)
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
    parser.add_argument('-ent-factor', type=float, default=0.)
    parser.add_argument('-discount-factor', type=float, default=0.95)
    parser.add_argument('-grad-clip', type=float, default=1.)
    parser.add_argument('-n-hid', type=int, default=256)
    parser.add_argument('-buffer-size', type=int, default=50000)
    parser.add_argument('-n-frames', type=int, default=1)
    parser.add_argument('-max-episodes', type=int, default=1000)
    parser.add_argument('-env', type=str, default='Pong-ram-v0')
    parser.add_argument('-nn', type=str, default='ff')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-optimizer-player', type=str, default='ASGD')
    parser.add_argument('-optimizer-value', type=str, default='Adam')
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-lr-factor', type=float, default=0.)
    parser.add_argument('-l2', type=float, default=0.)
    parser.add_argument('-priority-ratio', type=float, default=0.)
    parser.add_argument('-store-ratio', type=float, default=1.)
    parser.add_argument('-cont', action="store_true", default=False)
    parser.add_argument('-critic-aware', action="store_true", default=False)
    parser.add_argument('-iw', action="store_true", default=False)
    parser.add_argument('-n-simulators', type=int, default=2)
    parser.add_argument('-n-cores', type=int, default=1)
    parser.add_argument('-deterministic-ratio', type=float, default=0.)
    parser.add_argument('-player-coeff', type=float, default=0.)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args)
