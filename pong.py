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
import time

import gym

from utils import Buffer, collect_one_episode, copy_params, avg_params

import ff
import conv

def simulator(idx, player_queue, episode_queue, args, valid=False):

    seed = 0
    for ii in range(idx):
        seed = numpy.random.randint(0, numpy.iinfo(int).max)

    torch.manual_seed(seed)

    if valid:
        idx = "valid"

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
    if args.resize is None:
        resize = None
    else:
        resize = [int(v) for v in args.resize.split(',')]

    while True:
        # first sync the player if possible
        try:
            player_state = player_queue.get_nowait()

            if type(player_state) == str and player_state == "END":
                break

            for p, c in zip(player.parameters(), player_state[:n_params]):
                p.data.copy_(c)
            for p, c in zip(player.buffers(), player_state[n_params:]):
                p.data.copy_(c)
            if player_queue.qsize() > 0:
                print('Simulator {} queue overflowing'.format(idx))
        except queue.Empty:
            pass

        # run one episode
        for m in player.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

        if valid:
            _, _, _, _, ret_ = collect_one_episode(env, 
                    player, max_len=max_len, discount_factor=discount_factor, 
                    n_frames=n_frames, 
                    deterministic=True,
                    queue=None, interval=-1,
                    resize=resize)
            episode_queue.put(ret_)
        else:
            o_, r_, a_, ap_, ret_ = collect_one_episode(env, 
                    player, max_len=max_len, discount_factor=discount_factor, 
                    n_frames=n_frames, 
                    deterministic=numpy.random.rand() <= args.deterministic_ratio,
                    queue=episode_queue, interval=args.collect_interval,
                    resize=resize)
            episode_queue.put((o_, r_, a_, ap_, ret_))

def main(args):

    torch.manual_seed(args.seed)

    # start simulators
    mp.set_start_method('spawn')

    episode_q = Queue()
    player_qs = []
    simulators = []
    for si in range(args.n_simulators):
        player_qs.append(Queue())
        simulators.append(mp.Process(target=simulator, args=(si, player_qs[-1], episode_q, args, False,)))
        simulators[-1].start()

    return_q = Queue()
    valid_q = Queue()
    valid_simulator = mp.Process(target=simulator, args=(args.n_simulators, valid_q, return_q, args, True,))
    valid_simulator.start()

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
    max_collected_frames = args.max_collected_frames

    clip_coeff = args.grad_clip
    ent_coeff = args.ent_coeff
    discount_factor = args.discount_factor

    value_loss = -numpy.Inf
    entropy = -numpy.Inf
    valid_ret = -numpy.Inf
    ess = -numpy.Inf
    n_collected_frames = 0

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

        for m in player.parameters():
            m.data.normal_(0., 0.01)
        for m in value.parameters():
            m.data.normal_(0., 0.01)
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
        n_collected_frames = checkpoint['n_collected_frames']

    copy_params(value, value_old)
    if args.player_coeff > 0.:
        copy_params(player, player_old)

    # start simulators
    player.to('cpu')
    copy_params(player, player_copy)
    for si in range(args.n_simulators):
        player_qs[si].put([copy.deepcopy(p.data) for p in player_copy.parameters()]+
                          [copy.deepcopy(p.data) for p in player_copy.buffers()])
    valid_q.put([copy.deepcopy(p.data) for p in player_copy.parameters()]+
                [copy.deepcopy(p.data) for p in player_copy.buffers()])
    player.to(args.device)

    if args.device == 'cuda':
        torch.set_num_threads(1)

    initial = True
    pre_filled = 0
        
    for ni in range(n_iter):
        try:
            # re-initialize optimizers
            opt_player = eval(args.optimizer_player)(player.parameters(), 
                                                     lr=args.lr, weight_decay=args.l2)
            opt_value = eval(args.optimizer_value)(value.parameters(), 
                                                   lr=args.lr, weight_decay=args.l2)

            if not initial:
                lr = args.lr / (1 + (ni-pre_filled+1) * args.lr_factor)
                ent_coeff = args.ent_coeff / (1 + (ni-pre_filled+1) * args.ent_factor)
                print('lr', lr, 'ent_coeff', ent_coeff)

                for param_group in opt_player.param_groups:
                    param_group['lr'] = lr
                for param_group in opt_value.param_groups:
                    param_group['lr'] = lr

            if numpy.mod((ni-pre_filled+1), save_iter) == 0:
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
                    'n_collected_frames': n_collected_frames, 
                    }, '{}_{}.th'.format(args.saveto,(ni-pre_filled+1)+offset+1))

            player.eval()

            ret_ = -numpy.Inf
            while True:
                try:
                    ret_ = return_q.get_nowait()
                except queue.Empty:
                    break
            if ret_ != -numpy.Inf:
                return_history.append(ret_)
                if valid_ret == -numpy.Inf:
                    valid_ret = ret_
                else:
                    valid_ret = 0.9 * valid_ret + 0.1 * ret_
                print('Valid run', ret_, valid_ret)

            #st = time.time()

            player.to('cpu')
            copy_params(player, player_copy)
            for si in range(args.n_simulators):
                while True:
                    try:
                        # empty the queue, as the new one has arrived
                        player_qs[si].get_nowait()
                    except queue.Empty:
                        break
                
                player_qs[si].put([copy.deepcopy(p.data) for p in player_copy.parameters()]+
                                  [copy.deepcopy(p.data) for p in player_copy.buffers()])
            while True:
                try:
                    # empty the queue, as the new one has arrived
                    valid_q.get_nowait()
                except queue.Empty:
                    break
            valid_q.put([copy.deepcopy(p.data) for p in player_copy.parameters()]+
                        [copy.deepcopy(p.data) for p in player_copy.buffers()])

            player.to(args.device)

            #print('model push took', time.time()-st)

            #st = time.time()

            n_collected_frames_ = 0 
            while True:
                try:
                    epi = episode_q.get_nowait()
                    replay_buffer.add(epi[0], epi[1], epi[2], epi[3])
                    n_collected_frames_ = n_collected_frames_ + len(epi[0])
                except queue.Empty:
                    break
                if n_collected_frames_ >= max_collected_frames \
                        and (len(replay_buffer.buffer) + len(replay_buffer.priority_buffer)) > 0:
                    break
            n_collected_frames = n_collected_frames + n_collected_frames_

            if len(replay_buffer.buffer) + len(replay_buffer.priority_buffer) < 1:
                continue

            if len(replay_buffer.buffer) + len(replay_buffer.priority_buffer) < args.initial_buffer:
                if initial:
                    print('Pre-filling the buffer...', 
                            len(replay_buffer.buffer) + len(replay_buffer.priority_buffer))
                    continue
            else:
                if initial:
                    pre_filled = ni
                    initial = False

            #print('collection took', time.time()-st)

            #print('Buffer size', len(replay_buffer.buffer) + len(replay_buffer.priority_buffer))

            # fit a value function
            # TD(0)
            #st = time.time()

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
            
            if numpy.mod((ni-pre_filled+1), disp_iter) == 0:
                print('# frames', n_collected_frames, 
                      'value_loss', value_loss, 
                      'entropy', -entropy,
                      'ess', ess)

            #print('value update took', time.time()-st)

            
            # fit a policy
            #st = time.time()

            value.eval()
            player.train()
            if args.player_coeff > 0.:
                player_old.eval()

            for pi in range(n_policy):
                if numpy.mod(pi, update_every) == 0:
                    opt_player.zero_grad()
                    opt_value.zero_grad()
                
                #st = time.time()

                batch = replay_buffer.sample(batch_size)

                #print('batch collection took', time.time()-st)
                
                #st = time.time()

                #batch_x = [ex.current_['obs'] for ex in batch]
                #batch_xn = [ex.next_['obs'] for ex in batch]
                #batch_r = [ex.current_['rew'] for ex in batch]

                #print('list construction took', time.time()-st)

                #st = time.time()

                batch_x = numpy.zeros(tuple([len(batch)] + list(batch[0].current_['obs'].shape)), dtype='float32')
                batch_xn = numpy.zeros(tuple([len(batch)] + list(batch[0].current_['obs'].shape)), dtype='float32')
                batch_r = numpy.zeros((len(batch)), dtype='float32')[:, None]

                for ei, ex in enumerate(batch):
                    batch_x[ei,:] = ex.current_['obs']
                    batch_xn[ei,:] = ex.next_['obs']
                    batch_r[ei,0] = ex.current_['rew']

                #batch_x = numpy.stack(batch_x).astype('float32')
                #batch_xn = numpy.stack(batch_xn).astype('float32')
                #batch_r = numpy.stack(batch_r).astype('float32')[:,None]

                #print('batch stack for value took', time.time()-st)

                #st = time.time()

                batch_x = torch.from_numpy(batch_x).to(args.device)
                batch_xn = torch.from_numpy(batch_xn).to(args.device)
                batch_r = torch.from_numpy(batch_r).to(args.device)

                #print('batch push for value took', time.time()-st)

                #st = time.time()

                batch_v = value(batch_x).clone().detach()
                batch_vn = value(batch_xn).clone().detach()

                #print('value forward pass took', time.time()-st)
                
                #st = time.time()

                batch_a = torch.from_numpy(numpy.stack([ex.current_['act'] for ex in batch]).astype('float32')[:,None]).to(args.device)
                batch_q = torch.from_numpy(numpy.stack([ex.current_['prob'] for ex in batch]).astype('float32')).to(args.device)

                batch_pi = player(batch_x)
                logp = torch.log(batch_pi.gather(1, batch_a.long())+1e-8)

                if args.player_coeff > 0.:
                    batch_pi_old = player_old(batch_x).clone().detach()

                #print('policy computation took', time.time()-st)
                
                #st = time.time()

                # entropy regularization
                ent = -(batch_pi * torch.log(batch_pi+1e-8)).sum(1)
                if entropy == -numpy.Inf:
                    entropy = ent.mean().item()
                else:
                    entropy = 0.9 * entropy + 0.1 * ent.mean().item()

                #print('entropy computation took', time.time()-st)
                
                #st = time.time()

                # advantage: r(s,a) + \gamma * V(s') - V(s)
                adv = batch_r + discount_factor * batch_vn - batch_v
                #adv = adv / adv.abs().max().clamp(min=1.)
                
                loss = -(adv * logp).squeeze()

                loss = loss - ent_coeff * ent

                #print('basic loss computation took', time.time()-st)

                #st = time.time()

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

                #print('importance weighting took', time.time()-st)
                
                if critic_aware:
                    #st = time.time()

                    pred_y = value(batch_x).squeeze()
                    pred_next = value(batch_xn).squeeze()
                    critic_loss_ = -((batch_r.squeeze() + discount_factor * pred_next - pred_y) ** 2).clone().detach()

                    critic_loss_ = torch.exp(critic_loss_)
                    loss = loss * critic_loss_

                    #print('critic aware weighting took', time.time()-st)

                loss = loss.mean()

                if args.player_coeff > 0.:
                    #st = time.time()

                    loss_old = -(batch_pi_old * torch.log(batch_pi + 1e-8)).sum(1).mean()
                    loss = (1.-args.player_coeff) * loss + args.player_coeff * loss_old

                    #print('player interpolation took', time.time()-st)

                #st = time.time()
                loss.backward()
                if numpy.mod(pi, update_every) == (update_every-1):
                    if clip_coeff > 0.:
                        nn.utils.clip_grad_norm_(player.parameters(), clip_coeff)
                    opt_player.step()
                #print('backward computation and update took', time.time()-st)

            if args.player_coeff > 0.:
                copy_params(player, player_old)

            ##print('policy update took', time.time()-st)

        except KeyboardInterrupt:
            print('Terminating...')
            break

    for si in range(args.n_simulators):
        player_qs[si].put("END")

    print('Waiting for the simulators...')

    for si in range(args.n_simulators):
        simulators[-1].join()

    print('Done')



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
    parser.add_argument('-max-collected-frames', type=int, default=1000)
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
    parser.add_argument('-initial-buffer', type=int, default=0)
    parser.add_argument('-collect-interval', type=int, default=10)
    parser.add_argument('-resize', type=str, default=None)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args)
