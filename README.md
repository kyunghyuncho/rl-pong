# rl-pong

This repository demonstrates an actor-critic algoithm implemented using function
approximation from scratch. The critic approximates the value function using a
deep neural network (either `ff` or `conv`) and is trained using a SARSA update.
The actor looks exactly like the critic except that the output is a categorical
distribution over the action space, and is trained using REINFORCE with a
baseline provided by the critic. Both the critic and actors are trained in an
off-policy manner using importance sampling with the importance weight capped at
1 (similarly to Retrace(\lambda) with \lambda=1). Critic-aware leaning
is used for the actor (https://arxiv.org/abs/1702.02429). The negative entropy
of the actor's output is penalized with a small coefficient.

There are a few bells and whistles implemented, but not all of them seems to be
useful anyway. I'll list them here with brief descriptions:

* -player-coeff: this should be used. this minimizes the KL divergence between
  the previous policy and the new policy. <= necessary
* -iw: this should be used. without this flag, it will not correct the gradient
  with the importance weights. <= necessary
* -critic-aware: this should be used. probably not necessary, but it seems to
  help at least from experiences. <= necessary
* -update-every: it accumulates the gradients over this many forward-backwad
  procedures. <= necessary with memory limitation
* -n-value, -n-policy: the number of updates taken per iteration. i thought of
  this as a quantity similar to an epoch in supervised learning. <= robust
* -n-simulators, -n-cores: it collects episodes from multiple processes, and
  each process runs on n-cores many cpu cores. <= depends on the resource
* -priority-ratio: a portion of the experience replay buffer can be reserved for
  highly rewarding tuples (i.e., tuples with high return.) <= not necessary
* -store-ratio: if your buffer size is small, you may not want to store all the
  tuples from a new experiences, as it will decrease the diversity of the tuples
  and make them correlated heavily. use this to store only a small portion. <=
  not necessary

what needs to be implemented is to let the simulator pushes tuples in the middle
of the episode without having to wait until it's terminated.

## Pong-ram-v0

### feedforward policy

4 frame
``
python pong.py -buffer-size 100 -init-collect 1 -n-frames 4 -env Pong-v0 -batch-size 1024 -update-every 1 -n-hid 128 -n-value 50 -n-policy 50 -n-iter 5000 -nn ff -n-simulators 5 -device cuda -priority-ratio 0.  -store-ratio 1.  -deterministic-ratio 0.  -ent-coeff 0. -ent-factor 0. -optimizer-value Adam -optimizer-player Adam -iw -critic-aware  -lr 0.0001 -lr-factor 0.001 -player-coeff .5 -collect-interval 10 ./models/pong-ff-4fr-test
``

### convolutional policy

4 frame

``
python pong.py -buffer-size 100 -init-collect 1 -n-frames 4 -env Pong-v0 -batch-size 1024 -update-every 1 -n-hid 128 -n-value 50 -n-policy 50 -n-iter 5000 -nn conv -n-simulators 5 -device cuda -priority-ratio 0.  -store-ratio 1.  -deterministic-ratio 0.  -ent-coeff 0. -ent-factor 0. -optimizer-value Adam -optimizer-player Adam -iw -critic-aware  -lr 0.0001 -lr-factor 0.001 -player-coeff .5 -collect-interval 10 -resize 48,48 ./models/pong-conv-4fr-test
``

### playing with the policy

``
python ./pong_vid.py
-max-len 100000
-n-frames 4
-device cpu ./models/pong-ff-4fr_10191.th 
``




