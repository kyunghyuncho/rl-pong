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

There are a few bells and whistles implemented, but none of them seems to be
useful anyway. I'll list them here with brief descriptions:

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

## Pong-ram-v0

### feedforward policy

1 frame

``
python pong.py
-buffer-size 100000
-init-collect 1
-n-frames 1
-env Pong-ram-v0
-batch-size 1024
-update-every 1
-n-hid 64
-n-value 100
-n-policy 100
-n-iter 5000
-nn ff
-n-simulators 8
-device cuda
-priority-ratio 0.
-store-ratio 1.
-deterministic-ratio 0.
-ent-coeff 0.01
-ent-factor 0.01
-optimizer-value Adam
-optimizer-player Adam
-iw
-grad-clip 1.
-lr 0.0001
-lr-factor 0.01
-critic-aware 
./models/pong-ff-1fr-test
``

4 frame
``
``

### convolutional policy

1 frame

``
``

### playing with the policy

``
python ./pong_vid.py
-max-len 100000
-n-frames 4
-device cpu ./models/pong-ff-4fr_10191.th 
``




