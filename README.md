# rl-pong


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
-deterministic-ratio .1
-ent-coeff 0.01
-optimizer-value Adam
-optimizer-player Adam
-iw
-critic-aware 
./models/pong-ff-1fr-test
``

4 frame
``
python pong.py
-buffer-size 100000
-init-collect 1
-n-frames 4
-env Pong-ram-v0
-batch-size 1024
-update-every 1
-n-hid 128 
-n-value 100
-n-policy 100
-n-iter 5000
-nn ff
-n-simulators 8
-device cuda
-priority 0.
-store 1.
-ent-coeff 0.01
-optimizer-value Adam
-optimizer-player Adam
-iw
-critic-aware 
./models/pong-ff-4fr-test
``


## Assault-ram-v0

### feedforward policy

1 frame

``
python pong.py
-buffer-size 100000
-init-collect 1
-n-frames 1
-env Assault-ram-v0
-batch-size 1024
-update-every 1
-n-hid 64 
-n-value 100
-n-policy 100
-n-iter 5000
-nn ff
-n-simulators 8
-device cuda
-priority 0.
-store 1.
-ent-coeff 0.01
-optimizer-value Adam
-optimizer-player Adam
-iw
-critic-aware 
./models/assault-ff-1fr-test
``



### working policy

python ./pong_vid.py
-max-len 100000
-n-frames 4
-device cpu ./models/pong-ff-4fr_10191.th 
