# rl-pong


## Pong-ram-v0

### feedforward policy

1 frame

``
python pong.py
-buffer-size 15000
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
-n-simulators 4
-device cuda
-priority 0.
-ent-coeff 10.
-grad-clip 1.
-optimizer-value Adam
-optimizer-player Adam ./models/pong-ff-1fr-test
``



### working policy

python ./pong_vid.py
-max-len 100000
-n-frames 4
-device cpu ./models/pong-ff-4fr_10191.th 
