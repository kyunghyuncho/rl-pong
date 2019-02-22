# rl-pong


## Pong-ram-v0

### feedforward policy

1 frame

`python pong.py -buffer-size 5000 -init-collect 1 -n-frames 1 -env Pong-ram-v0 -batch-size 512 -n-hid 64  -n-value 50 -n-policy 50 -n-iter 5000 -nn ff ./models/pong-ff`

4 frames

`python pong.py -buffer-size 5000 -init-collect 1 -n-frames 4 -env Pong-ram-v0 -batch-size 512 -n-hid 64  -n-value 50 -n-policy 50 -n-iter 5000 -nn ff ./models/pong-ff-4fr`



### working policy

python ./pong_vid.py -max-len 100000 -n-frames 4 -device cpu ./models/pong-ff-4fr_10191.th 
