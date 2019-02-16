import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam, SGD


class ResLinear(nn.Module):
    def __init__(self, n_in, n_out, act=nn.ReLU()):
        super(ResLinear, self).__init__()
        self.act = act
        self.linear = nn.Linear(n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        
        assert(n_in == n_out)
    
    def forward(self, x):
        h = self.act(self.bn(self.linear(x)))
        return h + x

class Player(nn.Module):
    def __init__(self, n_in=128, n_hid=100, n_out=6):
        super(Player, self).__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, n_hid),
                                    nn.BatchNorm1d(n_hid),
                                    nn.ReLU(),
                                    ResLinear(n_hid, n_hid, nn.ReLU()),
                                    nn.Linear(n_hid, n_out))
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, obs, normalized=False):
        if normalized:
            return self.softmax(self.layers(obs))
        else:
            return self.layers(obs)

class Value(nn.Module):
    def __init__(self, n_in=128, n_hid=100):
        super(Value, self).__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, n_hid),
                                    nn.BatchNorm1d(n_hid),
                                    nn.ReLU(),
                                    ResLinear(n_hid, n_hid, nn.ReLU()),
                                    nn.Linear(n_hid, 1))
    
    def forward(self, obs):
        return self.layers(obs)


