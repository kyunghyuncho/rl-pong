import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam, SGD


class ResLinear(nn.Module):
    def __init__(self, n_in, n_out, act=nn.ELU()):
        super(ResLinear, self).__init__()
        self.act = act
        self.linear = nn.Linear(n_in, n_out)
        self.bn = nn.LayerNorm(n_out)

        if n_in != n_out:
            self.linear_linear = nn.Linear(n_in, n_out)

        self.n_in = n_in
        self.n_out = n_out
    
    def forward(self, x):
        h = self.act(self.bn(self.linear(x)))
        if self.n_in != self.n_out:
            return h + self.linear_linear(x)
        return h + x

class Player(nn.Module):
    def __init__(self, n_in=128, n_hid=100, n_out=6):
        super(Player, self).__init__()
        self.layers = nn.Sequential(ResLinear(n_in, n_hid, nn.ELU()),
                                    ResLinear(n_hid, n_hid, nn.ELU()),
                                    ResLinear(n_hid, n_hid, nn.ELU()),
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
        self.layers = nn.Sequential(ResLinear(n_in, n_hid, nn.ELU()),
                                    ResLinear(n_hid, n_hid, nn.ELU()),
                                    ResLinear(n_hid, n_hid, nn.ELU()),
                                    nn.Linear(n_hid, 1))
    
    def forward(self, obs):
        return self.layers(obs)


