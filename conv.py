import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam, SGD
from torchvision.models.resnet import BasicBlock

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
            padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvNet(nn.Module):

    def __init__(self, num_classes=6, 
                 n_frames=1, n_hid=64,
                 value=False):
        super(ConvNet, self).__init__()
        self.value = value

        self.cnn = nn.Sequential(
                nn.Conv2d(3 * n_frames, n_hid, kernel_size=3, stride=1),
                nn.BatchNorm2d(n_hid),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(n_hid, n_hid, kernel_size=3, stride=1),
                nn.BatchNorm2d(n_hid),
                nn.ReLU(),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                #nn.Conv2d(n_hid, n_hid, kernel_size=3, stride=1),
                #nn.BatchNorm2d(n_hid),
                #nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                )

        self.fc = nn.Sequential(
                nn.Linear(n_hid, num_classes)
                )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)

        if not self.value:
            x = self.softmax(x)

        return x

def Player(n_frames=1, n_hid=64, **kwargs):
    model = ConvNet(n_frames=n_frames, n_hid=n_hid, num_classes=6)
    return model

def Value(n_frames=1, n_hid=64, **kwargs):
    model = ConvNet(n_frames=n_frames, n_hid=n_hid, num_classes=1, value=True)
    return model




