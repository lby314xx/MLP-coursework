import torch.nn as nn
from model.SKLayer import SKLayer


class SKBlock(nn.Module):
    def __init__(self, planes, stride=1, use_sk=True):
        super(SKBlock, self).__init__()
        self.use_sk = use_sk
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu2 = nn.LeakyReLU()
        self.sk = SKLayer(planes)

    def forward(self, x):
        residual = x
        # out = self.relu1(self.conv1(x))
        # out = self.relu2(self.conv2(out))
        out = x

        if self.use_sk:
            out = self.sk(out)
        out += residual
        return out
