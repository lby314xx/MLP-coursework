from torch import nn
from functools import reduce


class SKLayer(nn.Module):
    def __init__(self, channel, stride=1, M=2, r=16, L=32):
        super(SKLayer, self).__init__()
        self.M = M
        self.out_channels = channel
        d = max(channel//r, L)
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(nn.Conv2d(channel, channel, 3, stride, padding=1+i, dilation=1+i, groups=32, bias=False),
                                           nn.BatchNorm2d(channel),
                                           # nn.ReLU(inplace=True)
                                           nn.PReLU()))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(channel, d, 1, bias=False),
                               nn.BatchNorm2d(d),
                               # nn.ReLU(inplace=True)
                               nn.PReLU())
        self.fc2 = nn.Conv2d(d, channel*M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, xx):
        batch_size = xx.size(0)
        output = []
        #  the part of split
        for i, conv in enumerate(self.conv):
            # print(i,conv(input).size())
            output.append(conv(xx))
        # the part of fusion
        U = reduce(lambda x, y: x+y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        #  the part of selection
        a_b = list(a_b.chunk(self.M, dim=1))   # split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x*y, output, a_b))
        V = reduce(lambda x, y: x+y, V)
        return V

