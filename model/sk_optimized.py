import torch
import torch.nn as nn
import torch.nn.init as init
from functools import reduce


class Net(nn.Module):
    def __init__(self, blocks, rate):
        super(Net, self).__init__()
        self.convt_I1 = nn.ConvTranspose2d(1, 1, kernel_size=int(4*rate//2), stride=rate, padding=rate//2, bias=False)
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F1 = self._make_layer(SKBlock(64), blocks)
        self.Transpose = nn.ConvTranspose2d(64, 64, kernel_size=int(4*rate//2), stride=rate, padding=rate//2, bias=False)
        self.relu_transpose = nn.PReLU()
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                init.orthogonal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        convt_I1 = self.convt_I1(x)
        out = self.relu(self.conv_input(x))
        convt_F1 = self.convt_F1(out)
        convt_out = self.relu_transpose(self.Transpose(convt_F1))
        convt_R1 = self.convt_R1(convt_out)
        HR = convt_I1 + convt_R1
        return HR


class L1_Charbonnier_loss(nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


class SKBlock(nn.Module):
    def __init__(self, planes, stride=1, use_sk=True):
        super(SKBlock, self).__init__()
        self.use_sk = use_sk
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.sk = SKLayer(planes)
        self.channelAttention = channelAttention(planes, planes)

    def forward(self, x):
        residual = x
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))

        if self.use_sk:
            out = self.sk(out)
            out = self.channelAttention(out)
        out += residual
        return out


class SKLayer(nn.Module):
    def __init__(self, channel, stride=1, M=2, r=16, L=32):
        super(SKLayer, self).__init__()
        self.M = M
        self.out_channels = channel
        d = max(channel//r, L)
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(nn.Conv2d(channel, channel, 3, stride, padding=1+i, dilation=1+i, groups=32, bias=False),
                                           nn.PReLU()))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(channel, d, 1, bias=False),
                                 nn.PReLU())
        self.fc2 = nn.Conv2d(d, channel*M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.channelAttention = channelAttention(channel, channel)

    def forward(self, xx):
        batch_size = xx.size(0)
        output = []
        #  split
        for i, conv in enumerate(self.conv):
            output.append(conv(xx))
        # fusion
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
        V = self.channelAttention(V)
        return V


class channelAttention(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(channelAttention, self).__init__()

        self.swish = nn.Sigmoid()
        self.channel_squeeze = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(inChannels * 4, inChannels // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(inChannels // 4, inChannels * 4, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=inChannels * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU(),
        )
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=inChannels * 4, out_channels=outChannels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU(),
        )

    def forward(self, x):
        ex = self.trans1(x)
        out1 = self.channel_squeeze(ex)
        out1 = self.conv_down(out1)
        out1 = out1*self.swish(out1)
        out1 = self.conv_up(out1)
        weight = self.sig(out1)
        out = ex*weight
        out = self.trans2(out)
        return out
