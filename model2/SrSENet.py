import torch
import torch.nn as nn
import torch.nn.init as init
from model2.SrSEBlock import SrSEBlock
import numpy as np


# class Net(nn.Module):
#     def __init__(self, blocks, rate, use_se=True):
#         super(Net, self).__init__()
#         self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
#
#         self.conv_res = self._make_layer(SrSEBlock(64, use_se=use_se), blocks)
#
#         self.Transpose = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
#
#         self.conv_t = nn.Conv2d(in_channels=64, out_channels=64 * rate ** 2, kernel_size=1, stride=1, padding=1,
#                                 bias=False)
#         self.subpixel = nn.PixelShuffle(rate)
#         self.relu = nn.PReLU()
#
#         self.convt_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.orthogonal(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def _make_layer(self, block, blocks):
#         layers = []
#         for _ in range(blocks):
#             layers.append(block)
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv_input(x)
#         residual = out
#         out = self.conv_res(out)
#         # out += residual
#         print(out.size())
#         # out = self.conv_t(out)
#         out = self.Transpose(out)
#         print(out.size())
#         out += residual
#         out = self.relu(self.subpixel(out))
#         out = self.convt_out(out)
#         return out


def tconv(planes=1, rate=2):
    if rate%2==0:
        return nn.ConvTranspose2d(planes, planes, kernel_size=int(4*rate//2), stride=rate, padding=rate//2, bias=False)
    else:
        return nn.ConvTranspose2d(planes, planes, kernel_size=int(4*rate//2), stride=rate, padding=rate//2, bias=False)


class Net(nn.Module):
    def __init__(self, blocks, rate):
        super(Net, self).__init__()
        # self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_I1 = tconv(planes=1, rate=rate)

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.convt_F1 = self.make_layer(SrSEBlock(64), blocks)
        self.convt_F1 = self._make_layer(SrSEBlock(64), blocks)

        # self.Transpose = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.Transpose = tconv(planes=64, rate=rate)
        self.relu_transpose = nn.LeakyReLU(0.2, inplace=True)

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
        """放大LR"""
        convt_I1 = self.convt_I1(x)

        """利用SE-ResNet进行特征提取"""
        out = self.conv_input(x)



        convt_F1 = self.convt_F1(out)

        """放大提取到的feature map"""
        convt_out = self.relu_transpose(self.Transpose(convt_F1))
        convt_R1 = self.convt_R1(convt_out)
        HR = convt_I1 + convt_R1
        return HR


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        # print(loss.data[0])
        return loss
