"""
    DenseNet简单实现
"""

import torch
from torch import nn
from d2l import torch as d2l


# 每一层卷积操作
def conv_block(input_channels, num_channels):
    """
    :param input_channels: 输入通道数
    :param num_channels: 输出通道数
    :return:
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


# 将n个卷积层组合成一个稠密块
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        """
        :param num_convs: 卷积层的数目
        :param input_channels: 稠密块的输入通道数目
        :param num_channels: 输出通道数 —— 每一个卷积层的输出通道数是相同的
        """
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X


# 过渡层：例如，输入64，每一层卷积输出32
def transition_block(input_channels, num_channels):
    """
        使用1*1卷积调整通道数目
    :param input_channels: 输入通道数
    :param num_channels: 输出通道数
    :return:
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
