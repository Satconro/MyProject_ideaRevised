# coding: utf-8

import torch
import torch.nn as nn

"""
    Description:
        PatchGANs鉴别器
    模型结构：
        
"""


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=None):
        super(Discriminator, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.network = nn.Sequential(
            # 输入图像大小为：3*256*256
            self.block(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect", bias=True),
            # 64*128*128
            self.block(features[0], features[1], kernel_size=4, stride=2, padding=1, padding_mode="reflect", bias=True),
            # 128*64*64
            self.block(features[1], features[2], kernel_size=4, stride=2, padding=1, padding_mode="reflect", bias=True),
            # 256*32*32
            self.block(features[2], features[3], kernel_size=4, stride=1, padding=1, padding_mode="reflect", bias=True),
            # 512*31*31

            # The last layer: 最后一层不进行实例正则化，使用Sigmoid函数将输入转换为[0-1]的概率分布
            # 最终输出为 1*30*30的概率张量矩阵，每一个坐标代表原图中一片70*70的区域
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=1, padding_mode="reflect"),
            nn.Sigmoid(),
        )

    def block(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
                # nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
                nn.Conv2d(in_channels, out_channels, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            )

    def forward(self, x):
        return self.network(x)


def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)
    print(preds.data)


if __name__ == "__main__":
    test()
