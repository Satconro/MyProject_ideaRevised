# coding:utf-8

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

"""
    Description:
        CycleGAN的生成器
    模型结构：
        
        
"""

# 卷积层：集成卷积运算、实例正则化和激活函数
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()  # nn.Identity() 占位符，即当use_act为False时，不使用激活函数
        )

    def forward(self, x):
        return self.conv(x)


# 残差块：两层卷积 + 输入残差
class Residual_block(nn.Module):
    def __init__(self, channels):  # 注：CycleGAN_generator的残差网络部分通道数不发生改变
        super(Residual_block, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_residuals=9):
        super(Generator, self).__init__()
        self.num_residuals = num_residuals

        # The first layer: 第一层不改变图像大小
        self.first = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1, padding=3,
                      padding_mode="reflect"),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        """
        InstanceNorm2d and LayerNorm are very similar, but have some subtle differences. 
        InstanceNorm2d is applied on each channel of channeled data like RGB images, but LayerNorm is usually applied on entire sample and often in NLP tasks. 
        Additionally, LayerNorm applies elementwise affine transform, while InstanceNorm2d usually don’t apply affine transform.
        """
        # Down_blocks
        self.down_blocks = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        )
        # Residual_blocks
        self.residual_blocks = nn.Sequential(
            *[Residual_block(channels=256) for _ in range(num_residuals)]
        )
        # Up_blocks: 转置卷积部分与卷积部分相对应，实现图像的恢复
        self.up_blocks = nn.Sequential(
            ConvBlock(in_channels=256, out_channels=128, down=False, kernel_size=3, stride=2, padding=1,
                      output_padding=1),
            # output_padding=1确保放大后的尺寸是128而不是127
            ConvBlock(in_channels=128, out_channels=64, down=False, kernel_size=3, stride=2, padding=1,
                      output_padding=1),
        )
        # The last layer: 最后一层不改变图像大小
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3,
                      padding_mode="reflect"),
            nn.Tanh()
        )
        # Compose
        self.network = nn.Sequential(
            self.first,
            self.down_blocks,
            self.residual_blocks,
            self.up_blocks,
            self.last,
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    img_channels = 3
    img_size = 256

    x = torch.randn((2, img_channels, img_size, img_size))
    imgs = Generator(9).detach()

    plt.figure()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for i in range(imgs.shape[0]):
        plt.subplot(1, 2, i + 1)
        plt.axis("off")
        plt.imshow(imgs[i].permute(1, 2, 0))  # 一般神经网络中对图像处理之后的格式是（3，512，512）这种，分别为通道，高，宽。但是plt显示的图像格式为（512，512，3
        # ）也就是高，宽，通道。所以会出现错误。
    plt.show()
