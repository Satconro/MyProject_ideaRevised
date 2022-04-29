import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

from losses.vgg_loss import VGGLoss
from losses.base_loss import *


def conv(input_channels, output_channels, is_down, layer_norm=False):
    """
        上采样/下采样过程中的卷积块，上采样过程中不使用批量正则化
    """
    if is_down is True:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(output_channels) if not layer_norm else nn.GroupNorm(1, output_channels),
            nn.ReLU(),
        )
    elif is_down is False:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )


class ResDenseBlock(nn.Module):
    def __init__(self, num_layers, input_channels, output_channels, growth_ratio, is_down=True, lay_norm=False):
        """
            残差密集块，一个残差密集块包含多个密集连接的卷积层和位于最后的过渡层，过渡层调整最终输出的通道数到指定的大小并与原始的输入进行残差连接

            :param num_layers: 密集块的卷积层数
            :param input_channels: 输入通道数
            :param output_channels: 输出通道数，输出通道数必须等于输入通道数
            :param growth_ratio: 增长率
            :param is_down: 采用卷积还是转置卷积
        """
        super(ResDenseBlock, self).__init__()
        self.num_layers = num_layers
        self.is_down = is_down
        assert input_channels == output_channels, "The input_channels and output_channels of ResDenseBlock are supposed to match"

        layer = []
        for i in range(num_layers):
            layer.append(
                conv(input_channels + growth_ratio * i, growth_ratio, is_down, lay_norm),
            )
        self.dense = nn.Sequential(*layer)

        self.transition = nn.Conv2d(input_channels + growth_ratio * num_layers, output_channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        ori_input = x
        for i, blk in enumerate(self.dense):
            y = blk(x)
            # 连接通道维度上每个块的输入和输出
            x = torch.cat((x, y), dim=1)
        x = self.transition(x)
        return x + ori_input


def down_sampling(in_channels, out_channels):
    """
        下采样：每次下采样后Feature Map的尺寸减半，通道数乘2
    """
    assert out_channels == in_channels*2, \
        "The number of out_channels is supposed to be twice the number of in_channels"
    module = nn.Sequential(
        # nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )
    return module


def up_sampling(in_channels, out_channels):
    """
        上采样：转置卷积上采样，每次上采样后Feature Map的尺寸乘2
    """
    # assert out_channels == in_channels // 4, \
    #     "The number of out_channels is supposed to be a quarter of the number of in_channels"
    module = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )
    return module


# Encoder
class Enhancement_Encoder(nn.Module):

    def __init__(self, in_channels=3):
        super(Enhancement_Encoder, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )

        self.block1 = ResDenseBlock(3, 64, 64, 64)  # 64, 256, 256
        self.down1 = down_sampling(64, 128)  # 128, 128

        self.block2 = ResDenseBlock(6, 128, 128, 64)  # 128, 128, 128
        self.down2 = down_sampling(128, 256)  # 64, 64

        self.block3 = ResDenseBlock(8, 256, 256, 64)  # 256, 64, 64
        self.down3 = down_sampling(256, 512)  # 32, 32

        self.block4 = ResDenseBlock(8, 512, 512, 64)  # 512, 32, 32
        # self.down4 = down_sampling(512, 1024)  # 16, 16

        # self.battle_neck = nn.Sequential(
        #     nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #     nn.BatchNorm2d(1024), nn.LeakyReLU(0.2),
        #     nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #     nn.BatchNorm2d(1024), nn.LeakyReLU(0.2),
        # )  # 1024, 16, 16

    def forward(self, x):
        """
            x is the input image
        """
        x1 = self.block1(self.in_conv(x))
        x2 = self.block2(self.down1(x1))
        x3 = self.block3(self.down2(x2))
        x4 = self.block4(self.down3(x3))
        embedding = x4
        # embedding = self.battle_neck(self.down4(x4))
        return embedding, (x1, x2, x3, x4)


# Decoder
class Enhancement_Decoder(nn.Module):

    def __init__(self):
        super(Enhancement_Decoder, self).__init__()

        # self.up1 = up_sampling(512, 512)  # 512, 32, 32
        self.block1 = ResDenseBlock(8, 512, 512, 64, False)  # 512, 32, 32

        self.up2 = up_sampling(512, 256)  # 256, 64, 64
        self.block2 = ResDenseBlock(8, 512, 512, 64, False)  # 512, 64, 64

        self.up3 = up_sampling(512, 128)  # 128, 128, 128
        self.block3 = ResDenseBlock(6, 256, 256, 64, False)  # 256, 128, 128

        self.up4 = up_sampling(256, 64)  # 64, 256, 256
        self.block4 = ResDenseBlock(3, 128, 128, 64, False)  # 128, 256, 256

        self.out_conv = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(3),
            nn.ReLU(),
        )

    def forward(self, x, enc_outs):
        # x = self.up1(x)
        x = self.block1(x)
        x = self.up2(x)
        x = self.block2(self._padding_concat(x, enc_outs[2]))
        x = self.up3(x)
        x = self.block3(self._padding_concat(x, enc_outs[1]))
        x = self.up4(x)
        x = self.block4(self._padding_concat(x, enc_outs[0]))
        x = self.out_conv(x)
        return x

    def _padding_concat(self, x, y):
        x_size = x.shape[2:]     # w*h
        y_size = y.shape[2:]

        # 如果两个的形状不同，使用0进行填充
        w_dif = x_size[0] - y_size[0]
        h_dif = x_size[1] - y_size[1]
        if w_dif + h_dif != 0:
            if w_dif > 0:
                y = F.pad(y, (0, 0, 0, w_dif), mode='constant', value=0)

            elif w_dif < 0:
                x = F.pad(x, (0, 0, 0, -w_dif), mode='constant', value=0)

            if h_dif > 0:
                y = F.pad(y, (0, h_dif, 0, 0), mode='constant', value=0)   # [left, right, top, bot]
            elif h_dif < 0:
                x = F.pad(x, (0, -h_dif, 0, 0), mode='constant', value=0)

        return torch.cat((x, y), dim=1)


# Discriminator
class Enhancement_Discriminator(nn.Module):
    def __init__(self, in_channels=512):
        super(Enhancement_Discriminator, self).__init__()
        self.network = nn.Sequential(
            # 输入图像大小为：1024*16*16
            ResDenseBlock(3, in_channels, in_channels, 64, lay_norm=True),
            self.conv(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            ResDenseBlock(3, in_channels // 2, in_channels // 2, 64, lay_norm=True),
            self.conv(in_channels // 2, in_channels // 4, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            ResDenseBlock(3, in_channels // 4, in_channels // 4, 64, lay_norm=True),
            self.conv(in_channels // 4, 1, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
        )

    # 重构PatchGAN的块函数
    def conv(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            # 因为WGAN-GP会对每个输入的鉴别器梯度范数进行单独惩罚，而批量标准化将使其无效。所以图像转换部分的图片鉴别器不使用批量正则化
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


def test():
    encoder = Enhancement_Encoder()
    decoder = Enhancement_Decoder()
    discriminator = Enhancement_Discriminator()

    x = torch.randn(1, 3, 1014, 514)
    embedding, enc_outs = encoder(x)
    generated = decoder(embedding, enc_outs)
    result = discriminator(embedding)


if __name__ == "__main__":
    test()
