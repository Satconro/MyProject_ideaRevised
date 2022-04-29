# coding: utf-8

import torch

"""
    Description:
        基于Pix2pix修改过的U-Net和Discriminator           
    备注：
        Discriminator的实现可能有点问题，为什么输出不缩放到[0,1]？
"""


def encoder_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.LeakyReLU(0.2),
    )
    return block


def decoder_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
    )
    return block


class U_Net(torch.nn.Module):
    """
        i.e. U-Net
    """
    def __init__(self):
        super(U_Net, self).__init__()
        self.encoder_module_list = torch.nn.ModuleList(
            [
                encoder_block(3, 64), # 64 128 128
                encoder_block(64, 128), # 128 64 64
                encoder_block(128, 256), # 256 32 32
                encoder_block(256, 512), # 512 16 16
                encoder_block(512, 512), # 512 8 8
                encoder_block(512, 512), # 512 4 4
                encoder_block(512, 512), # 512 2 2
            ]
        )
        # Every decoder block result layer then concatenated with encoder layer
        # This is the reason why 2nd to last blocks have these sizes, doubled outputs
        self.decoder_module_list = torch.nn.ModuleList(
            [
                decoder_block(512, 512), # 4
                decoder_block(1024, 512), # 8
                decoder_block(1024, 512), # 16
                decoder_block(1024, 256), # 32
                decoder_block(512, 128), # 64
                decoder_block(256, 64), # 128
                torch.nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1), # 256
            ]
        )

    def forward(self, x):
        """
            X is an object, which is changeable while computing.
        :param x:
        :return:
        """
        encoded_layers = dict()
        for i, encoder_module in enumerate(self.encoder_module_list):
            x = encoder_module(x)
            encoded_layers[str(i)] = x
        for i, decoder_module in enumerate(self.decoder_module_list):
            x = decoder_module(x)
            encoded_layer_index = len(self.decoder_module_list) - i - 2
            if encoded_layer_index >= 0:
                x = torch.cat([x, encoded_layers[str(encoded_layer_index)]], dim=1)
        x = torch.tanh(x)
        return x


class DiscriminatorNet(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 4, stride=2, padding=2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, 4, stride=2, padding=2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, 4, stride=2, padding=2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 512, 4, stride=1, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 1, 1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.net(x)
        return x
