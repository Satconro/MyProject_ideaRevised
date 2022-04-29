import torch
import torch.nn as nn

from models.reference.PatchGANs import Discriminator as PatchGAN
from losses.vgg_loss import VGGLoss
from losses.base_loss import *

"""
    阶段一：
        Domain_1：合成的水下数据集和其真值图像 (Xs, Ys) -> (Xst, Ys)
        Domain_2：真实的水下图像 (Xr)
        
        real = Xr, fake = Xst
            Xr和Xst趋于一致
        
        # 训练判别器
        ## 生成图像、计算距离、反向传播
        real = Xr
        fake = Translation(Xs) = Xst detach
        discriminator_real = d(real)
        discriminator_fake = d(fake)
        d_loss = XXX: (wgan-gp)
        更新d
        
        # 训练生成器
        ## 生成图像
        fake = fake
        ## 计算距离
        ### 欺骗discriminator的对抗损失 + 转换前后的语义一致性损失
        ## 反向传播，更新参数
"""


def conv_block(in_channels, out_channels, use_IN=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.InstanceNorm2d(out_channels) if use_IN else nn.Identity(),
        nn.ReLU(),
    )


class Res_Dense_Block(nn.Module):
    """
    一个Res-Dense Block由4层卷积和一个混合层构成
        除最后一层外，前三层卷积都使用实例正则化
        最后原始的输入与最后一层的输出进行残差连接
        模块的输出通道与输入通道数相同
    """
    def __init__(self, input_channels, output_channels, num_layers=4):    # 应有初始的input_channels = output_channels
        super(Res_Dense_Block, self).__init__()
        self.num_layers = num_layers

        layer = []
        for i in range(num_layers):
            layer.append(conv_block(
                input_channels + i * output_channels, output_channels, use_IN=True if not i == num_layers-1 else False
            ))
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        input = x.clone()
        for i, blk in enumerate(self.net):
            y = blk(x)
            # 连接通道维度上每个块的输入和输出，最后一层不连接
            x = torch.cat((x, y), dim=1) if i != self.num_layers - 1 else y
        return x + input


# 第一阶段用于合成数据增强的转换器
class Translation_Module(nn.Module):

    def __init__(self, input_channels=3, num_blocks=3):
        super(Translation_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU()
        )
        self.blocks = self.dense_blocks(64, 64, num_blocks)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU()
        )

    def dense_blocks(self, input_channel, output_channel, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(Res_Dense_Block(input_channel, output_channel))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.blocks[0](x1)
        x3 = self.blocks[1](x2)
        x4 = self.blocks[2](x3 + x1)
        x5 = self.out_conv(x2 + x4)
        return x5

    class Loss(Base_Loss_Module):
        def __init__(self, vgg_model, device, lambda_adv=1, lambda_sema=1):
            super().__init__()
            self.vgg_loss_criterion = VGGLoss(vgg_model, device)
            self.lambda_adv = lambda_adv
            self.lambda_sema = lambda_sema

        def forward(self, discriminator_fake, before_images, generated_images):
            # 对抗性损失
            adversarial = self.lambda_adv * (-torch.mean(discriminator_fake))
            # 语义一致性损失
            semantic_loss = self.lambda_sema * self.vgg_loss_criterion(before_images, generated_images)
            total_loss = adversarial + semantic_loss
            return total_loss


# 第一阶段用于合成数据鉴别的鉴别器
class Translation_Discriminator(PatchGAN):
    """
        i.e. PatchGAN
    """

    def __init__(self):
        super(Translation_Discriminator, self).__init__(in_channels=3)

    # 重构PatchGAN的块函数
    def block(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            # 因为WGAN-GP会对每个输入的鉴别器梯度范数进行单独惩罚，而批量标准化将使其无效。所以图像转换部分的图片鉴别器不使用批量正则化
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.LeakyReLU(0.2),
        )

    class Loss(Base_Loss_Module):
        """

        """
        def __init__(self, lambda_gp=10):
            super().__init__()
            self.lambda_gp = lambda_gp

        def forward(self, discriminator_fake, discriminator_correct, correct_images, generated_images,
                    discriminator_net):
            # 对抗性损失
            adversarial_loss_weighted = wgan_loss(discriminator_fake, discriminator_correct) \
                                        + self.lambda_gp * gradient_penalty(correct_images, generated_images,
                                                                            discriminator_net)
            return adversarial_loss_weighted


def test():
    model = Translation_Module()
    x = torch.randn(2, 3, 1024, 1024)
    print(model(x).shape)


if __name__ == "__main__":
    test()