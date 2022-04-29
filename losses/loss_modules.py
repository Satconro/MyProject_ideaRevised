import torch
from losses import base_loss
from losses.vgg_loss import VGGLoss


class EncoderLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_result_fake):
        # 对抗性损失
        adversarial_loss = -torch.mean(dis_result_fake)
        return adversarial_loss


class DecoderLoss(torch.nn.Module):
    # 用于更新整个增强网络的损失函数: MSE + VGG +SSIM
    def __init__(
        self,
        vgg_model,
        device,
        lambda_MSE=1.0,
        lambda_sema=1.0,
    ):
        super().__init__()
        self.vgg_loss_criterion = VGGLoss(vgg_model, device)
        self.lambda_MSE = lambda_MSE
        self.lambda_sema = lambda_sema

    def forward(self, generated_images, clear_images):
        # MSE
        mse = torch.nn.MSELoss()
        mse_loss = mse(generated_images, clear_images)
        MSE_loss = self.lambda_MSE * mse(clear_images, generated_images)
        # Perpetual loss
        semantic_loss = self.lambda_sema * self.vgg_loss_criterion(generated_images, clear_images)
        total_loss = MSE_loss + semantic_loss
        return total_loss


class DiscriminatorLoss(torch.nn.Module):
    def __init__(
        self,
        lambda_gp=10.0
    ):
        super().__init__()
        self.lambda_gp = lambda_gp

    def forward(self, discriminator_fake, discriminator_correct, correct, generated, discriminator_net):
        # 对抗性损失
        adversarial_loss_weighted = base_loss.wgan_loss(discriminator_fake, discriminator_correct) \
                                    + self.lambda_gp * base_loss.gradient_penalty(correct, generated,
                                                                                  discriminator_net)
        return adversarial_loss_weighted
