import torch
import torch.nn as nn


def l1_loss(correct_images, generated_images):
    torch_l1_dist = torch.nn.PairwiseDistance(p=1)
    loss = torch.mean(torch_l1_dist(correct_images, generated_images))
    return loss


def l2_loss(correct_images, generated_images):
    torch_l2_dist = torch.nn.PairwiseDistance(p=2)
    loss = torch.mean((torch_l2_dist(correct_images, generated_images)))
    return loss


# Image Gradient Difference Loss
def igdl_loss(correct_images, generated_images, igdl_p=1):
    correct_images_gradient_x = __calculate_x_gradient(correct_images)
    generated_images_gradient_x = __calculate_x_gradient(generated_images)
    correct_images_gradient_y = __calculate_y_gradient(correct_images)
    generated_images_gradient_y = __calculate_y_gradient(generated_images)
    pairwise_p_distance = torch.nn.PairwiseDistance(p=igdl_p)
    distances_x_gradient = pairwise_p_distance(
        correct_images_gradient_x, generated_images_gradient_x
    )
    distances_y_gradient = pairwise_p_distance(
        correct_images_gradient_y, generated_images_gradient_y
    )
    loss_x_gradient = torch.mean(distances_x_gradient)
    loss_y_gradient = torch.mean(distances_y_gradient)
    loss = 0.5 * (loss_x_gradient + loss_y_gradient)
    return loss


def __calculate_x_gradient(images):
    x_gradient_filter = torch.Tensor(
        [
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
        ]
    ).cuda()
    x_gradient_filter = x_gradient_filter.view(3, 1, 3, 3)
    result = torch.functional.F.conv2d(
        images, x_gradient_filter, groups=3, padding=(1, 1)
    )
    return result


def __calculate_y_gradient(images):
    y_gradient_filter = torch.Tensor(
        [
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
        ]
    ).cuda()
    y_gradient_filter = y_gradient_filter.view(3, 1, 3, 3)
    result = torch.functional.F.conv2d(
        images, y_gradient_filter, groups=3, padding=(1, 1)
    )
    return result


# Wasserstein Distance
def wgan_loss(discriminator_fake, discriminator_correct):
    loss = torch.mean(discriminator_fake) - torch.mean(discriminator_correct)
    return loss


# Gradient Penalty
def gradient_penalty(
        correct_images, generated_images, discriminator_net, norm_epsilon=1e-12
):
    batch_size = correct_images.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1).cuda()
    x_interpolated = correct_images * epsilon + (1 - epsilon) * generated_images
    interpolated_labels = discriminator_net.forward(x_interpolated).cuda()
    # Following solution source is taken from https://github.com/arturml/pytorch-wgan-gp/blob/master/wgangp.py
    grad_outputs = torch.ones(interpolated_labels.size()).cuda()
    gradients = torch.autograd.grad(
        outputs=interpolated_labels,
        inputs=x_interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + norm_epsilon)
    return torch.mean(((gradients_norm - 1) ** 2))


# Wasserstein GAN with gradient penalty
def wgan_gp_loss():
    # wgan_gp_loss = wgan_loss + λ * gradient_penalty
    pass


class Base_Loss_Module(nn.Module):
    """
        For example:

        Build:
            class MSE_Loss(Base_Loss_Module):
                def __init__(self, p_norm=2)
                    super(MSE_Loss, self).__init__()
                    self.p_norm = 2
                    self.criterion = torch.nn.PairwiseDistance(p=2)

                def forward(self, source, target):
                    return self.criterion(source, target)

        Call:
            l2 = MSE_Loss(2)
            l2_loss = l2(source, target)
    """

    def __init__(self):
        super(Base_Loss_Module, self).__init__()

    def forward(self, *args, **kwargs):
        pass
