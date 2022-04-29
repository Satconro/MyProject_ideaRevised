import torch
import torch.nn as nn
import torchvision


class VGGLoss(nn.Module):
    """
        VGG-19损失计算函数，为了节约显存，预训练的VGG-19模型需要在外部预先进行加载
        参考代码：
            import torchvision
            vgg = torchvision.models.vgg19(pretrained=True)
            print("VGG-19 model is loaded")
        注释：
            预训练好的VGG-19大小为548MB
            输出的损失值大小在 0-1 之间
            device指定用于计算VGGLoss的模型存放的位置
    """
    def __init__(self, vgg_model, device, n_layers=5):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1/32, 1/16, 1/8, 1/4, 1.0)  # 参考 Domain Adaptation for Underwater Image Enhancement

        vgg = vgg_model.features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)
        return loss


if __name__ == "__main__":
    source_img = torch.randn(2, 3, 256, 256)
    target_img = torch.randn(2, 3, 256, 256)
    vgg = torchvision.models.vgg19(pretrained=True)
    vgg_loss_criterion = VGGLoss(vgg, "cpu")
    print("VGG-19 model is loaded")

    vgg_loss = vgg_loss_criterion(source_img, target_img)
    print(vgg_loss)
