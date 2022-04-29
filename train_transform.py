# 路径及sys.path处理
import sys
from pathlib import Path

CURRENT_FILE_PATH = Path(__file__).resolve()
UPPER_DIR = CURRENT_FILE_PATH.parents[0]  # 内容根，即当前文件的上级目录
print("CURRENT_FILE_PATH：" + __file__)
if str(UPPER_DIR) not in sys.path:
    sys.path.extend([str(UPPER_DIR)])  # 执行时，添加内容根和项目根到pythonPath

import argparse
from tqdm import tqdm
import numpy
import os

import torch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader

import utils
from datasets.my_dataset import Trans_Train_Dataset
from models.stage1_model import Translation_Module, Translation_Discriminator


class Translation_Model_Trainer:
    def __init__(self, running_config):
        # Config
        self.config = running_config
        # Data
        self.dataloader = DataLoader(
            Trans_Train_Dataset(root_real=running_config.root_real, root_synthetic=running_config.root_synthetic),
            batch_size=running_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        # Model
        self.translation_module = Translation_Module()
        self.translation_module.to(self.config.device).train(mode=True)
        self.translation_discriminator = Translation_Discriminator()
        self.translation_discriminator.to(self.config.device).train(mode=True)
        self.vgg = torchvision.models.vgg19(pretrained=True)
        print("VGG-19 model is loaded")
        # Optimizer & Loss
        self.optimizer_g = Adam(self.translation_module.parameters(), lr=running_config.learning_rate)
        self.optimizer_d = Adam(self.translation_discriminator.parameters(), lr=running_config.learning_rate)
        self.loss_g = self.translation_module.Loss(self.vgg, running_config.device).to(self.config.device)
        self.loss_d = self.translation_discriminator.Loss().to(self.config.device)

        # utils.seed_everything()
        self._load_model()

    def training(self):
        # 记录每一个epoch后的平均损失
        G_losses_of_epoch = []
        D_losses_of_epoch = []
        # 记录每一个batch后的平均损失
        G_losses_of_batch = {}
        D_losses_of_batch = {}

        for epoch_n in range(self.config.epoch):
            pbar = tqdm(enumerate(self.dataloader),
                        total=len(self.dataloader),
                        desc=f'Epoch {epoch_n + 1}/{self.config.epoch}',
                        unit='batch',
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                        ncols=150)
            for batch_n, (real_imgs, synthetic_img_d, _) in pbar:
                real_imgs = real_imgs.to(self.config.device)
                synthetic_img_d = synthetic_img_d.to(self.config.device)

                # 训练分类器
                self.optimizer_d.zero_grad()
                fake_images = self.translation_module(synthetic_img_d)
                d_result_real = self.translation_discriminator(real_imgs)
                d_result_fake = self.translation_discriminator(fake_images)
                d_loss = self.loss_d(d_result_fake, d_result_real, real_imgs, fake_images,
                                     self.translation_discriminator)
                d_loss.backward()
                self.optimizer_d.step()

                # 训练生成器
                self.optimizer_g.zero_grad()
                fake_images = self.translation_module(synthetic_img_d).to(self.config.device)
                d_result_fake = self.translation_discriminator(fake_images)
                g_loss = self.loss_g.forward(d_result_fake, synthetic_img_d, fake_images)
                # Perform backprop
                g_loss.backward()
                self.optimizer_g.step()

                # 在进度条后显示当前batch的损失
                pbar.set_postfix({'generator_loss': g_loss.item(), 'discriminator_loss': d_loss.item()})
                # 更新显示条信息，1表示完成了一个batch的训练
                pbar.update(1)

                # 记录训练数据
                if epoch_n not in D_losses_of_batch:
                    D_losses_of_batch[epoch_n] = [d_loss.item()]
                else:
                    D_losses_of_batch[epoch_n].append(d_loss.item())
                if epoch_n not in G_losses_of_batch:
                    G_losses_of_batch[epoch_n] = [g_loss.item()]
                else:
                    G_losses_of_batch[epoch_n].append(g_loss.item())

            D_losses_of_epoch.append(numpy.mean(D_losses_of_batch[epoch_n]))
            G_losses_of_epoch.append(numpy.mean(G_losses_of_batch[epoch_n]))

            # 每5个epoch保存一次
            if (epoch_n + 1) % 5 == 0 and self.config.save_model:
                if not os.path.exists(self.config.snapshots_folder):
                    os.makedirs(self.config.snapshots_folder)
                utils.save_checkpoint(self.translation_module, self.optimizer_g,
                                      os.path.join(self.config.snapshots_folder, 'generator.ckpt'))
                utils.save_checkpoint(self.translation_discriminator, self.optimizer_d,
                                      os.path.join(self.config.snapshots_folder, 'discriminator.ckpt'))

    def start_training(self):
        self.training()

    def _load_model(self):
        if self.config.load_checkpoint is True:
            g_path = os.path.join(self.config.snapshots_folder, 'generator.ckpt')
            d_path = os.path.join(self.config.snapshots_folder, 'discriminator.ckpt')
            utils.load_checkpoint(g_path, self.translation_module, self.optimizer_g, self.config.device)
            utils.load_checkpoint(d_path, self.translation_discriminator, self.optimizer_d, self.config.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--root_real', type=str, default="/home/lsm/home/My_Dataset/Real_Underwater_Image_2")
    parser.add_argument('--root_synthetic', type=str, default="/home/lsm/home/My_Dataset/Synthetic_Underwater_Image")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--snapshots_folder', type=str, default=str(UPPER_DIR) + "/snapshots/s2")
    parser.add_argument('--output_images_path', type=str, default=str(UPPER_DIR) + "/data/output/")

    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # config.device = "cuda"
    # torch.cuda.set_device(3)

    Translation_Model_Trainer(config).start_training()
