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
from itertools import cycle, chain
from collections.abc import Iterable

import torch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader

import utils
from datasets.my_dataset import UIEBD_DatasetBase, EUVP_Dataset, RUIE_Dataset
from losses.loss_modules import EncoderLoss, DecoderLoss, DiscriminatorLoss
from models.stage2_model import Enhancement_Encoder, Enhancement_Decoder, Enhancement_Discriminator


class Translation_Model_Trainer:
    def __init__(self, running_config):
        # Config:
        self.config = running_config
        # Dataset:
        self.UIEB_dataset = self._get_UIEB_dataset()
        self.EUVP_dataset = self._get_EUVP_dataset("all")
        self.RUIE_dataset = self._get_RUIE_dataset()
        # Dataloader:
        self.dataloader_synthetic = DataLoader(
            torch.utils.data.ConcatDataset([self.UIEB_dataset, self.EUVP_dataset]),
            batch_size=running_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.dataloader_real = DataLoader(
            self.RUIE_dataset,
            batch_size=running_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        # Model:
        self.encoder = Enhancement_Encoder()
        self.encoder.to(self.config.device).train(mode=True)
        self.decoder = Enhancement_Decoder()
        self.decoder.to(self.config.device).train(mode=True)
        self.discriminator = Enhancement_Discriminator()
        self.discriminator.to(self.config.device).train(mode=True)
        # Optimizer & Loss:
        self.optimizer_enc = Adam(self.encoder.parameters(), lr=running_config.learning_rate)
        self.optimizer_dec = Adam(self.decoder.parameters(), lr=running_config.learning_rate)
        self.optimizer_dis = Adam(self.discriminator.parameters(), lr=running_config.learning_rate)
        self.vgg = torchvision.models.vgg19(pretrained=True)
        self.vgg.to(self.config.device).eval()
        print("VGG-19 model is loaded")
        self.loss_enc = EncoderLoss().to(self.config.device)
        self.loss_dec = DecoderLoss(self.vgg, self.config.device).to(self.config.device)
        self.loss_dis = DiscriminatorLoss().to(self.config.device)
        # threshold:
        self.rec_threshold = 0.4
        self.dis_threshold = 0.005
        # Pre-action:
        # utils.seed_everything()
        self._load_model()
        # # 记录每一个epoch后的平均损失
        # self.enc_losses_of_epoch = []
        # self.dec_losses_of_epoch = []
        # self.dis_losses_of_epoch = []
        # # 记录每一个batch后的平均损失
        # self.enc_losses_of_batch = {}
        # self.dec_losses_of_batch = {}
        # self.dis_losses_of_batch = {}

    def start_training(self):
        for epoch_n in range(self.config.epoch):
            if len(self.dataloader_real) < len(self.dataloader_synthetic):
                dataloader = zip(cycle(self.dataloader_real), self.dataloader_synthetic)
            else:
                dataloader = zip(self.dataloader_real, cycle(self.dataloader_synthetic))
            pbar = tqdm(enumerate(dataloader),
                        total=max(len(self.dataloader_real), len(self.dataloader_synthetic)),
                        desc=f'Epoch {epoch_n + 1}/{self.config.epoch}',
                        unit='batch',
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                        ncols=150)
            switch = None
            enc_loss = dis_loss = rec_loss = torch.zeros((1, 1, 1, 1))
            for batch_n, data in pbar:
                # 解包
                _, real, _, synthetic, clear = chain.from_iterable(data)
                real = real.to(self.config.device)
                synthetic = synthetic.to(self.config.device)
                clear = clear.to(self.config.device)

                if not switch:
                    with torch.no_grad():
                        generated_img = self.decoder(*self.encoder(synthetic))
                        rec_loss = self.loss_dec(generated_img, clear)
                    if rec_loss.item() > self.rec_threshold:
                        # 训练编码器和解码器
                        self.optimizer_enc.zero_grad()
                        self.optimizer_dec.zero_grad()
                        generated_img = self.decoder(*self.encoder(synthetic))
                        rec_loss = self.loss_dec(generated_img, clear)
                        rec_loss.backward()
                        self.optimizer_enc.step()
                        self.optimizer_dec.step()
                    else:
                        switch = None # TODO: 先训练10个epoch的增强网络，看看能大概收敛到什么位置
                else:
                    # with torch.no_grad(): 这里需要计算梯度
                    fake_embedding, _ = self.encoder(synthetic)
                    real_embedding, _ = self.encoder(real)
                    dis_result_fake = self.discriminator(fake_embedding)
                    dis_result_real = self.discriminator(real_embedding)
                    dis_loss = self.loss_dis(dis_result_fake, dis_result_real, real_embedding, fake_embedding,
                                             self.discriminator)

                    if dis_loss > self.dis_threshold:
                        # 训练分类器
                        self.optimizer_dis.zero_grad()
                        fake_embedding, _ = self.encoder(synthetic)
                        real_embedding, _ = self.encoder(real)
                        dis_result_fake = self.discriminator(fake_embedding)
                        dis_result_real = self.discriminator(real_embedding)
                        dis_loss = self.loss_dis(dis_result_fake, dis_result_real, real_embedding, fake_embedding,
                                                 self.discriminator)
                        dis_loss.backward()
                        self.optimizer_dis.step()

                        # 训练生成器
                        self.optimizer_enc.zero_grad()
                        fake_embedding, _ = self.encoder(synthetic)
                        dis_result_fake = self.discriminator(fake_embedding)
                        enc_loss = self.loss_enc(dis_result_fake)
                        enc_loss.backward()
                        self.optimizer_enc.step()
                    else:
                        switch = None

                # 在进度条后显示当前batch的损失
                pbar.set_postfix({'g_loss': enc_loss.item(), 'd_loss': dis_loss.item(), 'rec_loss': rec_loss.item()})
                # 更新显示条信息，1表示完成了一个batch的训练
                pbar.update(1)

            # 每5个epoch保存一次
            if (epoch_n + 1) % 5 == 0:
                if not os.path.exists(self.config.snapshots_folder):
                    os.makedirs(self.config.snapshots_folder)
                enc_path = os.path.join(self.config.snapshots_folder, str(epoch_n), 'encoder.ckpt')
                dec_path = os.path.join(self.config.snapshots_folder, str(epoch_n), 'decoder.ckpt')
                dis_path = os.path.join(self.config.snapshots_folder, str(epoch_n), 'discriminator.ckpt')

                utils.save_checkpoint(self.encoder, self.optimizer_enc, enc_path)
                utils.save_checkpoint(self.decoder, self.optimizer_dec, dec_path)
                utils.save_checkpoint(self.discriminator, self.optimizer_dis, dis_path)

    def _load_model(self):
        if self.config.load_checkpoint is True:
            enc_path = os.path.join(self.config.snapshots_folder, 'encoder.ckpt')
            dec_path = os.path.join(self.config.snapshots_folder, 'decoder.ckpt')
            dis_path = os.path.join(self.config.snapshots_folder, 'discriminator.ckpt')

            utils.load_checkpoint(enc_path, self.encoder, self.config.device, self.optimizer_enc)
            utils.load_checkpoint(dec_path, self.decoder,  self.config.device, self.optimizer_dec)
            utils.load_checkpoint(dis_path, self.discriminator, self.config.device, self.optimizer_dis)

    def _get_UIEB_dataset(self):
        # UIEB数据集下的所有数据
        return UIEBD_DatasetBase(self.config.root_UIEB)  # 890

    def _get_EUVP_dataset(self, name):
        #
        if name == 'dark':
            return EUVP_Dataset(self.config.root_EUVP, "underwater_dark")  # 5530
        elif name == 'imagenet':
            return EUVP_Dataset(self.config.root_EUVP, "underwater_imagenet")  # 3700
        elif name == 'scenes':
            return EUVP_Dataset(self.config.root_EUVP, "underwater_scenes")  # 2185
        elif name == "all":
            return torch.utils.data.ConcatDataset([
                EUVP_Dataset(self.config.root_EUVP, "underwater_dark"),
                EUVP_Dataset(self.config.root_EUVP, "underwater_imagenet"),
                EUVP_Dataset(self.config.root_EUVP, "underwater_scenes")
            ])

    def _get_RUIE_dataset(self):
        return RUIE_Dataset(root=self.config.root_RUIE)  # 总大小3930

    # def _record_batch(self):
    #     # 记录训练数据
    #     if epoch_n == 0:
    #         G_losses_of_batch[epoch_n] = [enc_loss.item()]
    #         D_losses_of_batch[epoch_n] = [dis_loss.item()]
    #         R_losses_of_batch[epoch_n] = [rec_loss.item()]
    #     else:
    #         G_losses_of_batch[epoch_n].append(enc_loss.item())
    #         D_losses_of_batch[epoch_n].append(dis_loss.item())
    #         R_losses_of_batch[epoch_n].append(rec_loss.item())
    #     pass

    def _record_epoch(self):
        # D_losses_of_epoch.append(numpy.mean(D_losses_of_batch[epoch_n]))
        # G_losses_of_epoch.append(numpy.mean(G_losses_of_batch[epoch_n]))
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--root_UIEB', type=str, default="/home/lsm/home/datasets/UIEB")
    parser.add_argument('--root_EUVP', type=str, default="/home/lsm/home/datasets/EUVP")
    parser.add_argument('--root_RUIE', type=str, default="/home/lsm/home/datasets/RUIE")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--rec_epoch', type=int, default=40)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--load_checkpoint', type=bool, default=True)
    parser.add_argument('--snapshots_folder', type=str, default=str(UPPER_DIR) + "/snapshots/exp_2/")
    parser.add_argument('--output_images_path', type=str, default=str(UPPER_DIR) + "/data/output/")

    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # config.device = "cuda"
    # torch.cuda.set_device(3)

    Translation_Model_Trainer(config).start_training()
