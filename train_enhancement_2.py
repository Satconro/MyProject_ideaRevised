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
import os
from itertools import cycle, chain

import torch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader

import utils
from datasets.my_dataset import EUVP_Dataset, DUO_Dataset
from losses.loss_modules import EncoderLoss, DecoderLoss, DiscriminatorLoss
from models.stage2_model import Enhancement_Encoder, Enhancement_Decoder, Enhancement_Discriminator


class Enhancement_Module_Trainer:
    def __init__(self, running_config):
        # Config:
        self.config = running_config
        # Dataset:
        # self.UIEB_dataset = UIEBD_Dataset(self.config.root_UIEB)
        self.data_real = DUO_Dataset(root=self.config.root_DUO, num_of_imgs=5885)  # 总大小6671
        self.data_syn = torch.utils.data.ConcatDataset([
            EUVP_Dataset(self.config.root_EUVP, "underwater_imagenet"),
            EUVP_Dataset(self.config.root_EUVP, "underwater_scenes")
        ])  # 总大小5885
        print("length of real: {}, length of syn: {}".format(len(self.data_real), len(self.data_syn)))
        # Dataloader:
        self.dataloader_real = DataLoader(
            self.data_real,
            batch_size=running_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.dataloader_synthetic = DataLoader(
            self.data_syn,
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
        # # threshold:
        # self.rec_threshold = 0.3
        # self.dis_threshold = 0.005
        self.rec_threshold = 0.3
        self.dis_threshold = 0.01
        utils.seed_everything()

        if self.config.load_checkpoint is True:
            self._load_model()

        if self.config.pre_training is True:
            self._pre_training()

    def _load_model(self, epoch=None):
        if epoch is not None:
            enc_path = os.path.join(self.config.load_snapshots_folder, "epoch_" + str(epoch), 'encoder.ckpt')
            dec_path = os.path.join(self.config.load_snapshots_folder, "epoch_" + str(epoch), 'decoder.ckpt')
            dis_path = os.path.join(self.config.load_snapshots_folder, "epoch_" + str(epoch), 'discriminator.ckpt')
        else:
            enc_path = os.path.join(self.config.load_snapshots_folder, 'encoder.ckpt')
            dec_path = os.path.join(self.config.load_snapshots_folder, 'decoder.ckpt')
            dis_path = os.path.join(self.config.load_snapshots_folder, 'discriminator.ckpt')

        utils.load_checkpoint(enc_path, self.encoder, self.config.device, self.optimizer_enc)
        utils.load_checkpoint(dec_path, self.decoder, self.config.device, self.optimizer_dec)
        utils.load_checkpoint(dis_path, self.discriminator, self.config.device, self.optimizer_dis)

    def _save_model(self, epoch=None):
        if not os.path.exists(self.config.save_snapshots_folder):
            os.makedirs(self.config.save_snapshots_folder)
        if epoch is not None:
            if not os.path.exists(os.path.join(self.config.save_snapshots_folder, "epoch_" + str(epoch))):
                os.makedirs(os.path.join(self.config.save_snapshots_folder, "epoch_" + str(epoch)))
            enc_path = os.path.join(self.config.save_snapshots_folder, "epoch_" + str(epoch), 'encoder.ckpt')
            dec_path = os.path.join(self.config.save_snapshots_folder, "epoch_" + str(epoch), 'decoder.ckpt')
            dis_path = os.path.join(self.config.save_snapshots_folder, "epoch_" + str(epoch), 'discriminator.ckpt')
        else:
            enc_path = os.path.join(self.config.save_snapshots_folder, 'encoder.ckpt')
            dec_path = os.path.join(self.config.save_snapshots_folder, 'decoder.ckpt')
            dis_path = os.path.join(self.config.save_snapshots_folder, 'discriminator.ckpt')

        utils.save_checkpoint(self.encoder, self.optimizer_enc, enc_path)
        utils.save_checkpoint(self.decoder, self.optimizer_dec, dec_path)
        utils.save_checkpoint(self.discriminator, self.optimizer_dis, dis_path)

    def _train_enhancement(self, synthetic, clear):
        # 训练编码器和解码器
        self.optimizer_enc.zero_grad()
        self.optimizer_dec.zero_grad()
        generated_img = self.decoder(*self.encoder(synthetic))
        rec_loss = self.loss_dec(generated_img, clear)
        rec_loss.backward()
        self.optimizer_enc.step()
        self.optimizer_dec.step()
        return rec_loss.item()

    def _train_discriminator(self, synthetic, real):
        self.optimizer_dis.zero_grad()
        fake_embedding, _ = self.encoder(synthetic)
        real_embedding, _ = self.encoder(real)
        fake_result = self.discriminator(fake_embedding)
        real_result = self.discriminator(real_embedding)
        dis_loss = self.loss_dis(fake_result, real_result, real_embedding, fake_embedding,
                                 self.discriminator)
        dis_loss.backward(retain_graph=True)  # 对于每个batch的数据，dis需要多次更新
        self.encoder.zero_grad()  # 清除掉encoder的梯度
        self.optimizer_dis.step()
        return dis_loss.item()

    def _train_encoder(self, synthetic):
        self.optimizer_enc.zero_grad()
        fake_embedding, _ = self.encoder(synthetic)
        fake_result = self.discriminator(fake_embedding)
        enc_loss = self.loss_enc(fake_result)
        enc_loss.backward()
        self.optimizer_enc.step()
        return enc_loss.item()

    def cal_dis_loss(self, synthetic, real):
        fake_embedding, _ = self.encoder(synthetic)
        real_embedding, _ = self.encoder(real)
        fake_result = self.discriminator(fake_embedding)
        real_result = self.discriminator(real_embedding)
        dis_loss = self.loss_dis(fake_result, real_result, real_embedding, fake_embedding,
                                     self.discriminator)
        return dis_loss

    def _pre_training(self):  # 在合成数据集上预先进行训练
        print("Start pre-training".format(self.config.load_checkpoint))
        epoch = self.config.pre_training_epoch
        for epoch_n in range(epoch):
            pbar = tqdm(enumerate(self.dataloader_synthetic),
                        total=len(self.dataloader_synthetic),
                        desc=f'Epoch {epoch_n + 1}/{epoch}',
                        unit='batch',
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                        ncols=150)
            sum_loss = 0
            for batch_n, (_, distorted, clear) in pbar:
                distorted = distorted.to(self.config.device)
                clear = clear.to(self.config.device)
                rec_loss = self._train_enhancement(distorted, clear)
                sum_loss += rec_loss
                pbar.set_postfix({'rec_loss': rec_loss, 'avg:': sum_loss / (batch_n + 1)})
        self._save_model()

    def training(self):  # 预训练完成后，交替训练两个网络
        for epoch_n in range(self.config.epoch):
            pbar = tqdm(enumerate(zip(self.dataloader_real, self.dataloader_synthetic)),
                        total=max(len(self.dataloader_real), len(self.dataloader_synthetic)),
                        desc=f'Epoch {epoch_n + 1}/{self.config.epoch}',
                        unit='batch',
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                        ncols=150
                        )

            sum_r_loss = 0
            for batch_n, data in pbar:
                _, real, _, synthetic, clear = chain.from_iterable(data)  # 解包
                if not len(synthetic) == len(real):  # 跳过batch_size不对齐的那一组数据
                    continue

                real = real.to(self.config.device)
                synthetic = synthetic.to(self.config.device)
                clear = clear.to(self.config.device)

                # 训练分类器
                dis_loss = self.cal_dis_loss(synthetic, real)
                dis_loss = 0
                for i in range(self.config.num_critic):
                    dis_loss += self._train_discriminator(synthetic, real)
                dis_loss = dis_loss / self.config.num_critic
                # 训练生成器
                enc_loss = self._train_encoder(synthetic)

                # 训练编码器和解码器
                rec_loss = self._train_enhancement(synthetic, clear)
                sum_r_loss += rec_loss

                # 在进度条后显示当前batch的损失
                pbar.set_postfix(
                    {
                        'enc_loss': enc_loss,
                        'dis_loss': dis_loss,
                        'rec_loss': rec_loss,
                        'average_r': sum_r_loss / (batch_n + 1)})
                # 更新显示条信息，1表示完成了一个batch的训练
                pbar.update(1)

            # 每5个epoch保存一次
            if (epoch_n + 1) % 5 == 0:
                self._save_model(epoch_n + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--root_UIEB', type=str, default="/home/lsm/home/datasets/UIEB")
    parser.add_argument('--root_EUVP', type=str, default="/home/lsm/home/datasets/EUVP")
    parser.add_argument('--root_RUIE', type=str, default="/home/lsm/home/datasets/RUIE")
    parser.add_argument('--root_DUO', type=str, default="/home/lsm/home/datasets/DUO/DUO/images/train")
    parser.add_argument('--pre_training', type=bool, default=False)
    parser.add_argument('--pre_training_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--num_critic', type=int, default=2)
    parser.add_argument('--load_checkpoint', type=bool, default=True)
    parser.add_argument('--load_snapshots_folder', type=str, default="/home/lsm/home/snapshots/v4.5/epoch_25")
    parser.add_argument('--save_snapshots_folder', type=str, default="/home/lsm/home/snapshots/v4.5")


    # parser.add_argument('--snapshots_folder', type=str, default=str(UPPER_DIR) + "/snapshots/exp_2/")

    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # config.device = "cuda"
    # torch.cuda.set_device(3)
    Enhancement_Module_Trainer(config).training()
