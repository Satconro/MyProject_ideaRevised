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
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import utils
from datasets.my_dataset import Base_Dataset
from models.stage2_model import Enhancement_Encoder, Enhancement_Decoder


class Enhancement_Model_Tester:
    def __init__(self, running_config):
        # Config
        self.config = running_config
        # Data:
        self.dataloader = DataLoader(
            Base_Dataset(running_config.root),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        # Model: load tested model and set mode to eval()
        self.encoder = Enhancement_Encoder().to(self.config.device).eval()
        self.decoder = Enhancement_Decoder().to(self.config.device).eval()
        self._load_model()

    def testing(self):
        pbar = tqdm(enumerate(self.dataloader),
                    total=len(self.dataloader),
                    unit='batch',
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    ncols=150)
        for batch_n, (batch_img_names, batch_imgs) in pbar:     # batch_img_names 是一个列表
            with torch.no_grad():
                generated_imgs = self.decoder(*self.encoder(batch_imgs))  # (<=batch_size, channel, w, h)
                if not os.path.exists(self.config.output_dir):
                    utils.mkdirs(self.config.output_dir)
                # Save mini-batch images
                for img_name, gen_img in zip(batch_img_names, generated_imgs):
                    save_image(gen_img, os.path.join(self.config.output_dir, img_name))

    def start_testing(self):
        self.testing()

    def _load_model(self):
        if self.config.load_checkpoint is True:
            enc_path = os.path.join(self.config.snapshots_folder, 'encoder.ckpt')
            dec_path = os.path.join(self.config.snapshots_folder, 'decoder.ckpt')
            utils.load_checkpoint(enc_path, self.encoder, self.config.device)
            utils.load_checkpoint(dec_path, self.decoder, self.config.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--root', type=str, default="/home/lsm/home/My_Dataset/Real_Underwater_Image_2")
    parser.add_argument('--output_dir', type=str, default="/home/lsm/home/My_Dataset/Enhancement_result_1/")
    parser.add_argument('--load_checkpoint', type=bool, default=True)
    parser.add_argument('--snapshots_folder', type=str, default=str(UPPER_DIR) + "/snapshots/stage_2")

    config = parser.parse_args()
    config.device = "cpu"
    # config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # config.device = "cuda"
    # torch.cuda.set_device(2)

    Enhancement_Model_Tester(config).start_testing()
