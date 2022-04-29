# 路径及sys.path处理
import random
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
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms

import utils
from datasets.my_dataset import Base_Dataset
from models.stage2_model import Enhancement_Encoder, Enhancement_Decoder

import torch.nn.functional as F


def find_img(root, relative_root=''):
    # 递归地找到一个文件夹下的所有图片路径，返回该文件夹的绝对路径，和其下的所有图片在该文件夹下的相对路径
    if os.path.isdir(root):
        sub_list = []
        for i in os.listdir(root):
            sub_list.extend(find_img(os.path.join(root, i), i))
        return [os.path.join(relative_root, i) for i in sub_list]
    else:
        return [os.path.join(root.split(os.sep)[-1])]


class dataset(Dataset):
    def __init__(self, root):
        super(dataset, self).__init__()
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.imgs = find_img(self.root)
        # self.imgs = random.sample(self.imgs, 200)
        self.len = len(self.imgs)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name, img_path = self.imgs[index], os.path.join(self.root, self.imgs[index])
        img = self.transform(Image.open(img_path))

        img_size = img.shape
        if img_size[1]*img_size[2] > 1920*1080:
            trans = transforms.Resize([img_size[1] // 2, img_size[2] // 2])
            # trans = transforms.RandomCrop([img_size[1] // 2, img_size[2] // 2])
            img = trans(img)
        return img_name, img



class Enhancement_Model_Tester:
    def __init__(self, running_config):
        # Config
        self.root = running_config.root
        self.target_dir = running_config.target_dir
        self.snapshots_folder = running_config.snapshots_folder
        self.device = running_config.device
        # Model: load tested model and set mode to eval()
        self.encoder = Enhancement_Encoder().to(self.device).eval()
        self.decoder = Enhancement_Decoder().to(self.device).eval()
        self._load_model()
        # Data:
        self.dataset = dataset(root=self.root)
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
        )
        torch.set_grad_enabled(False)

    def testing(self):
        pbar = tqdm(enumerate(self.dataloader),
                    total=len(self.dataloader),
                    unit='batch',
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    ncols=150)
        for batch_n, (img_path, img) in pbar:
            # img = self.transform(Image.open(os.path.join(self.root, img_path)))
            # img = img.to(self.device).unsqueeze(0)
            with torch.no_grad():
                img = img.to(self.device)
                generated_imgs = self.decoder(*self.encoder(img))
                # # 测试不使用中间变量的生成结果
                # embedding, enc_outs = self.encoder(img)
                # generated_imgs = self.decoder(torch.zeros_like(embedding), enc_outs)
                if not os.path.exists(self.target_dir):
                    utils.mkdirs(self.target_dir)
                if not os.path.exists(os.path.join(self.target_dir, *img_path[0].split(os.sep)[0:-1])):
                    utils.mkdirs(os.path.join(self.target_dir, *img_path[0].split(os.sep)[0:-1]))
                # Save img
                print(img_path[0])
                save_image(generated_imgs, os.path.join(self.target_dir, img_path[0]))
                del img
                del generated_imgs
                torch.cuda.empty_cache()

    def start_testing(self):
        self.testing()

    def _load_model(self):
        enc_path = os.path.join(self.snapshots_folder, 'encoder.ckpt')
        dec_path = os.path.join(self.snapshots_folder, 'decoder.ckpt')
        utils.load_checkpoint(enc_path, self.encoder, self.device)
        utils.load_checkpoint(dec_path, self.decoder, self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--root', type=str, default=r"/home/lsm/home/datasets/DUO_YOLO/images")
    parser.add_argument('--target_dir', type=str, default=r"/home/lsm/home/datasets/DUO_YOLO/images2")
    parser.add_argument('--snapshots_folder', type=str, default="/home/lsm/home/snapshots/v4.5/keep_training_after50/epoch_15")
    config = parser.parse_args()
    # config.device = "cpu"
    # config.device = "cuda"
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.set_device(2)
    Enhancement_Model_Tester(config).start_testing()
