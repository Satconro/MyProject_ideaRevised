import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.base import Base_Dataset, Base_Paired_Dataset, read_img


def get_imgs(root, relative_root=''):
    # 递归地找到一个文件夹下的所有图片路径，返回该文件夹的绝对路径，和其下的所有图片在该文件夹下的相对路径
    if os.path.isdir(root):
        sub_list = []
        for i in os.listdir(root):
            sub_list.extend(get_imgs(os.path.join(root, i), i))
        return [os.path.join(relative_root, i) for i in sub_list]
    else:
        return [os.path.join(root.split(os.sep)[-1])]


class EUVP_Dataset(Base_Paired_Dataset):
    def __init__(self, root, sub_set, num_of_imgs=None):
        """
            根据指定的名称读取EUVP数据集下的成对数据
        :param root:
        :param sub_set:
        """
        assert sub_set in os.listdir(os.path.join(root, "Paired"))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256]),
        ])
        super(EUVP_Dataset, self).__init__(os.path.join(root, "Paired", sub_set, "trainA"), os.path.join(root, "Paired", sub_set, "trainB"), transform)
        # 随机筛选指定个数的样本
        temp = list(zip(self.imgs_distorted, self.imgs_clear))
        if num_of_imgs is not None:
            temp = random.sample(temp, num_of_imgs)
        self.imgs_distorted, self.imgs_clear = zip(*temp)
        self.imgs_distorted = list(self.imgs_distorted)
        self.imgs_clear = list(self.imgs_clear)
        self.set_length(len(temp))


class DUO_Dataset(Base_Dataset):
    def __init__(self, root, num_of_imgs=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256])
        ])
        super().__init__(root, transform=transform)
        if num_of_imgs is not None:
            self.imgs = random.sample(self.imgs, num_of_imgs)
            self.len = len(self.imgs)


class UIEBD_DatasetBase(Base_Paired_Dataset):
    def __init__(self, root):
        """
            读取UIEB数据集下的所有数据，针对不同分辨率的图像采取不同的裁剪操作，如果输入图像尺寸大于256*256，采用随机裁剪，否则resize为256*256
        :param root: 文件根目录
        """
        self.root = root
        super(UIEBD_DatasetBase, self).__init__(
            root_distorted=os.path.join(root, "raw-890"),
            root_clear=os.path.join(root, "reference-890"),
        )
        self.transform_1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256]),  # 图像增强执行随机翻转的意义不大
        ])
        self.transform_2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop([256, 256])
            # transforms.RandomCrop(256, 256)
        ])

    # 重写__getitem__方法，修改格式转换操作
    def __getitem__(self, index):
        # 根据索引号读取图像数据，
        name_1, img_1 = read_img(self.root_distorted, self.imgs_distorted[index % self.len])
        name_2, img_2 = read_img(self.root_clear, self.imgs_clear[index % self.len])
        assert name_1 == name_2

        imgs = []
        for img in [img_1, img_2]:
            img_size = img.size
            if not (img_size[0] > 256 and img_size[1] > 256):
                img = self.transform_1(img)
                imgs.append(img)
            else:
                img = self.transform_2(img)
                imgs.append(img)
        return name_1, imgs[0], imgs[1]


def test():
    pass


if __name__ == '__main__':
    test()