"""

"""
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_img(root, img):
    return img, Image.open(os.path.join(root, img))


class Base_Dataset(Dataset):
    def __init__(self, root, transform=None):
        """
            最简单的数据集，从一个文件中读取，转换图像为Tensor数据集格式并满足Dataset规范
        """
        super(Base_Dataset, self).__init__()
        self.root = root
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # 将PIL.Image转化为tensor，即归一化。 注：shape 会从(H，W，C)变成(C，H，W)；先ToTensor转换为Tensor才能进行正则化
            ])

        self.imgs = os.listdir(self.root)
        self.len = len(self.imgs)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name, img_path = self.imgs[index], os.path.join(self.root, self.imgs[index])
        img = Image.open(img_path)
        return img_name, self.transform(img)


class Base_Paired_Dataset(Dataset):
    def __init__(self, root_distorted, root_clear, transform=None):
        super(Base_Paired_Dataset, self).__init__()
        self.root_distorted = root_distorted
        self.root_clear = root_clear
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # 将PIL.Image转化为tensor，即归一化。 注：shape 会从(H，W，C)变成(C，H，W)；先ToTensor转换为Tensor才能进行正则化
            ])
        else:
            self.transform = transform

        # 读取文件路径下的所有图像名称，此处未转换为绝对路径
        self.imgs_distorted = os.listdir(self.root_distorted)
        self.imgs_clear = os.listdir(self.root_clear)
        # 分别获取各个数据集的大小
        len1 = len(self.imgs_distorted)
        len2 = len(self.imgs_clear)
        # 检测原数据和标签值的长度是否相等
        assert len1 == len2, "The length of the raw images and the reference does not match."
        # 设置大数据集的长度为总数据集的长度
        self.len = max(len1, len2)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # 根据索引号读取图像数据，
        name_1, img_1 = read_img(self.root_distorted, self.imgs_distorted[index % self.len])
        name_2, img_2 = read_img(self.root_clear, self.imgs_clear[index % self.len])
        return name_1, self.transform(img_1), self.transform(img_2)

    def set_length(self, len):
        self.len = len