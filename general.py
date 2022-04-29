import os
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class General_Dataset(Dataset):
    """
        General dataset, input the root dir of the img, return (the name of the image, transformed img) once at one time
    """
    def __init__(self, root_dir, transform=None):
        super(General_Dataset, self).__init__()
        self.root_dir = root_dir
        self.list_imgs = os.listdir(self.root_dir)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.ToTensor(),  # 将PIL.Image转化为tensor，即归一化。 注：shape 会从(H，W，C)变成(C，H，W)；先ToTensor转换为Tensor才能进行正则化
            ])
        self.len = len(self.list_imgs)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        #
        img_name, img_path = self.list_imgs[index], os.path.join(self.root_dir, self.list_imgs[index])
        img = Image.open(img_path)
        return img_name, self.transform(img)

class Base_Model_Trainer:
    def __init__(self, config):
        # Data
        self.dataset = None
        self.dataloader = None
        # Model
        self.model = None
        # forward: Only model needed for this phase

        # Calculate loss: loss function & optimizer

        # backward: loss value and optimizer

    def training_procedure(self):
        """
            Training demo:
            # Epoch setUp()
            for epoch_count in epochs:
                # Batch setUp()
                for batch_count in batchs： # Dataloader
                    # Training procedure here
                    ## Basically: forward -> calculate loss -> backward & update

                    # tearDown_AtEachBatch()
                # tearDown_AtEachEpoch()
            # TotalTeardown
        """
        pass

    def start_trainning(self):
        # Call training_procedure
        pass