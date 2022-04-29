import torch
import os
import random
import numpy as np
from collections import Iterable


def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint {}".format(filename))
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, device, optimizer=None, lr=None):
    print("=> Loading checkpoint from {}".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if lr is not None:
        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdirs(dir_path):
    if isinstance(dir_path, list):
        for dir in dir_path:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
