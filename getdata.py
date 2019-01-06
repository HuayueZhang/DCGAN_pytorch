import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
from PIL import Image
import os
import glob


def img_transforms(opt):
    if opt.isCrop and opt.isResize:
        return Compose([
            CenterCrop(opt.input_height),
            Resize(opt.output_height),
            ToTensor()
        ])
    else:
        return ToTensor()


def list_datapaths(opt):
    dataroot = opt.dataroot
    dataroot = os.path.join(dataroot, opt.dataset, opt.input_pattern)
    datapaths = glob.glob(dataroot)
    num_batches = int(len(datapaths)/opt.batch_size)
    datapaths = datapaths[0: num_batches*opt.batch_size]
    return datapaths


class myDataset(Dataset):
    def __init__(self, opt):
        super(myDataset, self).__init__()
        self.transforms = img_transforms(opt)
        self.datapaths = list_datapaths(opt)

    def __getitem__(self, idx):
        img_path = self.datapaths[idx]
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.datapaths)


