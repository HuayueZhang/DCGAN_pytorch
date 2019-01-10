import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize
import torchvision.datasets as tvset
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
    return datapaths, num_batches


class myDataset(Dataset):
    def __init__(self, opt):
        super(myDataset, self).__init__()    # 子类重写了init构造方法，还想要继承父类的构造方法时，显式地调用父类构造方法
        # 或者Dataset.__init__()
        self.transforms = img_transforms(opt)
        self.datapaths, self.num_batches = list_datapaths(opt)

    def __getitem__(self, idx):
        img_path = self.datapaths[idx]
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.datapaths)


def get_data(opt):
    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = tvset.ImageFolder(root=opt.dataroot,
                                    transform=Compose([
                                        CenterCrop(opt.input_height),
                                        Resize(opt.output_height),
                                        ToTensor(),
                                        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))
    elif opt.dataset == 'lsun':
        dataset = tvset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                             transform=Compose([
                                 CenterCrop(opt.input_height),
                                 Resize(opt.output_height),
                                 ToTensor(),
                                 Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))
    elif opt.dataset == 'mnist':
        dataset = tvset.MNIST(root=opt.dataroot, download=True,
                              transform=Compose([
                                  CenterCrop(opt.input_height),
                                  Resize(opt.output_height),
                                  ToTensor(),
                                  Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))
    elif opt.dataset == 'fake':
        dataset = tvset.FakeData(image_size=(3, opt.ouput_height, opt.output_height),
                                 transform=ToTensor())
    else:
        dataset = myDataset(opt)

    return dataset

