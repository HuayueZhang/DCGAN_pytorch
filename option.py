import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--isTrain', required=True, help='True for training and False for testing')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu idx: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--workers', type=int, default=1, help='Number of multi-spread')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='path to save models/checkpoint')
        parser.add_argument('--sample_dir', type=str, default='./samples', help='path to save eval samples during training')

        parser.add_argument('--dataroot', required=True, help='Path to images')
        parser.add_argument('--dataset', type=str, default='celebA', help='The name of dataset [celebAm mnist, lsun, zhy, we]')
        parser.add_argument('--input_pattern', type=str, default='*.jpg', help='Glob pattern of filename of input images')
        parser.add_argument('--max_epoch', type=int, default=25, help='Epoch to train')

        parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning tare of Adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='Momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='Momentum term of adam')
        parser.add_argument('--batch_size', type=int, default=64, help='The size of batch images')

        parser.add_argument('--isCrop', type=bool, default=True, help='Whether crop the input image to size(input_height, input_height)')
        parser.add_argument('--input_height', type=int, default=108, help='The size of image to use (will be center cropped)')
        parser.add_argument('--input_width', default=None, help='The size of image to use (will be center cropped)')

        parser.add_argument('--isResize', type=bool, default=True, help='Whether resize the input image to size(output_height, output_height)')
        parser.add_argument('--output_height', type=int, default=64, help='The size of image to use (will be center cropped)')
        parser.add_argument('--output_width', default=None, help='The size of image to use (will be center cropped)')

        parser.add_argument('--input_nz', type=int, default=100, help='The length of the input noise of Generator')
        parser.add_argument('--output_nc', type=int, default=3, help='The number of channels of the output of Generator')
        parser.add_argument('--ngf', type=int, default=64, help='Number of generator filters')

        parser.add_argument('--input_nc', type=int, default=3, help='The number of channels of the input of discriminator')
        parser.add_argument('--ndf', type=int, default=64, help='Number of discriminator filters')


        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()

        # create optional dir
        if not os.path.exists(opt.checkpoint_dir):
            os.mkdir(opt.checkpoint_dir)
        if not os.path.exists(opt.sample_dir):
            os.mkdir(opt.sample_dir)

        # get gpu idx
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

