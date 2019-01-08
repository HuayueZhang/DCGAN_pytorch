# -*- coding: utf-8 -*
from __future__ import division
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.input_nz = opt.input_nz
        self.output_nc = opt.output_nc
        self.ngf = opt.ngf

        # Build generator layers
        layers = [
            nn.ConvTranspose2d(self.input_nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, self.output_nc, 4, 2, 1, bias=False),
            nn.Tanh()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        output = self.layers(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.input_nc = opt.input_nc
        self.ndf = opt.ndf

        layers = [
            nn.Conv2d(self.input_nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.ndf * 1, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        output = self.layers(input)
        return output.squeeze(1)