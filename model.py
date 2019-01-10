# -*- coding: utf-8 -*
from nets import Generator, Discriminator
import torch
from torch import nn, optim
import os
import numpy as np

# 整个模型的类，包括网络结构（类），网络运行，loss，反传等操作
class DCGAN:
    # the whole model related operations
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # define net instance
        self.net_G = Generator(self.opt).to(self.device)
        self.net_D = Discriminator(self.opt).to(self.device)
        if self.opt.load_model:
            self._load_pre_model(self.net_G, 'G')
            self._load_pre_model(self.net_D, 'D')

        # define objective and optimizer
        self.criterion_D = nn.BCELoss()
        self.criterion_G = nn.BCELoss()
        self.optim_D = optim.Adam(self.net_D.parameters(), lr=self.opt.learning_rate,
                                  betas=(self.opt.beta1, self.opt.beta2))
        self.optim_G = optim.Adam(self.net_G.parameters(), lr=self.opt.learning_rate,
                                  betas=(self.opt.beta1, self.opt.beta2))

        # set labels
        self.real_I_label_D = torch.ones((opt.batch_size,)).to(self.device)
        self.fake_I_label_D = torch.zeros((opt.batch_size,)).to(self.device)
        self.fake_I_label_G = torch.ones((opt.batch_size,)).to(self.device)

        # set eval samples
        self.z_sample = self._get_z()

    def _load_pre_model(self, net, net_module):
        load_filename = 'net%s_epoch_%s.pth' % (net_module, str(self.opt.load_epoch))
        load_path = os.path.join(self.opt.checkpoint_dir, load_filename)
        assert os.path.exists(load_path), 'Weights file not found. Have you trained a model!? ' \
                                          'We are not providing one' % load_path
        net.load_state_dict(torch.load(load_path))
        print('load net: %s' % load_path)

    def _get_z(self):
        z = torch.from_numpy(np.random.uniform(-1, 1, size=(self.opt.batch_size, self.opt.input_nz, 1, 1)))
        z = z.requires_grad_(requires_grad=True).float().to(self.device)
        return z

    def set_input(self, input):
        self.real_I = input.requires_grad_(requires_grad=True).to(self.device)
        self.z_G = self._get_z()

    def _forward_D(self):
        self.fake_I = self.net_G(self.z_G)
        self.fake_I_logits = self.net_D(self.fake_I.detach())
        # detach阻断梯度继续反向计算，计算多余梯度，反正优化器里也不会优化之前的参数
        self.real_I_logits = self.net_D(self.real_I)

    def _forward_G(self):
        self.fake_I = self.net_G(self.z_G)
        self.fake_I_logits = self.net_D(self.fake_I)
        # 此处fake_I没有加detach()，是因为更新G网络，梯度要从D到G一直反向计算到初始位置，
        # 然后在优化器里面只更新G的参数。计算D的梯度只是为了把loss反传到G

    def _backward_D(self):
        loss_D_fake = self.criterion_D(self.fake_I_logits, self.fake_I_label_D)
        loss_D_real = self.criterion_D(self.real_I_logits, self.real_I_label_D)
        self.loss_D = 0.5 * (loss_D_fake + loss_D_real)
        self.optim_D.zero_grad()
        self.loss_D.backward()
        self.optim_D.step()

    def _backward_G(self):
        self.loss_G = self.criterion_G(self.fake_I_logits, self.fake_I_label_G)
        self.optim_G.zero_grad()
        self.loss_G.backward()
        self.optim_G.step()

    def optimizer(self):
        # update D one time
        self._forward_D()
        self._backward_D()
        # update G two times
        self._forward_G()
        self._backward_G()
        self._forward_G()
        self._backward_G()

    def eval_sample(self):
        self.eval_fake_I = self.net_G(self.z_sample)
        eval_fake_I_logits = self.net_D(self.eval_fake_I)
        self.eval_loss_G = self.criterion_G(eval_fake_I_logits, self.fake_I_label_G)

    def save_model(self, epoch):
        filename = 'netG_epoch_%d.pth' % (epoch+1)
        torch.save(self.net_G.state_dict(), os.path.join(self.opt.checkpoint_dir, filename))
        filename = 'netD_epoch_%d.pth' % (epoch+1)
        torch.save(self.net_D.state_dict(), os.path.join(self.opt.checkpoint_dir, filename))