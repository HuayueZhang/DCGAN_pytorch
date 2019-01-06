# -*- coding: utf-8 -*
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import dataloader, dataset, DataLoader
from model import Generator, Discriminator
from option import BaseOptions
import numpy as np
from getdata import myDataset
import os

def main():
    opt = BaseOptions().parse()

    # ----Get input data(data loader)----
    dataset = myDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.workers)

    # ------------Set labels------------
    real_I_label_D = torch.from_numpy(np.ones((opt.batch_size,))).long()
    fake_I_label_D = torch.from_numpy(np.zeros((opt.batch_size,))).long()
    fake_I_label_G = torch.from_numpy(np.ones((opt.batch_size,))).long()

    # --------Define class object-------
    net_G = Generator(opt)
    net_D = Discriminator(opt)
    if torch.cuda.is_available():
        net_G.cuda()
        net_D.cuda()

    # ---Define objective and optimizer---
    criterion_D = nn.CrossEntropyLoss()
    criterion_G = nn.CrossEntropyLoss()
    optim_D = optim.Adam(net_D.parameters(), lr=opt.learning_rate, betas=(opt.beta1, opt.beta2))
    optim_G = optim.Adam(net_G.parameters(), lr=opt.learning_rate, betas=(opt.beta1, opt.beta2))

    # -------------Train model------------
    for epoch in range(opt.max_epoch):
        for batch, real_I in enumerate(dataloader):
            # Prepare input data
            z_G = torch.randn(opt.batch_size, opt.input_nz, 1, 1)
            if torch.cuda.is_available():
                real_I = Variable(real_I).cuda()
                z_G = Variable(z_G).cuda()
                real_I_label_D = Variable(real_I_label_D).cuda()
                fake_I_label_D = Variable(fake_I_label_D).cuda()
                fake_I_label_G = Variable(fake_I_label_G).cuda()
            else:
                real_I = Variable(real_I)
                z_G = Variable(z_G)
                real_I_label_D = Variable(real_I_label_D)
                fake_I_label_D = Variable(fake_I_label_D)
                fake_I_label_G = Variable(fake_I_label_G)

            # Train Discriminator
            # Forward
            fake_I = net_G(z_G)
            fake_I_logits = net_D(fake_I)
            real_I_logits = net_D(real_I)
            loss_D = criterion_D(fake_I_logits, fake_I_label_D) + \
                     criterion_D(real_I_logits, real_I_label_D)
            # Backward
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            # Train Generator
            # Forward
            fake_I = net_G(z_G)
            fake_I_logits = net_D(fake_I)
            loss_G = criterion_G(fake_I_logits, fake_I_label_G)
            # Backward
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            # Train Generator twice to make sure that loss_D does not go to 0
            # Forward
            fake_I = net_G(z_G)
            fake_I_logits = net_D(fake_I)
            loss_G = criterion_G(fake_I_logits, fake_I_label_G)
            # Backward
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            # Print loss every batch
            print('Epoch[%d/%d], Batch[%d/%d], loss_G = %.4f, loss_D = %.4f' %
                  (epoch+1, opt.max_epoch, batch+1, len(dataloader), loss_G.item(), loss_D.item()))

        # Save model every epoch
        filename = 'netG_epoch_%d.pth' % epoch
        torch.save(net_G.state_dict(), os.path.join(opt.checkpoint_dir, filename))
        filename = 'netD_epoch_%d.pth' % epoch
        torch.save(net_D.state_dict(), os.path.join(opt.checkpoint_dir, filename))

if __name__ == '__main__':
    main()