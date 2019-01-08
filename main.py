# -*- coding: utf-8 -*
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from option import BaseOptions
import numpy as np
from getdata import get_data
import os
from tb_visualizer import TBVisualizer
from utils import load_pre_model

def main():
    opt = BaseOptions().parse()
    writer = TBVisualizer(opt)

    # -------Device configuration--------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ----Get input data(data loader)----
    dataset = get_data(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.workers)

    # ------------Set labels------------
    real_I_label_D = torch.ones((opt.batch_size,)).to(device)
    fake_I_label_D = torch.zeros((opt.batch_size,)).to(device)
    fake_I_label_G = torch.ones((opt.batch_size,)).to(device)

    # ---Fixed z input to eval/visualize---
    z_sample = torch.from_numpy(np.random.uniform(-1, 1, size=(opt.batch_size , opt.input_nz, 1, 1)))
    z_sample = z_sample.requires_grad_(requires_grad=True).float().to(device)

    # ------Define model_class object------
    net_G = Generator(opt).to(device)
    net_D = Discriminator(opt).to(device)
    if opt.load_model:
        load_pre_model(net_G, 'G', opt)
        load_pre_model(net_D, 'D', opt)

    # ---Define objective and optimizer---
    # criterion_D = nn.CrossEntropyLoss()
    # criterion_G = nn.CrossEntropyLoss()
    criterion_D = nn.BCELoss()
    criterion_G = nn.BCELoss()
    optim_D = optim.Adam(net_D.parameters(), lr=opt.learning_rate, betas=(opt.beta1, opt.beta2))
    optim_G = optim.Adam(net_G.parameters(), lr=opt.learning_rate, betas=(opt.beta1, opt.beta2))

    # -------------Train model------------
    pre_epoch = opt.load_epoch
    global_step = dataset.num_batches * pre_epoch
    for epoch in range(pre_epoch, opt.max_epoch):
        for batch, real_I in enumerate(dataloader):
            global_step += 1
            # Prepare input data
            real_I = real_I.requires_grad_(requires_grad=True).to(device)
            z_G = torch.from_numpy(np.random.uniform(-1, 1, size=(opt.batch_size , opt.input_nz, 1, 1)))
            z_G = z_G.requires_grad_(requires_grad=True).float().to(device)

            # ------Train Discriminator------
            # Forward
            fake_I = net_G(z_G)
            fake_I_logits = net_D(fake_I.detach())
            # ****** detach阻断梯度继续反向计算，计算多余梯度，反正优化器里也不会优化之前的参数******
            real_I_logits = net_D(real_I)
            loss_D = 0.5 * criterion_D(fake_I_logits, fake_I_label_D) + \
                     0.5 * criterion_D(real_I_logits, real_I_label_D)
            writer.scalar('loss_D', loss_D.item(), global_step)
            # Backward
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()   # net_G没有变

            # --------Train Generator--------
            # Forward
            # fake_I = net_G(z_G)  # 此时net_G没有变，z_G和上一个fake_I都是一样的
            fake_I_logits = net_D(fake_I)
            loss_G = criterion_G(fake_I_logits, fake_I_label_G)
            # Backward
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()   # net_D没有变

            # Train Generator twice to make sure that loss_D does not go to 0
            # Forward
            fake_I = net_G(z_G)
            fake_I_logits = net_D(fake_I)
            # ****** 此处fake_I没有加detach()，是因为更新G网络，梯度要从D到G一直反向计算到初始位置，******
            # ****** 然后在优化器里面只更新G的参数。计算D的梯度只是为了把loss反传到G ******
            loss_G = criterion_G(fake_I_logits, fake_I_label_G)
            writer.image('fake_I', fake_I, global_step)
            writer.scalar('loss_G', loss_G.item(), global_step)
            # Backward
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            # Print loss every batch
            writer.log(('Epoch[%d/%d], Batch[%d/%d], loss_G = %.4f, loss_D = %.4f' %
                        (epoch+1, opt.max_epoch, batch+1, len(dataloader), loss_G.item(), loss_D.item())))

            # Eval every 100 batches
            if np.mod(batch, 100) == 1:
                fake_I = net_G(z_sample)
                fake_I_logits = net_D(fake_I)
                loss_G = criterion_G(fake_I_logits, fake_I_label_G)
                writer.image('eval_face_I', fake_I, global_step)
                writer.save_imgs(fake_I, os.path.join(opt.sample_dir, 'train_epoch_%d_batch_%d.png' % (epoch, batch)))
                writer.log('Eval loss %.4f' % loss_G.item())

        # Save model every epoch
        filename = 'netG_epoch_%d.pth' % (epoch+1)
        torch.save(net_G.state_dict(), os.path.join(opt.checkpoint_dir, filename))
        filename = 'netD_epoch_%d.pth' % (epoch+1)
        torch.save(net_D.state_dict(), os.path.join(opt.checkpoint_dir, filename))

if __name__ == '__main__':
    main()