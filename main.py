import torch
from torch.utils.data import DataLoader
from model import DCGAN
from option import BaseOptions
import numpy as np
from getdata import get_data
import os
from tb_visualizer import TBVisualizer

def main():
    opt = BaseOptions().parse()
    writer = TBVisualizer(opt)
    model = DCGAN(opt)

    dataset = get_data(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.workers)

    pre_epoch = opt.load_epoch
    global_step = dataset.num_batches * pre_epoch
    for epoch in range(pre_epoch, opt.max_epoch):
        for batch, real_I in enumerate(dataloader):
            global_step += 1
            model.set_input(real_I)
            # 把input传入到model的类里面成为成员变量，让model类里面的其他方法可以直接使用
            model.optimizer()

            writer.scalar('loss_D', model.loss_D.item(), global_step)
            writer.image('fake_I', model.fake_I, global_step)
            writer.scalar('loss_G', model.loss_G.item(), global_step)
            writer.log(('Epoch[%d/%d], Batch[%d/%d], loss_G = %.4f, loss_D = %.4f' %
                        (epoch+1, opt.max_epoch, batch+1, len(dataloader),
                         model.loss_G.item(), model.loss_D.item())))
            # 在类的成员函数里面“初始化”的变量，在这个成员函数调用以后，该变量可以在类外调用

            # Eval every 100 batches
            if np.mod(batch, 100) == 1:
                model.eval_sample()
                writer.image('eval_face_I', model.eval_fake_I, global_step)
                writer.save_imgs(model.eval_fake_I, os.path.join(opt.sample_dir, 'train_epoch_%d_batch_%d.png' % (epoch, batch)))
                writer.log('Eval loss %.4f' % model.eval_loss_G.item())

        # Save model every epoch
        model.save_model(epoch)


if __name__ == '__main__':
    main()