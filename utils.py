# -*- coding: utf-8 -*
import numpy as np
import torch
import os

def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    # assert判断语句，抛出错误。assert不可乱用，为什么用在这呢，因为一般此处不会出现错误
    # 但一旦此处出现错误，manifold_h * manifold_w != num_images，并不会在此处报错
    # 而会影响到后面某一步骤出现错误，到时候一步步debug回来才知道是这的问题
    # 添加一句assert，就是为了避免一步步的回溯debug
    return [manifold_h, manifold_w]


def merge(images, size):
    num_h, num_w = size[0], size[1]
    c, h, w = images.shape[1], images.shape[2], images.shape[3]
    if c == 3 or c == 4:
        img = np.zeros((c, h * num_h, w * num_w))
        for idx, image in enumerate(images):
            x = idx % num_w
            y = idx // num_w
            img[:, y*h: y*h+h, x*w: x*w+w] = image.detach()
        return img
    elif c == 1:
        img = np.zeros((h * num_h, w * num_w))
        for idx, image in enumerate(images):
            x = idx % num_w
            y = idx // num_w
            img[y*h: y*h+h, x*w: x*w+w] = image[0, :, :]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def get_manifold_image_array(images):
    images = (images + 1.) / 2.
    size = image_manifold_size(images.shape[0])
    manifold_image = merge(images, size)      # (C, H, W)
    manifold_image = np.transpose(manifold_image, (1, 2, 0))   # (H, W, C)
    return manifold_image   # (H, W, C)


def load_pre_model(net, net_module, opt):
    load_filename = 'net%s_epoch_%s.pth' % (net_module, str(opt.load_epoch))
    load_path = os.path.join(opt.checkpoint_dir, load_filename)
    assert os.path.exists(load_path), 'Weights file not found. Have you trained a model!? ' \
                                      'We are not providing one' % load_path
    net.load_state_dict(torch.load(load_path))
    print('load net: %s' % load_path)