from tensorboardX import SummaryWriter
import os
import time
import utils
import numpy as np
from PIL import Image
import torchvision

class TBVisualizer:
    def __init__(self, opt):
        self._opt = opt
        self._save_path = os.path.join(opt.checkpoint_dir, opt.log_folder)

        self._log_path = os.path.join(self._save_path, 'loss_log2.txt')  #?????
        self._tb_path = os.path.join(self._save_path, 'summary.josn')    #?????
        self._writer = SummaryWriter(self._save_path)

        with open(self._log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('=============== Training Loss (%s) =============\n' % now)

    def __del__(self):
        self._writer.close()

    def scalar(self, tag, scalar, global_step):
        self._writer.add_scalar(tag, scalar, global_step)
        self._writer.export_scalars_to_json(self._tb_path)

    def image(self, tag, imgs, global_step):
        manifold_img_array = utils.get_manifold_image_array(imgs) # (H, W, C)
        self._writer.add_image(tag, manifold_img_array, global_step, dataformats='HWC')

    def log(self, message):
        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_imgs(self, images, path):
        # return torchvision.utils.save_image(images, path, normalize=True)  # 功能和我自己实现的是一样的
        manifold_image_array = utils.get_manifold_image_array(images) * 255.  # (H, W, C)
        manifold_image_array = manifold_image_array.astype(np.uint8) # np.uint8(manifold_image_array)
        manifold_image_PIL = Image.fromarray(manifold_image_array)
        return manifold_image_PIL.save(path)