from PIL import Image
import torch
from torchvision.utils import make_grid
import os
import torchvision
import numpy as np

# torchvision.utils.save_image(img, imgPath)

def save_multi_channel_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    # 用于把几个图像按照网格排列的方式绘制出来
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def save_gray_image(img, save_dir, type, name, Gray=False):
    fname, fext = name.split('.')
    imgPath = os.path.join(save_dir, "%s_%s.%s" % (fname, type, fext))
    # torchvision.utils.save_image(img, imgPath)
    # 改写：torchvision.utils.save_image
    grid = torchvision.utils.make_grid(img, nrow=8, padding=2, pad_value=0,
                                       normalize=False, range=None, scale_each=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    # im.show()
    if Gray:
        im.convert('L').save(imgPath)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
    else:
        im.save(imgPath)


# 适用各种图片保存
def save_tensor_image(img, save_dir, type, name, Gray=False):
    fname, fext = name.split('.')
    imgPath = os.path.join(save_dir, "%s_%s.%s" % (fname, type, fext))
    img_array = tensor2array(img.data[0])
    image_pil = Image.fromarray(img_array)
    if Gray:
        image_pil.convert('L').save(imgPath)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
    else:
        image_pil.save(imgPath)

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2array(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2array(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)
