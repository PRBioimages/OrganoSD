import cv2
import tqdm
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import wandb

from model.detecseg_y import UNet
from model.detecseg_u import UNet512
from data.dataset import DetecSegSet, ToPercentCoords
from data.augment import ODSegAugment
from loss.multibox_loss import SegLoss
from config import organ

import argparse
import warnings
warnings.filterwarnings("ignore")



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Unet_complex Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--test_setroot', default='',
                    help='Dataset root directory path')
parser.add_argument('--save_img', default='',
                    help='Directory for saving checkpoint models')
parser.add_argument('--resume', default='',
                    type=str, help='Checkpoint state_dict file to resume training from')

args = parser.parse_args()

cfg512 = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

num_classes = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet512(1, cfg512, num_classes)
net = net.to(device)

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    net.load_weights(args.resume)
    print('successful load weight！')
else:
    print('Initializing weights...')

file_list = os.listdir(args.test_setroot)
for file in file_list:
    fileName = os.path.splitext(file)[0]
    imgpath = os.path.join(args.test_setroot, file)
    img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img = torch.tensor(img.copy())
    img = img.unsqueeze(0).unsqueeze(1).to(device)
    _, out_seg = net(Variable(img.float(), requires_grad=True))
    out = out_seg[0].squeeze(0) * 255
    out = out.detach().numpy()
    cv2.imencode('.jpg', out)[1].tofile(args.save_img + file)
    print('finish ' + str(file))
