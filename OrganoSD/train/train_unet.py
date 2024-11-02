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

parser.add_argument('--train_set_root', default='',
                    help='Dataset root directory path')
parser.add_argument('--val_set_root', default='',
                    help='Dataset root directory path')
parser.add_argument('--save_folder', default='',
                    help='Directory for saving checkpoint models')
parser.add_argument('--save_img', default='',
                    help='Directory for saving checkpoint models')
parser.add_argument('--resume', default=False,
                    type=str, help='Checkpoint state_dict file to resume training from')

parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.97, type=float,
                    help='Gamma update for SGD')

parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if not os.path.exists(args.save_img):
    os.mkdir(args.save_img)

if args.visdom:
    import visdom
    viz = visdom.Visdom()

cfg = {'simple': [32, 64, 128, 256, 512],
       'complex':[64, 128, 256, 512, 1024]}
cfg512 = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def train():
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

    # wandb.login()
    # run = wandb.init(
    #     # Set the project where this run will be logged
    #     project="my-awesome-project",
    #     # Track hyperparameters and run metadata
    #     config={
    #         "learning_rate": args.lr,
    #         "epochs": 700,
    #     })

    if args.visdom:
        vis_title = 'Unet512'
        vis_legend = ['train', 'val']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    opt = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # train_set = DetecSegSet(args.train_set_root, transform=ODSegAugment(512), target_transform = ToPercentCoords())
    train_set = DetecSegSet(args.train_set_root, transform=None, target_transform=ToPercentCoords())
    data_loader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    val_set = DetecSegSet(args.val_set_root, transform=None, target_transform=ToPercentCoords())
    data_loader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    val_iterator = iter(data_loader_val)

    epoch = 0
    epoch_size = len(train_set) / args.batch_size
    while epoch < 200:
        loss_train_total = []
        loss_val_total = []
        for i, (img, box, mask) in enumerate(tqdm.tqdm(data_loader_train)):

            img, box, mask = img.to(device), box.to(device), mask.to(device)
            _, out_seg = net(Variable(img.unsqueeze(1).float(), requires_grad=True))

            loss_seg = SegLoss()
            train_loss = loss_seg(out_seg, mask)
            if i % 100 == 0:
                adjust_learning_rate(opt, args.gamma)
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            loss_train_total.append(train_loss.data.detach().numpy())

            try:
                img_val, box_val, mask_val = next(val_iterator)

            except:
                val_iterator = iter(data_loader_val)
                img_val, box_val, mask_val = next(val_iterator)

            _, out_seg_val = net(Variable(img_val.unsqueeze(1).float()))
            val_loss = loss_seg(out_seg_val, mask_val)
            loss_val_total.append(val_loss.data.detach().numpy())

            if args.visdom:
                update_vis_plot(i + epoch * epoch_size, train_loss, val_loss,
                                loss_train_total, loss_val_total,
                                iter_plot, epoch_plot, 'append', epoch_size)

            # wandb.log({"train_loss": train_loss.item(), "val_loss": val_loss.item()})

            if i % 5 == 0:
                print(f'{epoch}-{i}-train_loss==>>{train_loss.item()}-----val_loss==>>{val_loss.item()}')

            if i % 50 == 0:
                _img = img[0].squeeze(0)
                _out_seg = out_seg[0].squeeze(0) * 255
                _mask = mask[0].squeeze(0) * 255
                image = torch.cat((_img, _out_seg, _mask),dim=1).detach().numpy()
                cv2.imencode('.jpg', image)[1].tofile(args.save_img + str(epoch) + '_' + str(i) + '.jpg')
        epoch += 1

        if epoch % 1 == 0:
            torch.save(net.state_dict(), args.save_folder + str(epoch) + '.pth')
            print('save successfully!')



def adjust_learning_rate(optimizer, gamma):
    args.lr = args.lr * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr



def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias != None:
            m.bias.data.zero_()

def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 2)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, train, val, loss_train_total, loss_val_total, window1, window2, update_type, epoch_size):
    viz.line(
        X=torch.ones((1, 2)).cpu() * iteration,
        Y=torch.Tensor([train, val]).unsqueeze(0).cpu(),
        win=window1,
        update=update_type
    )
    if (iteration + 1) % epoch_size == 0:
        loss_train_total = sum(loss_train_total)/epoch_size
        loss_val_total = sum(loss_val_total)/epoch_size
        viz.line(
            X=torch.ones((1, 2)).cpu() * int(iteration / epoch_size),
            Y=torch.Tensor([loss_train_total, loss_val_total]).unsqueeze(0).cpu(),
            win=window2,
            update=update_type
        )


if __name__ == '__main__':
    train()
