import cv2
import tqdm
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn import functional as F

from reconstruction.ResAE.dataloaders.datasets import Organoiddataset
from reconstruction.ResAE.models.CAE import AE, AEadd, VAE
from reconstruction.ResAE.models.AE_residual import ResAE, ResVAE
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss

import argparse
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='DetecSeg Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--train_set_root', default='/home/hryang/single/',
                    help='Dataset root directory path')
parser.add_argument('--val_set_root', default='/home/hryang/single/',
                    help='Dataset root directory path')
parser.add_argument('--save_folder', default='/home/hryang/Whole/result/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--mode', default='N',
                    help='V:VAE while N:normal')
parser.add_argument('--modelname', default='ResAE',
                    help='Choose a model')
parser.add_argument('--resume', default=False,    #'/home/hryang/Whole/result/ResAE40_1e-05/30.pth'
                    type=str, help='Checkpoint state_dict file to resume training from')

parser.add_argument('--latent_dim', default=80, type=int,
                    help='latent dim')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,   #VAE40:8e-1
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.97, type=float,
                    help='Gamma update for SGD')
args = parser.parse_args()

savefoder = args.save_folder + args.modelname + f'{args.latent_dim}' + '_' + '2' + '/'
if not os.path.exists(savefoder):
    os.mkdir(savefoder)

class Loss_inter(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, pred, truth, rec):
        mseloss = MSELoss()
        crossloss = CrossEntropyLoss()
        l1loss = L1Loss()
        loss = 0
        for index in range(len(truth)):
            # xmin = xmax = 0
            # img = truth[index].detach().numpy()
            # for i in range(210):
            #     for j in range(210):
            #         if img[0, i, j] != 0:
            #             xmin = i + 1
            #             ymin = j + 1
            #             break
            #     if xmin > 0:
            #         break
            # for i in np.arange(419, 210, -1):
            #     for j in np.arange(419, 210, -1):
            #         if img[0, i, j] != 0:
            #             xmax = i + 1
            #             ymax = j + 1
            #             break
            #     if xmax > 0:
            #         break
            x1 = 210 - int(rec[0][index]/2) -1
            x2 = 210 + int(rec[0][index] / 2) + 1
            y1 = 210 - int(rec[1][index] / 2) - 1
            y2 = 210 + int(rec[1][index] / 2) + 1
            loss += l1loss(pred[index, 0, x1:x2, y1:y2], truth[index, 0, x1:x2, y1:y2].float())
        return loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    def forward(self, pred, truth):
        mseloss = MSELoss(size_average=None, reduce=None, reduction='mean')
        l1loss = L1Loss()
        # loss = l1loss(pred, truth)
        # loss = mseloss(pred, truth)
        loss = F.binary_cross_entropy(pred, truth, reduction='sum')
        return loss

def loss_vae(out, img, mu, logvar):
    cons = F.binary_cross_entropy(out, img, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return cons + KLD

def getmodel(modelname, latentdim):
    if modelname in ['AE']:
        return AE(latentdim)
    elif modelname in ['AEadd']:
        return AEadd(latentdim)
    elif modelname in ['VAE']:
        return VAE(latentdim)
    elif modelname in ['ResAE']:
        return ResAE(latentdim)
    elif modelname in ['ResVAE']:
        return ResVAE(latentdim)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = getmodel(args.modelname, args.latent_dim)
    net = net.to(device)

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_weights(args.resume)
        print('successful load weight！')
    else:
        print('Initializing weights...')
        net.apply(weights_init)

    opt = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    loss_func = Loss()

    train_ds = Organoiddataset(args.train_set_root, mode='train')
    train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)

    valid_ds = Organoiddataset(args.val_set_root, mode='train')
    valid_dl = DataLoader(dataset=valid_ds, batch_size=args.batch_size, shuffle=True)
    val_iterator = iter(valid_dl)

    epoch = 0
    net.train()

    while epoch < 1000:
        loss_total = []
        loss_val_total = []
        adjust_learning_rate(opt, args.gamma)
        for i, (img, rec, maxmin) in enumerate(tqdm.tqdm(train_dl)):
            img = img.float().to(device)

            if args.mode == 'N':
                out, latent = net(Variable(img, requires_grad=True))
                train_loss = loss_func(out, img)
            else:
                out, latent, mu, logvar = net(Variable(img, requires_grad=True))
                train_loss = loss_vae(out, img, mu, logvar)
            # x = img.detach().numpy()
            # y = out.detach().numpy()

            opt.zero_grad()
            train_loss.backward()
            opt.step()
            loss_total.append(train_loss.data.detach().numpy())
            # net.apply(up_show)
            if i % 20 == 0:
                try:
                    img_val, rec_val, maxmin_val = next(val_iterator)
                except:
                    val_iterator = iter(valid_dl)
                    img_val, rec_val, maxmin_val = next(val_iterator)
                img_val = img_val.float().to(device)

                if args.mode == 'N':
                    out_val, latent_val = net(Variable(img_val.float().to(device), requires_grad=True))
                    val_loss = loss_func(out_val, img_val)
                else:
                    out_val, latent_val, mu_val, logvar_val = net(Variable(img_val, requires_grad=True))
                    val_loss = loss_vae(out_val, img_val, mu_val, logvar_val)

                loss_val_total.append(val_loss.data.detach().numpy())
                print(f'{epoch}-{i}-train=={train_loss.item():.4f}----val=={val_loss.item():.4f}')
                out = out_val[0].detach().numpy()*255
                img = img_val[0].detach().numpy()*255
                image = np.hstack((img.astype(int), out.astype(int)))[0]
                cv2.imencode('.jpg', image)[1].tofile(savefoder + 'Reconstruction.jpg')
        epoch += 1
        if epoch % 100 == 0:
            torch.save(net.state_dict(), savefoder + str(epoch) + '.pth')
            print('save successfully!')


def draw_box(img, res, color):
    if img.ndim == 2:
        img = cv2.cvtColor(img.detach().numpy(), cv2.COLOR_GRAY2BGR)
    else:
        img = img.detach().numpy()
    width = img.shape[0]
    res = (res.detach().numpy() * width).astype(np.uint16)
    for pt in res:
        if pt[2] > pt[0] and pt[3] > pt[1]:
            cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)
    return img

def adjust_learning_rate(optimizer, gamma):
    args.lr = args.lr * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

def xavier(param):
    init.xavier_uniform(param)

def weights_init(net):
    if isinstance(net, nn.Conv2d):
        xavier(net.weight.data)
        if net.bias != None:
            net.bias.data.zero_()
    for m in net.children():
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            if m.bias != None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Module):
            weights_init(m)

def param_show(name, param):
    if param.grad is not None:
        print(f"Parameter {name} has been updated.")
    else:
        print(f"Parameter {name} has not been updated.")

def up_show(net):
    for name, param in net.named_parameters():
        param_show(name, param)
    for m in net.children():
        up_show(m)



if __name__ == '__main__':
    train()
