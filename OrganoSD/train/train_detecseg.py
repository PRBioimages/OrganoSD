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

from model.detecseg_y import build_odseg
from model.detecseg_u import ODSeg_U
from data.dataset import DetecSegSet, ToPercentCoords
from data.augment import ODSegAugment
from loss.multibox_loss import DetecLoss, SegLoss, total_loss
from config import organ

import argparse
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='DetecSeg Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--train_set_root', default='',
                    help='Dataset root directory path')
parser.add_argument('--val_set_root', default='',
                    help='Dataset root directory path')
parser.add_argument('--save_folder', default='',
                    help='Directory for saving checkpoint models')
parser.add_argument('--save_img', default='',
                    help='Directory for saving checkpoint models')
parser.add_argument('--resume', default='',
                    type=str, help='Checkpoint state_dict file to resume training from')


parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.97, type=float,
                    help='Gamma update for SGD')

parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if not os.path.exists(args.save_img):
    os.mkdir(args.save_img)

if args.visdom:
    import visdom
    viz = visdom.Visdom()

COLOR1 = [120, 30, 255]
COLOR2 = [180, 235, 30]


def train():
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = build_odseg('train', 732, num_classes, unetcfg = '732complex', extracfg = '732complex', add = '732complex', mbox_ = '732complex')
    cfg512 = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    cfg1013 = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    net = ODSeg_U('train', 1, 1, cfg512, num_classes)
    net = net.to(device)

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_weights(args.resume)
        print('successful load weight！')
    else:
        print('Initializing weights...')
        net.apply(weights_init)

    if args.visdom:
        vis_title = 'DetecSeg_Wpc'
        vis_legend = ['Loc Loss', 'Conf Loss', 'Seg Loss', 'Train', 'val', 'Cons Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    opt = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train_set = DetecSegSet(args.train_set_root, transform=None, target_transform=ToPercentCoords())
    data_loader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    val_set = DetecSegSet(args.val_set_root, transform=None, target_transform=ToPercentCoords())
    data_loader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    val_iterator = iter(data_loader_val)

    epoch = 0
    epoch_size = len(train_set) / args.batch_size
    net.train()

    while epoch < 200:
        loss_l_total = []
        loss_c_total = []
        loss_s_total = []
        loss_cs_total = []
        loss_val_total = []
        adjust_learning_rate(opt, args.gamma)
        for i, (img, box, mask) in enumerate(tqdm.tqdm(data_loader_train)):
            img, box, mask = img.to(device), box.to(device), mask.to(device)
            # train_loss = 1
            # thresh = sum(sum(mask[0].squeeze(0))) / (2.5 * 512**2)
            # i = -1
            # while train_loss > thresh:
            #     i = i + 1
            out_detect, out_seg = net(Variable(img.unsqueeze(1).float(), requires_grad=True))

            loss_s, loss_c, loss_l, pos_priors, neg_priors, loss_cs = total_loss(out_detect, out_seg, box, mask)
            train_loss = loss_l + loss_c + loss_s + loss_cs
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            loss_s_total.append(loss_s.data.detach().numpy())
            loss_l_total.append(loss_l.data.detach().numpy())
            loss_c_total.append(loss_c.data.detach().numpy())
            loss_cs_total.append(loss_cs.data.detach().numpy())
            # net.apply(up_show)
            try:
                img_val, box_val, mask_val = next(val_iterator)

            except:
                val_iterator = iter(data_loader_val)
                img_val, box_val, mask_val = next(val_iterator)

            out_detect_val, out_seg_val = net(Variable(img_val.unsqueeze(1).float(), requires_grad=False))
            loss_s_, loss_c_, loss_l_, __, _, loss_cs_= total_loss(out_detect_val, out_seg_val, box_val, mask_val)
            val_loss = loss_l_ + loss_c_ + loss_s_ + loss_cs_
            loss_val_total.append(val_loss.data.detach().numpy())

            if args.visdom:
                update_vis_plot(i+epoch*epoch_size, loss_l, loss_c, loss_s, loss_cs, val_loss,
                                loss_l_total, loss_c_total, loss_s_total, loss_cs_total, loss_val_total,
                                iter_plot, epoch_plot, 'append', epoch_size)

            print(f'{epoch}-{i}=train=={train_loss.item():.4f}==s={loss_s.item():.4f}==c={loss_c.item():.4f}'
                      f'==l={loss_l.item():.4f}==cons={loss_cs.item():.4f}'
                      f'=-----=val=={val_loss.item():.4f}')

            if i % 20 == 0:
                _img = img[0].squeeze(0)
                # _img = torch.as_tensor(cv2.cvtColor(np.array(_img), cv2.COLOR_GRAY2BGR))
                img_box = torch.as_tensor(cv2.cvtColor(_img.detach().numpy(), cv2.COLOR_GRAY2BGR))
                # img_box = torch.as_tensor(draw_box(_img, pos_priors[0], COLOR1))
                # img_box = torch.as_tensor(draw_box(img_box, neg_priors[0], COLOR2))
                # cv2.imencode('.jpg', img_box.detach().numpy())[1].tofile('/home/hryang/Detecseg/train_img/240319/Adetect.jpg')

                _out_seg = out_seg[0].squeeze(0) * 255
                # loc_data, conf_data, priors = out_detect[0][0], out_detect[1][0], out_detect[2][0]
                # pos = conf_data > 0.7
                # pos_idx = pos.expand_as(loc_data)
                # loc_p = loc_data[pos_idx].view(-1, 4)
                # _out_detec = torch.as_tensor(draw_box(_img, loc_p, COLOR1))
                _out_seg_ = torch.as_tensor(cv2.cvtColor(_out_seg.detach().numpy(), cv2.COLOR_GRAY2BGR))

                _mask = mask[0].squeeze(0) * 255
                truths = box[0].data
                truths = truths[truths[:, 0] + truths[:, 1] + truths[:, 2] + truths[:, 3] > 0]
                _mask = torch.as_tensor(draw_box(_mask, truths, COLOR1))

                img_ini = torch.as_tensor(cv2.cvtColor(_img.detach().numpy(), cv2.COLOR_GRAY2BGR))
                image = torch.cat((img_box, img_ini, _out_seg_, _mask),dim=1).detach().numpy()
                cv2.imencode('.jpg', image)[1].tofile(args.save_img + str(epoch) + '_' + str(i) + '.jpg')

            # if i % 1000 == 0:
            #     torch.save(net.state_dict(), args.save_folder + str(epoch) + '_' + str(i) + '.pth')
            #     print('save successfully!')
        epoch += 1

        if epoch % 1 == 0:
            torch.save(net.state_dict(), args.save_folder + str(epoch) + '.pth')
            print('save successfully!')




def draw_box(img, res, color):
    if img.ndim==2:
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

def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 6)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, seg, cons, val_loss,
                    loss_l_total, loss_c_total, loss_s_total, loss_cs_total, loss_val_total,
                    window1, window2, update_type, epoch_size):
    viz.line(
        X=torch.ones((1, 6)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, seg, seg, val_loss, cons]).unsqueeze(0).cpu(),
        win=window1,
        update=update_type
    )
    if (iteration + 1) % epoch_size == 0:
        loss_l_total = sum(loss_l_total)/epoch_size
        loss_c_total = sum(loss_c_total)/epoch_size
        loss_s_total = sum(loss_s_total)/epoch_size
        loss_cs_total = sum(loss_cs_total)/epoch_size
        loss_train_total = loss_s_total #+ loss_c_total + loss_l_total +loss_cs_total
        loss_val_total = sum(loss_val_total)/epoch_size
        viz.line(
            X=torch.ones((1, 6)).cpu() * int(iteration / epoch_size),
            Y=torch.Tensor([loss_l_total, loss_c_total, loss_s_total,
                            loss_train_total, loss_val_total, loss_cs_total]).unsqueeze(0).cpu(),
            win=window2,
            update=update_type
        )


if __name__ == '__main__':
    train()
