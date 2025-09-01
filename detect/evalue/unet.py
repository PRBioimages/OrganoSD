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
from sklearn.metrics import confusion_matrix

from model.detecseg_u import UNet512
from model.detecseg_y import build_odseg, UNet
from data.dataset import DetecSegSet, ToPercentCoords
from data.augment import NoAugmentation

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

parser.add_argument('--test_set_root', default='E:/YangHR/oganoid/data/train512',
                    type=str, help='Dataset root directory path')
parser.add_argument('--resume', default='E:/YangHR/oganoid/code/ODSeg/params/unet512_norm/92.pth',
                    type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--save_img', default='E:/YangHR/oganoid/code/ODSeg/test_img/unet/',
                    type=str, help='Directory for saving checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_img):
    os.mkdir(args.save_img)


cfg = {'simple': [32, 64, 128, 256, 512],
       'complex':[64, 128, 256, 512, 1024]}
cfg512 = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def test():
    num_classes = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet512(1, cfg512, num_classes)
    net = net.to(device)

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_weights(args.resume)
        print('successful load weight！')
    else:
        print('Please provide weights...')

    dataset = DetecSegSet(args.test_set_root, transform=NoAugmentation(), target_transform=ToPercentCoords())
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    IoU_m = []
    Acc_m = []
    Precision_m = []
    Recall_m = []
    Dice_m = []
    for i, (img, box, mask) in enumerate(data_loader):
        img, box, mask = img.to(device), box.to(device), mask.to(device)
        _, out_seg = net(Variable(img.unsqueeze(1).float(), requires_grad=True))

        pred = out_seg.squeeze(1).detach().numpy()
        gt = mask.squeeze(1).detach().numpy()
        batch = pred.shape[0]
        for b in range(batch):
            iou = IoU(pred[b], gt[b])
            cm = Confusion_Matrix(pred, gt)
            acc = Acc(cm)
            precision = Precision(cm)
            recall = Recall(cm)
            dice = Dice(cm)

            IoU_m.append(iou)
            Acc_m.append(acc)
            Precision_m.append(precision)
            Recall_m.append(recall)
            Dice_m.append(dice)

            i_m = sum(IoU_m) / (i + 1)
            a_m = sum(Acc_m) / (i + 1)
            p_m = sum(Precision_m) / (i + 1)
            r_m = sum(Recall_m) / (i + 1)
            d_m = sum(Dice_m) / (i + 1)
            print(f'{i}-{b}-IoU={iou}---acc={acc}---precision={precision}---recall={recall}---dice={dice}')


        _img = img[0].squeeze(0) * 255
        _out_seg = out_seg[0].squeeze(0) * 255
        _mask = mask[0].squeeze(0) * 255
        image = torch.cat((_img, _out_seg, _mask), dim=1).detach().numpy()
        cv2.imencode('.jpg', image)[1].tofile(args.save_img + str(int(i)) + '.jpg')







def IoU(pred, gt):
    # pred[pred >= 0.1] = 1
    # pred[pred < 0.1] = 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        iou = float('nan')
    else:
        iou = float(intersection) / float(union)
    return iou


def Confusion_Matrix(pred, gt):
    pred[pred >= 0.1] = 1
    pred[pred < 0.1] = 0
    pred = pred.astype(np.uint8).flatten()
    gt = gt.flatten()
    cm = confusion_matrix(gt, pred)
    return cm

def Acc(cm):
    acc = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])
    return acc

def Precision(cm):
    p = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    return p

def Recall(cm):
    r = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    return r

def Dice(cm):
    dice = 2*cm[0,0] / (2*cm[0,0] + cm[0,1] + cm[1,0])
    return dice


def compute_miou(seg_preds, seg_gts, num_classes):
    ious = []
    for i in range(len(seg_preds)):
        ious.append(IoU(seg_preds[i], seg_gts[i], num_classes))
    ious = np.array(ious, dtype=np.float32)
    miou = np.nanmean(ious, axis=0)
    return miou





if __name__ == '__main__':
    test()