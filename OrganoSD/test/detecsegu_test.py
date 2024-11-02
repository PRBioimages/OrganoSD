import cv2
from PIL import Image
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model.detecseg_u import UNet512, ODSeg_U
from data.dataset import DetecSegSet, ToPercentCoords
import argparse
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='DetecSeg Testing With Pytorch')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--test_setroot', default='',
                    help='Dataset root directory path')
parser.add_argument('--resume', default='',
                    type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--save_img', default="",
                    help='Directory for saving checkpoint models')
parser.add_argument('--testreport', default='wpc.txt', type=str)
args = parser.parse_args()

labelmap = ('background', 'organoid')


def test_net(save_folder, net, testset, device, thresh):
    filename = save_folder+args.testreport
    num_images = len(testset)
    for i in range(num_images):
        img, box_gt, mask_gt = testset.__getitem__(i)
        box_gt = box_gt[box_gt[:,2]-box_gt[:,0]>0, :] * 512
        box_gt = box_gt.astype(np.uint16)
        imgname = testset.pull_imgname(i)
        print(f'Testing image {i + 1}/{num_images}....{imgname}')
        img = img.unsqueeze(0).unsqueeze(1).to(device)
        pre_detec, pre_seg = net(Variable(img.float(), requires_grad=False))
        mask_pre = pre_seg[0].squeeze(0) * 255
        mask_pre = mask_pre.detach().numpy().astype(np.uint16)
        mask_pre = cv2.cvtColor(mask_pre, cv2.COLOR_RGB2BGR)
        box_pre = pre_detec.data

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+imgname+'\n')
            for box in box_gt:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')

        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img = Image.open(os.path.join(args.test_setroot, 'img/' + imgname))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        colorg = (0, 255, 0)
        colorp = (255, 0, 0)
        for j in range(len(box_gt)):
            cv2.rectangle(img, (box_gt[j, 0], box_gt[j, 1]), (box_gt[j, 2], box_gt[j, 3]), colorg, 1)

        pred_num = 0
        for i in range(box_pre.size(1)):
            j = 0
            while box_pre[0, i, j, 0] >= thresh:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = box_pre[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (box_pre[0, i, j, 1:]*512).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                cv2.rectangle(img, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), colorp, 1)
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1
        img = np.concatenate((img, mask_pre), axis=1)
        cv2.imwrite(os.path.join(args.save_img, imgname), img)


def test():
    # load net
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg1013 = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    net = ODSeg_U('test', 1, 1, cfg1013, num_classes)
    net = net.to(device)
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_weights(args.resume)
        print('successful load weight！')

    net.eval()
    # load data
    testset = DetecSegSet(root=args.test_setroot, transform=None, target_transform = ToPercentCoords())
    test_net(args.save_img, net, testset, device = device, thresh=0.3)

if __name__ == '__main__':
    test()
    # mask = np.zeros((512, 512))
    # mask[:150, :150] = 1
    # mask[150:180, 150:180] = 1
    # cv2.imencode('.jpg', mask * 255)[1].tofile('/home/hryang/Detecseg/train_img/240319/Amask.jpg')

