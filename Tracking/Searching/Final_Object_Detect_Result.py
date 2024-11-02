from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
import argparse
import sys
from os import path
import os
import numpy as np
from PIL import Image
from data import BaseTransform
from ssd import build_ssd
from utils.augmentation_organ import jaccard_numpy
import csv
import warnings
warnings.filterwarnings("ignore")

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--img_save_folder', default='',
                    type=str, help='Dir to save results')
parser.add_argument('--res_save_folder', default='',
                    type=str)
parser.add_argument('--img_source_path', default="",
                    type=str)
parser.add_argument('--weights', default="",
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')

args = parser.parse_args()

if not os.path.exists(args.img_save_folder):
    os.mkdir(args.img_save_folder)

if not os.path.exists(args.res_save_folder):
    os.mkdir(args.res_save_folder)

COLOR1 = (255, 0, 0)
COLOR2 = (0, 255, 0)
COLOR3 = (0, 0, 255)



def deImg(img,nimg, step):
    "把图片img分成nimg*nimg份，并返回IMG"
    img = cv2.resize(img,((nimg-1)*(300-step)+300, (nimg-1)*(300-step)+300))
    IMG = []
    for i in range(nimg):
        for j in range(nimg):
            img1 = img[(300-step)*i: (300-step)*i+300, (300-step)*j: (300-step)*j+300]
            IMG.append(img1)
    return IMG




def de_overlapbox(res, thresh):
    i = 0
    while i != len(res)-1:
        overlap = jaccard_numpy(res[i + 1: len(res)], res[i])
        max_over = max(overlap)
        if max_over > thresh:
            res = np.delete(res, i, 0)
        else:
            i += 1
    return res

def Iomin(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    inter = inter[:, 0] * inter[:, 1]
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    min_area = np.minimum(area_a, area_b)
    return inter / min_area, area_a, area_b



def de_IoMin(res, thresh):
    i = 0
    while i != len(res)-1:
        iomin, area_alli, area_i = Iomin(res[i + 1: len(res)], res[i])
        mask_iomin = (iomin > thresh).astype(np.int)
        area_all = np.insert(area_alli * mask_iomin, 0, area_i, axis = 0)
        area_all[np.argmax(area_all)] = 0
        flag = max(area_all)
        if flag != 0:
            where = tuple(np.asarray(np.where(area_all != 0)) + i)
            res = np.delete(res, where, 0)
        else:
            i += 1
    return res



def predict(net, transform, imgpath, nimg, savename, savepath, step=100, thresh=0.5, area_lim=89, bound_lim=1, imgsize_flag = True):
    t0 = time.time()
    img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    IMG = deImg(img, nimg, step)
    images = np.hstack((torch.from_numpy(transform(IMG[k])[0]).unsqueeze(1) for k in range(len(IMG))))
    images_mask = images < 75
    bl_pxl_num = np.sum(images_mask, (0, 2))
    im_mean_xz = bl_pxl_num < 20000
    newsize = (nimg - 1) * (300 - step) + 300

    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    img = 255 - img
    IMG = deImg(img, nimg, step)

    img_rgb = Image.open(imgpath)
    img_rgb = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    if imgsize_flag:
        img_rgb = cv2.resize(img_rgb, (1200, 1200))
    else:
        img_rgb = cv2.resize(img_rgb, ((nimg-1)*(300-step)+300, (nimg-1)*(300-step)+300))

    images = np.hstack((torch.from_numpy(transform(IMG[k])[0]).unsqueeze(1) for k in range(len(IMG))))
    images = Variable(torch.from_numpy(images).unsqueeze(0))
    images = torch.permute(images, (2, 0, 1, 3))
    height, width = images.shape[2:4]

    y = torch.empty(0, 2, 200, 5)
    pred_num = 32
    if nimg ** 2 < pred_num:
        y = net(images)
    else:
        for i in range(int(nimg ** 2 / pred_num) + 1):
            if i == int(nimg ** 2 / pred_num):
                y_a = net(images[pred_num * i:nimg ** 2 + 1, :, :, :])
            else:
                y_a = net(images[pred_num * i:pred_num * (i + 1), :, :, :])
            y = torch.cat([y, y_a], dim=0)
    detections = y.data


    scale = torch.Tensor([width, height, width, height])
    res = []
    for k in range(detections.size(0)):
        if im_mean_xz[k]:
            for i in range(detections.size(1)):
                for j in range (detections.size(2)):
                    while detections[k, i, j, 0] >= thresh:
                        pt = (detections[k, i, j, 1:] * scale).cpu().numpy()
                        for m in range(4):
                            if pt[m] < 0:
                                pt[m] = 1
                            if pt[m] > 300:
                                pt[m] = 299
                        ptmin = min(pt)
                        ptmax = max(pt)
                        area = (pt[2] - pt[0]) * (pt[3] - pt[1])
                        hw_ratio = (int(pt[2]) - int(pt[0])) / (int(pt[3]) - int(pt[1]))
                        if ptmin <= bound_lim or ptmax >= (300 - bound_lim) or area > area_lim**2 or hw_ratio > 1.25 or hw_ratio < 0.8:
                            None
                        else:
                            shang, yushu = divmod(k, nimg)
                            if imgsize_flag:
                                pt_img = [int((pt[0]+(300-step)*yushu)*1200/newsize), int((pt[1]+(300-step)*shang)*1200/newsize),
                                          int((pt[2]+(300-step)*yushu)*1200/newsize), int((pt[3]+(300-step)*shang)*1200/newsize)]
                            else:
                                pt_img = [int(pt[0]) + (300 - step) * yushu, int(pt[1]) + (300 - step) * shang,
                                          int(pt[2]) + (300 - step) * yushu, int(pt[3]) + (300 - step) * shang]
                            res.append(pt_img)
                        j += 1
                    else:
                        None

    res = de_IoMin(np.array(res), 0.6)
    for pt in res:
        cv2.rectangle(img_rgb, (pt[0], pt[1]), (pt[2], pt[3]), COLOR1, 2)
    cv2.imencode('.jpg', img_rgb)[1].tofile(os.path.join(savepath, '%s.jpg') % savename)

    with open(os.path.join(args.res_save_folder, savename + '.csv'), 'w', encoding='utf-8', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(res)

    t1 = time.time()
    print('timer: %.4f sec.' % (t1 - t0))
    print(os.path.join('Finish image ', savename))
    return None



if __name__ == '__main__':
    net = build_ssd('test', 300, 2)
    net.load_state_dict(torch.load(args.weights, map_location='cpu'))
    transform = BaseTransform(net.size, 0)

    file_name_list = os.listdir(args.img_source_path)
    for i in range(len(file_name_list)):
        fileName = os.path.splitext(file_name_list[i])[0]  # 分割，不带后缀名
        img_source = os.path.join(args.img_source_path, file_name_list[i])
        predict(net.eval(), transform, imgpath=img_source, nimg=9, savename=fileName, savepath=args.img_save_folder)
