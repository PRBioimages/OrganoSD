import torch
from torch.autograd import Variable
import cv2
import time
# from imutils.video import FPS, WebcamVideoStream
import argparse
import sys
from os import path
import os
import os.path as osp
import pandas as pd
from pandas import read_csv
import numpy as np
from PIL import Image
from data import BaseTransform
from ssd import build_ssd
import csv
import re
from utils.augmentation_organ import jaccard_numpy
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Object Tracking')
parser.add_argument('--img_source_path', default="E:\\YangHR\\oganoid\\data\\20220621\\2073-高俊钊-鼻咽癌C5拍照\\",
                    type=str)
parser.add_argument('--res_source_path', default="E:\\YangHR\\oganoid\\result\\ObjDe_Gao_res\\",
                    type=str)
parser.add_argument('--img_save_folder', default='E:\\YangHR\\oganoid\\result\\Tracking_Gao_img\\', type=str,
                    help='Dir to save results')
parser.add_argument('--res_save_folder', default="E:\\YangHR\\oganoid\\result\\Tracking_Gao_res\\",
                    type=str)
parser.add_argument('--crop_save_folder', default="E:\\YangHR\\oganoid\\result\\Tracking_Gao_crop\\",
                    type=str)
args = parser.parse_args()

if not os.path.exists(args.img_save_folder):
    os.mkdir(args.img_save_folder)

if not os.path.exists(args.res_save_folder):
    os.mkdir(args.res_save_folder)

if not os.path.exists(args.crop_save_folder):
    os.mkdir(args.crop_save_folder)


COLOR1 = (255, 0, 0)
COLOR2 = (0, 255, 0)
COLOR3 = (0, 0, 255)


def Move_r_d(mask_orin, r, d, step):
    r_step = r * step
    d_step = d * step
    mask = np.zeros((1200, 1200))
    if r_step >= 0 and d_step >= 0:
        mask[:1200 - d_step, :1200 - r_step] = mask_orin[d_step:1200, r_step:1200]
    if r_step > 0 and d_step < 0:
        mask[- d_step:1200, :1200 - r_step] = mask_orin[:1200 + d_step, r_step:1200]
    if r_step < 0 and d_step > 0:
        mask[:1200 - d_step, - r_step:1200] = mask_orin[d_step:1200, :1200 + r_step]
    if r_step <= 0 and d_step <= 0:
        mask[ - d_step:1200, - r_step:1200] = mask_orin[:1200 + d_step, :1200 + r_step]
    return mask


def Registration_IoU(Sque, file_name_list, res_source_path, move_range=12, step=2):
    mask = []
    res = []
    for Time in range(8):
        i = Sque * 8 + Time
        fileName = os.path.splitext(file_name_list[i])[0]
        res_source = os.path.join(res_source_path, fileName + '.csv')
        mask_ = np.zeros((1200, 1200))
        names = ['xmin', 'ymin', 'xmax', 'ymax']
        data = read_csv(res_source, names=names)
        for i in range(len(data.xmin)):
            mask_[data.ymin[i]:data.ymax[i], data.xmin[i]:data.xmax[i]] = 1
        mask.append(mask_)
        res.append(data)
    RD_pre = []
    for i in range(7):
        Inter = np.zeros((move_range * 2 + 1, move_range * 2 + 1))
        for r in range(-move_range, move_range + 1):
            for d in range(-move_range, move_range + 1):
                mask_next = Move_r_d(mask[i + 1], r, d, step)
                Inter[d + move_range, r + move_range] = np.sum(mask[i] * mask_next, (0, 1))
        d, r = np.asarray(divmod(np.argmax(Inter), move_range * 2 + 1)) - move_range
        RD_pre.append([r * step, d * step])

    RD_first = np.cumsum(np.asarray(RD_pre), axis=0)
    r = max(RD_first[:, 0]) if max(RD_first[:, 0]) > 0 else 0
    l = - min(RD_first[:, 0]) if min(RD_first[:, 0]) < 0 else 0
    d = max(RD_first[:, 1]) if max(RD_first[:, 1]) > 0 else 0
    u = - min(RD_first[:, 1]) if min(RD_first[:, 1]) < 0 else 0
    Crop = [[r, d, 1200 - l, 1200 - u]]
    for i in range(7):
        crop = [r - RD_first[i, 0], d - RD_first[i, 1], 1200 - l - RD_first[i, 0], 1200 - u - RD_first[i, 1]]
        Crop.append(crop)
    Res_crop = []
    for i in range(8):
        res_crop = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax'], index=[])
        for k in range(len(res[i])):
            if res[i].xmin[k]<Crop[i][0] or res[i].ymin[k]<Crop[i][1] or res[i].xmax[k]>Crop[i][2] or res[i].ymax[k]>Crop[i][3]:
                None
            else:
                res_crop = res_crop.append({'xmin':res[i].xmin[k]-Crop[i][0], 'ymin':res[i].ymin[k]-Crop[i][1],
                                            'xmax':res[i].xmax[k]-Crop[i][0], 'ymax':res[i].ymax[k]-Crop[i][1]}, ignore_index=True)
        Res_crop.append(res_crop)
    return Crop, Res_crop


def No_Registration(Sque, file_name_list, res_source_path):
    Res_crop = []
    for Time in range(8):
        i = Sque * 8 + Time
        fileName = os.path.splitext(file_name_list[i])[0]
        res_source = os.path.join(res_source_path, fileName + '.csv')
        names = ['xmin', 'ymin', 'xmax', 'ymax']
        data = read_csv(res_source, names=names)
        Res_crop.append(data)
    return Res_crop



def Color_rand(n):
    Color = []
    for i in range(n):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        color = (b, g, r)
        Color.append(color)
    return Color



def Select(Res_crop, thresh=300):
    res_sque = []
    Match = 1000*np.ones((1000,8))
    for i in range(len(Res_crop[0])):
        Match[i, 0] = i
    ID = len(Res_crop[0])
    for Time in range(7):
        match = []
        Res_crop0  =Res_crop[Time]
        Res_crop1 = Res_crop[Time + 1]
        Dis = np.ones((len(Res_crop0), len(Res_crop1)))
        # M0 = Res_crop0.max(axis=1)
        # m0 = Res_crop0.min(axis=1)
        # M1 = Res_crop1.max(axis=1)
        # m1 = Res_crop1.min(axis=1)
        for i in range(len(Res_crop0)):
            for j in range(len(Res_crop1)):
                # Res_0 = (Res_crop0.iloc[i] - m0[i]) / (M0[i] - m0[i])
                # Res_1 = (Res_crop1.iloc[j] - m1[j]) / (M1[j] - m1[j])
                # dis_min = abs(Res_1[0] - Res_0[0]) + abs(Res_1[1] - Res_0[1])
                # dis_max = abs(Res_1[2] - Res_0[2]) + abs(Res_1[3] - Res_0[3])
                dis_min = abs(Res_crop1.xmin[j] - Res_crop0.xmin[i]) + abs(Res_crop1.ymin[j] - Res_crop0.ymin[i])
                dis_max = abs(Res_crop1.xmax[j] - Res_crop0.xmax[i]) + abs(Res_crop1.ymax[j] - Res_crop0.ymax[i])
                Dis[i,j] = dis_min + dis_max
        r, c = divmod(np.argmin(Dis), len(Res_crop1))
        while Dis[r,c] < thresh:
            Dis[r,:] = 5 * thresh
            Dis[:, c] = 5 * thresh
            r, c = divmod(np.argmin(Dis), len(Res_crop1))
            match.append([r,c])
        for i in range(len(match)):
            Match[np.where(Match[:,Time] == match[i][0]), Time+1] = match[i][1]
        unident = np.setdiff1d(np.arange(len(Res_crop[Time+1])), Match[:, Time+1])
        for i in range(len(unident)):
            Match[ID + i, Time + 1] = unident[i]
        ID = ID + len(unident)

    Match = Match.astype(np.uint16)
    for i in range(ID):
        box_sque = []
        for Time in range(8):
            if Match[i, Time] == 1000:
                box= [0,0,0,0]
            else:
                box = [Res_crop[Time].xmin[Match[i, Time]], Res_crop[Time].ymin[Match[i, Time]],
                       Res_crop[Time].xmax[Match[i, Time]], Res_crop[Time].ymax[Match[i, Time]]]
            box_sque.extend(box)
        res_sque.append(box_sque)
    return res_sque

if __name__ == '__main__':
    file_name_list = os.listdir(args.img_source_path)
    for Sque in range(int(len(file_name_list)/8)):
        t0 = time.time()
        sequence_name = re.split('-\d', file_name_list[Sque * 8], 1)[0]
        Crop, res_crop = Registration_IoU(Sque, file_name_list, args.res_source_path, move_range=6, step=3)
        res_sque = Select(res_crop)
        Color = Color_rand(len(res_sque))

        for i in range(8):
            fileName = os.path.splitext(file_name_list[Sque * 8 + i])[0]
            imgpath = args.img_source_path + fileName +'.tif'
            img_rgb = Image.open(imgpath)
            img_rgb = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            img_rgb = cv2.resize(img_rgb, (1200,1200))
            img_rgb_crop = img_rgb[Crop[i][1]:Crop[i][3], Crop[i][0]:Crop[i][2], :]
            for j in range(len(res_sque)):
                cv2.rectangle(img_rgb_crop, (res_sque[j][i*4], res_sque[j][i*4+1]), (res_sque[j][i*4+2], res_sque[j][i*4+3]), Color[j], 2)
            cv2.imencode('.jpg', img_rgb_crop)[1].tofile(os.path.join(args.img_save_folder, '%s.jpg') % fileName)

        t1 = time.time()
        print('time of sequence: %.4f sec.' % (t1 - t0))
        with open(os.path.join(args.res_save_folder, sequence_name + '.csv'), 'w', encoding='utf-8', newline="") as file:
            writer = csv.writer(file)
            writer.writerows(res_sque)
        with open(os.path.join(args.crop_save_folder, 'crop_' + sequence_name + '.csv'), 'w', encoding='utf-8', newline="") as file:
            writer = csv.writer(file)
            writer.writerows(Crop)
