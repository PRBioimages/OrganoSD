import cv2
import time
import argparse
from os import path
import os
import os.path as osp
import pandas as pd
from pandas import read_csv
import numpy as np
from PIL import Image
import csv
import re
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Object Tracking Based Searching')
parser.add_argument('--img_root', default="/home/hryang/test/data_exp/tracking/img/",
                    type=str, help='待跟踪的图像存储路径')
parser.add_argument('--box_root', default="/home/hryang/test/data_exp/tracking/res/",
                    type=str, help='待跟踪的边界框存储路径')
parser.add_argument('--metrics_root', default="",
                    type=str, help='待跟踪的特征手工特征存储路径')
parser.add_argument('--save_root', default='/home/hryang/test/data_exp/tracking/',
                    type=str, help='跟踪结果保存路径')
parser.add_argument('--register', default=False,
                    type=bool, help='是否在跟踪前进行图像配准')
parser.add_argument('--searching_radius', default=100,
                    type=int, help='搜索半径')
parser.add_argument('--ratio_dist', default=0.5,
                    type=float, help='距离的重要性比值')
parser.add_argument('--sim_thresh', default=0.05,
                    type=float, help='相似性容忍阈值')
args = parser.parse_args()


def compute_dist_matrix(boxi, boxj, metrici, metricj, index_col, radius, ratio_dis):
    # 获取框的数量
    n_i = len(boxi)
    n_j = len(boxj)
    # 计算框的中心点
    cxi = (boxi[:, 0] + boxi[:, 2]) / 2
    cyi = (boxi[:, 1] + boxi[:, 3]) / 2
    cxj = (boxj[:, 0] + boxj[:, 2]) / 2
    cyj = (boxj[:, 1] + boxj[:, 3]) / 2
    # 如果没有metrici，计算框的面积并进行归一化处理
    if not metrici:
        sizei = (boxi[:, 2] - boxi[:, 0]) * (boxi[:, 3] - boxi[:, 1])  # 框的面积
        sizej = (boxj[:, 2] - boxj[:, 0]) * (boxj[:, 3] - boxj[:, 1])  # 框的面积
        # 使用广播计算归一化面积
        max_size = np.maximum(np.max(sizei), np.max(sizej))
        min_size = np.minimum(np.min(sizei), np.min(sizej))
        size_range = max_size - min_size if max_size - min_size != 0 else 1  # 避免除零错误
        sizei = (sizei - min_size) / size_range
        sizej = (sizej - min_size) / size_range
    # 计算中心点之间的距离（向量化计算）
    dist_matrix = np.sqrt((cxi[:, None] - cxj[None, :]) ** 2 + (cyi[:, None] - cyj[None, :]) ** 2)
    # 计算相似度：如果没有metrici，基于面积差异计算；否则计算特征向量的差异
    if not metrici:
        similarity_matrix = np.abs(sizei[:, None] - sizej[None, :])
    else:
        similarity_matrix = np.linalg.norm(np.array(metrici)[:, None] - np.array(metricj)[None, :], axis=2)
    valid_mask_dist = dist_matrix <= radius  # dist_matrix小于等于radius
    valid_mask_index = np.zeros((n_i, n_j), dtype=bool)
    valid_mask_index[:, index_col] = True  # 只保留index_col列为有效
    valid_mask = valid_mask_dist & valid_mask_index
    # 计算符合条件的 dist 和 similarity
    dist_valid = dist_matrix[valid_mask]
    similarity_valid = similarity_matrix[valid_mask]
    # 归一化距离和相似度
    dist_norm = (dist_valid - np.min(dist_valid)) / (np.max(dist_valid) - np.min(dist_valid))  # 归一化距离
    dist_norm = 1 - dist_norm
    similarity_norm = (similarity_valid - np.min(similarity_valid)) / (
                np.max(similarity_valid) - np.min(similarity_valid))
    # 结果矩阵
    matrix = np.zeros((n_i, n_j), dtype=np.float32)
    matrix[valid_mask] = (1 - ratio_dis) * similarity_norm + ratio_dis * dist_norm
    return matrix


def main(args):
    directories_box = [d for d in os.listdir(args.box_root) if os.path.isdir(os.path.join(args.box_root, d))]
    # directories_img = [d for d in os.listdir(args.img_root) if os.path.isdir(os.path.join(args.img_root, d))]
    for directory in directories_box:
        boxes_root = os.path.join(args.box_root, directory)
        # imgs_root = os.path.join(args.img_root, directory)
        metrics_root = os.path.join(args.metrics_root, directory)

        # IMG = []
        BOX = []
        METRIC = []
        for filename in os.listdir(boxes_root):
            boxes_path = os.path.join(boxes_root, filename)
            boxes = np.genfromtxt(boxes_path, delimiter=',', dtype=float, skip_header=1)
            BOX.append(boxes)
            # img_path = os.path.join(imgs_root, os.path.splitext(filename)[0] + '.tif')
            # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # IMG.append(image)
            if args.metrics_root:
                metric_path = os.path.join(metrics_root, filename)
                metric = np.genfromtxt(metric_path, delimiter=',', dtype=float, skip_header=1)
            else:
                metric = []
            METRIC.append(metric)
        max_length = max(len(inner_list) for inner_list in BOX)
        ID = np.zeros((max_length, len(BOX)), dtype=np.uint16)
        ID[:len(BOX[0]), 0] = list(range(1, len(BOX[0]) + 1))
        Num = len(BOX[0])
        i = 0
        j = 1
        index_col = tuple([True] * len(BOX[1]))
        # 开始匹配过程
        while j < len(BOX):
            Sim = compute_dist_matrix(BOX[i], BOX[j], METRIC[i], METRIC[j], index_col, args.searching_radius, args.ratio_dist)
            n = min(len(BOX[i]), len(np.where(np.any(Sim != 0, axis=0))[0]))
            flag = 1
            while n > 0 and flag > args.sim_thresh:
                max_sim = np.max(Sim)
                row, col = np.where(Sim == max_sim)
                if np.any(ID[:len(BOX[j]),j] == ID[row[0],i]):
                    Sim[row[0],col[0]] = 0
                    flag = max_sim
                else:
                    ID[col[0],j] = ID[row[0],i]
                    Sim[row[0],:] = 0
                    Sim[:, col[0]] = 0
                    n -= 1
                    flag = max_sim
            # 若还没有匹配完，看i-1帧中能不能匹配
            if np.any(ID[:len(BOX[j]), j] <= 0):
                if i > 0:
                    i -= 1
                    index_col = np.where(ID[:len(BOX[j]), j] == 0)[0]
                # 若i-1已经是第一帧，停止这一帧的匹配，为剩余的类器官分配新的ID
                else:
                    zero_positions = np.where(ID[:len(BOX[j]), j] == 0)[0]
                    new_values = np.arange(Num + 1, len(zero_positions) + Num + 1)
                    ID[zero_positions, j] = new_values
                    Num += len(zero_positions)
                    i = j
                    j += 1
                    index_col = tuple([True] * len(BOX[j])) if j < len(BOX) else 0
            # 若全部匹配，接着处理j+1帧
            else:
                i = j
                j += 1
                index_col = tuple([True] * len(BOX[j])) if j < len(BOX) else 0
        df = pd.DataFrame(ID)
        df.to_csv(os.path.join(args.save_root, directory + '.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main(args)