from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import cv2
import pygmtools as pygm
import functools
import math
import time
import os

def main():
    root = '/home/hryang/Detecseg/'
    imgname1 = '对照4-0h_J5_1_001'
    imgname2 = '对照4-12h_J5_1_002'
    Label = match2imgs([root + imgname1 + '.csv', root + imgname2 + '.csv'], [root + imgname1 + '.jpg', root + imgname2 + '.jpg'])

    return None

def img2patches(Img, Box, HFs, num, patchsize):
    '''
        把图像分成 num_w * num_h 个patch，每个patch的宽高为 patch_w, patch_h;
        input:
            Img: 被裁剪图像；
            Box: Img中类器官所对应的边界框，是一个csv文件；
            HFs: 各个类器官的手工特征；
            num: 一个向量[num_w, num_h]，Img被分成 num_w * num_h 个patch;
            patchsize: 一个向量[patch_w, patch_h],定义输出patch的宽度与高度；
        output:
            PATCH: 裁剪得到的 num_w * num_h 个patch；
            BOX: 裁剪得到的patch所对应的边界框；
            TRI: 各个patch中类器官的三角剖分；
        （为使裁剪拼接后的图像没有缝隙，要求num * patchsize >= Img的边长）;
    '''
    num_w, num_h = num
    patch_w, patch_h = patchsize
    PATCH = []
    BOX = []
    TRI = []
    HF = []
    width = Img.shape[0]
    height = Img.shape[1]
    start_w = int((width - patch_w) / (num_w - 1))
    start_h = int((height - patch_h) / (num_h - 1))
    Crop_w = []
    Crop_h = []
    for i in range(num_w):
        crop_w = [start_w * i, start_w * i + patch_w]
        crop_h = [start_h * i, start_h * i + patch_h]
        Crop_w.append(crop_w)
        Crop_h.append(crop_h)
    for i in range(num_h):
        start_x = Crop_h[i][0]
        end_x = Crop_h[i][1]
        for j in range(num_w):
            start_y = Crop_w[j][0]
            end_y = Crop_w[j][1]
            img = Img[start_x:end_x, start_y:end_y]
            #中心坐标在裁剪范围内的类器官被保留
            center = np.zeros((len(Box), 2))
            center[:, 0] = (Box[:, 0] + Box[:, 2]) / 2
            center[:, 1] = (Box[:, 1] + Box[:, 3]) / 2
            flag = (center[:, 0] > start_y) * (center[:, 1] > start_x) * (center[:, 0] < end_y) * (center[:, 1] < end_x)
            box = Box[flag, :]
            hfs = HFs[flag, :]

            points = np.zeros((len(box), 2))
            points[:, 0] = center[flag, 0] - start_y
            points[:, 1] = center[flag, 1] - start_x
            points = points.astype(np.uint16)
            tri = Delaunay(points)

            PATCH.append(img)
            BOX.append(box)
            TRI.append(tri)
            HF.append(hfs)
    return PATCH, BOX, TRI, HF


def match2imgs(boxpathes, imgpathes=None):
    Patch = []
    Box = []
    Tri = []
    Hf7 = []
    Label = []
    Label_dict = []
    for boxpath, imgpath in zip(boxpathes, imgpathes):
        box1 = read_csv(boxpath, names=['xmin', 'ymin', 'xmax', 'ymax']).to_numpy()
        handcrafeted_fetures7 = np.ones((len(box1), 7))
        img1 = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)
        patches, boxes, tries, hfs7 = img2patches(img1, box1, handcrafeted_fetures7, [5, 5], [300, 300])
        Patch.append(patches)
        Box.append(boxes)
        Tri.append(tries)
        Hf7.append(hfs7)
    for i in range(len(Patch[0])):
        label, Label_point_dict = match2patchs([Tri[0][i], Tri[1][i]], [Box[0][i], Box[1][i]])
        Label.append(label)
        Label_dict.append(Label_point_dict)

        # Color = []
        # for tri in [Tri[0][i], Tri[1][i]]:
        #     color = []
        #     for index, sim in enumerate(tri.points[tri.simplices]):
        #         center = np.sum(tri.points[tri.simplices], axis=1) / 3.0
        #         cx, cy = center[index][0], center[index][1]
        #         x1, y1 = sim[0][0], sim[0][1]
        #         x2, y2 = sim[1][0], sim[1][1]
        #         x3, y3 = sim[2][0], sim[2][1]
        #         s = ((x1 - cx) ** 2 + (y1 - cy) ** 2) ** 0.5 + ((cx - x3) ** 2 + (cy - y3) ** 2) ** 0.5 \
        #             + ((cx - x2) ** 2 + (cy - y2) ** 2) ** 0.5
        #         color.append(s)
        #     color = np.array(color)
        #     Color.append(color)
        # plt.figure(figsize=(80, 40))
        # plt.subplot(1, 2, 1)
        # plt.imshow(Patch[0][i])
        # plt.tripcolor(Tri[0][i].points[:, 0], Tri[0][i].points[:, 1], Tri[0][i].simplices.copy(), facecolors=Color[0],
        #               edgecolors='k', alpha=0.05)
        # plt.scatter(Tri[0][i].points[:, 0], Tri[0][i].points[:, 1], s=80, color='r', alpha=0.5)
        # for x, y, label0 in zip(Tri[0][i].points[:, 0], Tri[0][i].points[:, 1], label[0]):
        #     plt.annotate(label0, xy=(x, y), xytext=(0, -10), textcoords='offset points', fontsize=50)
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(Patch[1][i])
        # plt.tripcolor(Tri[1][i].points[:, 0], Tri[1][i].points[:, 1], Tri[1][i].simplices.copy(), facecolors=Color[1],
        #               edgecolors='k', alpha=0.05)
        # plt.scatter(Tri[1][i].points[:, 0], Tri[1][i].points[:, 1], s=80, color='r', alpha=0.5)
        # for x, y, label1 in zip(Tri[1][i].points[:, 0], Tri[1][i].points[:, 1], label[1]):
        #     plt.annotate(label1, xy=(x, y), xytext=(0, -10), textcoords='offset points', fontsize=50)
        #
        # plt.tick_params(labelbottom='off', labelleft='off', left='off', right='off', bottom='off', top='off')
        # plt.savefig('/home/hryang/Detecseg/Delaunay.png', transparent=True, dpi=100)

    Label = label_sym(Label, Label_dict, make_patch_order([5, 5], 's_horizontal'))
    return None

def label_sym(Label, Label_dict, patch_order):
    numA = 0
    numB = 0
    numC = 0
    symA = dict()
    symB = dict()
    symC = dict()
    for k in patch_order:
        label_dict = Label_dict[k]
        label_dict = {i: label_dict[i] for i in sorted(label_dict)}
        for key in label_dict.keys():
            if key[0] == 'A':
                u = label_dict[key][:2]
                x = [value[:2] for value in symA.values()]
                if check_list_in_2d_list(label_dict[key][:2], [value[:2] for value in symA.values()]):
                    None
                else:
                    symA['A' + str(numA)] = label_dict[key]
                numA += 1
            if key[0] == 'B':
                symB['B' + str(numB)] = label_dict[key]
                numB += 1
            if key[0] == 'C':
                symC['C' + str(numC)] = label_dict[key]
                numC += 1
    return None



def check_list_in_2d_list(list, two_d_list):
    for sublist in two_d_list:
        if sublist == list:
            return True
    return False

def make_patch_order(num, mode):
    m, n = num
    order = np.arange(0, m * n)
    if mode == 'normal':
        None
    if mode == 's_horizontal':
        num = 0
        for i in range(m):
            if i % 2 == 0:
                for j in range(n):
                    order[n * i + j] = num
                    num += 1
            else:
                for j in range(n - 1, -1, -1):
                    order[n * i + j] = num
                    num += 1
    if mode == 's_vertical':
        num = 0
        for i in range(n):
            if i % 2 == 0:
                for j in range(m):
                    order[i + n * j] = num
                    num += 1
            else:
                for j in range(m - 1, -1, -1):
                    order[i + n * j] = num
                    num += 1
    return order

def match2patchs(Tri, Box, patch1=None, patch2=None):
    Matrix = []
    for tri in Tri:
        matrix = np.ones((len(tri.points), len(tri.points)))
        for i in range(len(tri.points)):
            matrix[i, i] = 0
        Matrix.append(matrix)
    conn1, _ = pygm.utils.dense_to_sparse(Matrix[0])
    conn2, _ = pygm.utils.dense_to_sparse(Matrix[1])
    edge1 = edgefeature(Tri[0].points[conn1[:, 0]], Tri[0].points[conn1[:, 1]])
    edge2 = edgefeature(Tri[1].points[conn2[:, 0]], Tri[1].points[conn2[:, 1]])
    node1 = nodefeature(Box[0])
    node2 = nodefeature(Box[1])
    gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)
    K = pygm.utils.build_aff_mat(node1, edge1, conn1, node2, edge2, conn2, node_aff_fn=gaussian_aff,
                                 edge_aff_fn=gaussian_aff)
    X = pygm.rrwm(K, len(Matrix[0]), len(Matrix[1]))
    X = pygm.hungarian(X)

    label1 = np.arange(0, len(Tri[0].points), 1)
    label2 = np.zeros(len(Tri[1].points))
    for i in range(len(label2)):
        x = np.where(X[:, i] > 0)
        label2[i] = 1000 if len(x[0]) < 1 else x[0][0]
    label2 = label2.astype(np.uint16)
    Label, Label_point = label_trans(Tri, label1, label2, re_point_num=7)
    return Label, Label_point


def nodefeature(box):
    # 'width, height, center_x, center_y'
    feature = np.zeros((len(box), 4))
    feature[:, 0] = box[:, 2] - box[:, 0]
    feature[:, 1] = box[:, 3] - box[:, 1]
    feature[:, 2] = (box[:, 0] + box[:, 2]) / 2
    feature[:, 3] = (box[:, 1] + box[:, 3]) / 2
    for i in range(4):
        max = feature[:, i].max()
        min = feature[:, i].min()
        feature[:, i] = (feature[:, i] - min) / (max - min)
    return feature


def edgefeature(point1, point2):
    # 'feature = [slope, length, angle, position_x, position_y]'
    feature = np.zeros((len(point1), 5))
    x1, y1 = point1[:, 0], point1[:, 1]
    x2, y2 = point2[:, 0], point2[:, 1]
    flag1 = abs(x2 - x1) >= 1e-2
    flag2 = abs(x2 - x1) < 1e-2
    feature[flag1, 0] = (y2[flag1] - y1[flag1]) / (x2[flag1] - x1[flag1])
    feature[flag2, 0] = 1e2
    for i in range(len(point1)):
        feature[i, 1] = math.degrees(math.atan(feature[i, 0]))
        feature[i, 2] = math.dist(point1[i, :], point2[i, :])
    feature[:, 3] = x2 - x1
    feature[:, 4] = y2 - y1
    for i in range(5):
        if i == 1:
            for j in range(len(feature[:, i])):
                feature[j, i] = math.sin(0.75 * feature[j, i] / math.pi) / 2
        else:
            max = feature[:, i].max()
            min = feature[:, i].min()
            feature[:, i] = (feature[:, i] - min) / (max - min)
    return feature[:, 1:]



def edgefeature1(point1, point2):
    feature = np.zeros((len(point1), 4))
    x1, y1 = point1[:, 0], point1[:, 1]
    x2, y2 = point2[:, 0], point2[:, 1]
    feature[:, 0] = x1
    feature[:, 1] = y1
    feature[:, 2] = x2
    feature[:, 3] = y2
    return feature


def find5point(point, points, label, num):
    '''
    找到points中距离点point最近的num个点，返回这些点的标签
    :param point:
    :param points:
    :param label:
    :param num:
    :return:
    '''
    distances = np.linalg.norm(points - point, axis=1)
    closest5 = np.argsort(distances)[1:num + 1]
    label5 = label[closest5]
    return set(label5)


def label_trans(Tri, label1, label2, re_point_num=7):
    '''
    标签转换和检查，A(配对完成的类器官)，B(前一帧中有，后一帧中没有)，C(前一帧没有，后一帧有)
    :param Tri:
    :param label1: 前一帧的标签列表
    :param label2: 后一帧的标签列表
    :param re_point_num:当前后两帧中被配对类器官周围 re_point_num 个类器官中有一半以上重合时，二者才配对
    :return:
        Label: 转化后的标签
    '''
    numA = 0
    numB = 0
    numC = 0
    Label = np.zeros((max(len(label1), len(label2)), 2))
    Label = np.array([list(map(str, Label[:, 0])), list(map(str, Label[:, 1]))])
    Label_point = dict()
    for i in range(len(label2)):
        if label2[i] == 1000:
            Label[1, i] = 'C' + str(numC)
            numC += 1
            Label_point['C' + str(numC)] = np.append(np.array([None, None]), Tri[1].points[i, :])
    for i in range(len(label1)):
        given_point_f1 = Tri[0].points[i, :]
        point_f1 = find5point(given_point_f1, Tri[0].points[:, :], label1, re_point_num)
        index_f2 = None if len(np.where(label2 == i)[0]) < 1 else np.where(label2 == i)[0][0]
        if index_f2!=None:
            given_point_f2 = Tri[1].points[index_f2, :]
            point_f2 = find5point(given_point_f2, Tri[1].points[:, :], label2, re_point_num)
            repeat_num = point_f1 & point_f2
            if len(repeat_num) > re_point_num / 2 - 1:
                Label[0, i] = 'A' + str(numA)
                Label[1, index_f2] = 'A' + str(numA)
                Label_point['A' + str(numA)] = np.append(Tri[0].points[i, :], Tri[1].points[index_f2, :])
                numA += 1
            else:
                Label[0, i] = 'B' + str(numB)
                Label[1, index_f2] = 'C' + str(numC)
                Label_point['B' + str(numB)] = np.append(Tri[0].points[i, :], [None, None])
                Label_point['C' + str(numC)] = np.append([None, None], Tri[1].points[index_f2, :])
                numB += 1
                numC += 1
        else:
            Label[0, i] = 'B' + str(numB)
            Label_point['B' + str(numB)] = np.append(Tri[0].points[i, :], [None, None])
            numB += 1
    if len(label1) < len(label2):
        for i in range(len(label2)):
            if Label[0, i] == '0.0':
                Label[0, i] = 'B' + str(numB)
                Label_point['B' + str(numB)] = np.array([None, None, None, None])
                numB += 1
    else:
        for i in range(len(label1)):
            if Label[1, i] == '0.0':
                Label[1, i] = 'C' + str(numC)
                Label_point['C' + str(numC)] = np.array([None, None, None, None])
                numC += 1
    return Label, Label_point





if __name__ == '__main__':
    main()