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
    root = ''
    imgname1 = ''
    imgname2 = ''
    Label = match2imgs([root + imgname1 + '.csv', root + imgname2 + '.csv'], [root + imgname1 + '.jpg', root + imgname2 + '.jpg'])

    return None

def Img2patches(Img, Box, num, patchsize):
    PATCH = []
    BOX = []
    TRI = []
    width = Img.shape[0]
    start = int((width - patchsize) / (num - 1))
    Crop = []
    for i in range(num):
        crop = [start * i, start * i + patchsize]
        Crop.append(crop)
    for i in range(num):
        start_x = Crop[i][0]
        end_x = Crop[i][1]
        for j in range(num):
            start_y = Crop[j][0]
            end_y = Crop[j][1]
            img = Img[start_x:end_x, start_y:end_y]

            flag = (Box[:, 0] > start_y) * (Box[:, 1] > start_x) * (Box[:, 2] < end_y) * (Box[:, 3] < end_x)
            box = Box[flag, :]

            points = np.zeros((len(box), 2))
            points[:, 0] = (box[:, 0] + box[:, 2]) / 2 - start_y
            points[:, 1] = (box[:, 1] + box[:, 3]) / 2 - start_x
            points = points.astype(np.uint16)
            tri = Delaunay(points)

            PATCH.append(img)
            BOX.append(box)
            TRI.append(tri)
    return PATCH, BOX, TRI


def match2imgs(boxpathes, imgpathes=None):
    Patch = []
    Box = []
    Tri = []
    for boxpath, imgpath in zip(boxpathes, imgpathes):
        box1 = read_csv(boxpath, names=['xmin', 'ymin', 'xmax', 'ymax']).to_numpy()
        img1 = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)
        patches, boxes, tries = Img2patches(img1, box1, 5, 300)
        Patch.append(patches)
        Box.append(boxes)
        Tri.append(tries)
    for i in range(len(Patch[0])):
        label, numA, numB, numC = match2patchs([Tri[0][i], Tri[1][i]], [Box[0][i], Box[1][i]])

        Color = []
        for tri in [Tri[0][i], Tri[1][i]]:
            color = []
            for index, sim in enumerate(tri.points[tri.simplices]):
                center = np.sum(tri.points[tri.simplices], axis=1) / 3.0
                cx, cy = center[index][0], center[index][1]
                x1, y1 = sim[0][0], sim[0][1]
                x2, y2 = sim[1][0], sim[1][1]
                x3, y3 = sim[2][0], sim[2][1]
                s = ((x1 - cx) ** 2 + (y1 - cy) ** 2) ** 0.5 + ((cx - x3) ** 2 + (cy - y3) ** 2) ** 0.5 \
                    + ((cx - x2) ** 2 + (cy - y2) ** 2) ** 0.5
                color.append(s)
            color = np.array(color)
            Color.append(color)
        plt.figure(figsize=(80, 40))
        plt.subplot(1, 2, 1)
        plt.imshow(Patch[0][i])
        plt.tripcolor(Tri[0][i].points[:, 0], Tri[0][i].points[:, 1], Tri[0][i].simplices.copy(), facecolors=Color[0],
                      edgecolors='k', alpha=0.05)
        plt.scatter(Tri[0][i].points[:, 0], Tri[0][i].points[:, 1], s=80, color='r', alpha=0.5)
        for x, y, label0 in zip(Tri[0][i].points[:, 0], Tri[0][i].points[:, 1], label[0]):
            plt.annotate(label0, xy=(x, y), xytext=(0, -10), textcoords='offset points', fontsize=50)

        plt.subplot(1, 2, 2)
        plt.imshow(Patch[1][i])
        plt.tripcolor(Tri[1][i].points[:, 0], Tri[1][i].points[:, 1], Tri[1][i].simplices.copy(), facecolors=Color[1],
                      edgecolors='k', alpha=0.05)
        plt.scatter(Tri[1][i].points[:, 0], Tri[1][i].points[:, 1], s=80, color='r', alpha=0.5)
        for x, y, label1 in zip(Tri[1][i].points[:, 0], Tri[1][i].points[:, 1], label[1]):
            plt.annotate(label1, xy=(x, y), xytext=(0, -10), textcoords='offset points', fontsize=50)

        plt.tick_params(labelbottom='off', labelleft='off', left='off', right='off', bottom='off', top='off')
        plt.savefig('', transparent=True, dpi=100)


    return None

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
    Label = label_trans(Tri, label1, label2, re_point_num=7)
    return Label


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


def find5point(point, points, label, num):
    distances = np.linalg.norm(points - point, axis=1)
    closest5 = np.argsort(distances)[1:num + 1]
    label5 = label[closest5]
    return set(label5)


def label_trans(Tri, label1, label2, re_point_num=7):
    numA = 0
    numB = 0
    numC = 0
    Label = np.zeros((max(len(label1), len(label2)), 2))
    Label = np.array([list(map(str, Label[:, 0])), list(map(str, Label[:, 1]))])
    for i in range(len(label2)):
        if label2[i] == 1000:
            Label[1, i] = 'C' + str(numC)
            numC += 1
    for i in range(len(label1)):
        given_point_f1 = Tri[0].points[i, :]
        point_f1 = find5point(given_point_f1, Tri[0].points[:, :], label1, re_point_num)
        index_f2 = None if len(np.where(label2 == i)[0]) < 1 else np.where(label2 == i)[0][0]
        if index_f2:
            given_point_f2 = Tri[1].points[index_f2, :]
            point_f2 = find5point(given_point_f2, Tri[1].points[:, :], label2, re_point_num)
            repeat_num = point_f1 & point_f2
            if len(repeat_num) > re_point_num / 2:
                Label[0, i] = 'A' + str(numA)
                Label[1, index_f2] = 'A' + str(numA)
                numA += 1
            else:
                Label[0, i] = 'B' + str(numB)
                Label[1, index_f2] = 'C' + str(numC)
                numB += 1
                numC += 1
        else:
            Label[0, i] = 'B' + str(numB)
            numB += 1
    if len(label1) < len(label2):
        for i in range(len(label2)):
            if Label[0, i] == '0.0':
                Label[0, i] = 'B' + str(numB)
                numB += 1
    else:
        for i in range(len(label1)):
            if Label[1, i] == '0.0':
                Label[1, i] = 'C' + str(numC)
                numC += 1
    return Label, numA, numB, numC

def label_sym(Label):
    return None



if __name__ == '__main__':
    main()