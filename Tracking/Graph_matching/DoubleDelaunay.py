from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import cv2
import pygmtools as pygm
import functools
import math


class Triangle:
    def __init__(self, vertices):
        self.vertices = vertices

    def is_triangle(self):
        a, b, c = self.calculate_side_lengths()
        if a + b > c and a + c > b and b + c > a:
            return True
        else:
            return False

    def calculate_side_lengths(self):
        # 计算三角形的边长
        side_lengths = []
        for i in range(3):
            j = (i + 1) % 3
            side_lengths.append(math.dist(self.vertices[i], self.vertices[j]))
        return side_lengths

    def calculate_angles(self):
        # 计算三角形的内角
        angles = []
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            a, b, c = self.calculate_side_lengths()
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            angles.append(math.degrees(angle))
        return angles

    def calculate_slope(self):
        slopes = []
        for i in range(3):
            j = (i + 1) % 3
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[j]
            if abs(x2 - x1) < 1e-5:
                slope = 1e5
            else:
                slope = (y2 - y1) / (x2 - x1)
            slopes.append(slope)
        return slopes

    def is_consistent_with(self, other_triangle, soft=0.4):
        # 检查两个三角形是否一致
        slope_s = sorted(self.calculate_slope())
        slope_o = sorted(other_triangle.calculate_slope())
        x = np.array([(slope_s[i] / slope_o[i] if abs(slope_o[i]) > 1e-5 else 1e5) for i in range(3)])
        if np.all(x < 1 + soft) and np.all(x > 1 - soft):
            return True
        return False


def find_consistent_triangle(points, reference_triangle):
    reference_triangle = Triangle(reference_triangle)
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            for k in range(j+1, len(points)):
                # 选取三个点作为候选三角形的顶点
                candidate_triangle = Triangle([points[i], points[j], points[k]])
                # 判断这三个点能否构成一个三角形
                if candidate_triangle.is_triangle():
                    # 检查该三角形是否与给定三角形一致
                    if candidate_triangle.is_consistent_with(reference_triangle):
                        return candidate_triangle
    return False


path1 = ''
path2 = ''
mypath = [path1, path2]
Tri = []
Box = []
for path in mypath:
    box = read_csv(path, names=['xmin', 'ymin', 'xmax', 'ymax']).to_numpy()
    x = (box[:,2]-box[:,0]) * (box[:,3]-box[:,1]) > 150
    box = box[x, :]
    points = np.zeros((len(box), 2))
    points[:, 0] = (box[:,0] + box[:,2])/2
    points[:, 1] = (box[:,1] + box[:,3])/2
    points = points.astype(np.uint16)
    tri = Delaunay(points)
    Tri.append(tri)
    Box.append(box)

Color = []
for tri in Tri:
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

label1 = np.arange(0, len(Tri[0].points), 1)
label2 = np.zeros((len(Tri[1].points),))

index = sorted(range(len(Color[0])), key=lambda i: Color[0].copy()[i])
for i in range(-1, -len(Color[0]), -1):
    Flag = []
    T = Tri[0].simplices[index[i]]
    T_point = Tri[0].points[T]
    T_label = label1[T]
    s = 50
    domain = [T_point[:, 0].min()-s, T_point[:, 1].min()-s, T_point[:, 0].max()+s, T_point[:, 1].max()+s]
    flag = (Tri[1].points[:, 0] > domain[0]) * (Tri[1].points[:, 0] < domain[2]) * (Tri[1].points[:, 1] > domain[1]) * (
                Tri[1].points[:, 1] < domain[3])
    x = Tri[1].points[flag, :]
    match_tri = find_consistent_triangle(Tri[1].points[flag, :], T_point)
    if match_tri:
        Flag.append(match_tri)

    plt.figure(figsize=(80, 40))
    plt.subplot(1, 2, 1)
    plt.tripcolor(Tri[0].points[:, 0], Tri[0].points[:, 1], Tri[0].simplices.copy(), facecolors=Color[0],
                  edgecolors='k')
    # plt.tripcolor(triangle_point[:, 0], triangle_point[:, 1], [0,1,2], facecolors=[1500.0], edgecolors='k')
    plt.scatter(T_point[:, 0], T_point[:, 1], s=300, color='r', alpha=0.5)

    for x, y, label in zip(Tri[0].points[:, 0], Tri[0].points[:, 1], label1):
        plt.annotate('%s' % label, xy=(x, y), xytext=(0, -10), textcoords='offset points', fontsize=30)

    plt.subplot(1, 2, 2)
    plt.scatter(Tri[1].points[:, 0], Tri[1].points[:, 1], s=100, color='k', alpha=0.5)

    plt.savefig('', transparent=True, dpi=100)