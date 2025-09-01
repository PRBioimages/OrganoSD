import csv
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw

def json():
    import json

    def convert_to_tuple(input):
        xmin = int(min(input[::2]))
        xmax = int(max(input[::2]))
        ymin = int(min(input[1::2]))
        ymax = int(max(input[1::2]))
        points = []
        for i in range(len(input)):
            if i%2==0:
                x = int(input[i])
            else:
                y = int(input[i])
                point = [x, y]
                points.append(point)
        points = np.array(points).reshape((-1, 1, 2))
        return points, [xmin, ymin, xmax, ymax]


    def get_ellipse_param(major_radius, minor_radius, angle):
        '''
            根据椭圆的主轴和次轴半径以及旋转角度(默认圆心在原点)，得到椭圆参数方程的参数，
            椭圆参数方程为：
                A * x^2 + B * x * y + C * y^2 + F = 0
        '''
        a, b = major_radius, minor_radius
        sin_theta = np.sin(angle)
        cos_theta = np.cos(angle)
        A = a ** 2 * sin_theta ** 2 + b ** 2 * cos_theta ** 2
        B = 2 * (a ** 2 - b ** 2) * sin_theta * cos_theta
        C = a ** 2 * cos_theta ** 2 + b ** 2 * sin_theta ** 2
        F = -a ** 2 * b ** 2
        return A, B, C, F


    def calculate_rectangle(A, B, C, F):
        # 根据椭圆参数方程的参数，得到椭圆的外接矩形top-left和right-bottom坐标。
        # 椭圆上下外接点的纵坐标值
        y = np.sqrt(4 * A * F / (B ** 2 - 4 * A * C))
        y1, y2 = -np.abs(y), np.abs(y)
        # 椭圆左右外接点的横坐标值
        x = np.sqrt(4 * C * F / (B ** 2 - 4 * C * A))
        x1, x2 = -np.abs(x), np.abs(x)
        return (x1, y1), (x2, y2)

    def get_rectangle(major_radius, minor_radius, angle, center_x, center_y):
        # 按照数据集接口返回矩形框
        A, B, C, F = get_ellipse_param(major_radius, minor_radius, angle)
        p1, p2 = calculate_rectangle(A, B, C, F)
        return (int(center_x + p1[0]), int(center_y + p1[1])), (int(center_x + p2[0]), int(center_y + p2[1]))


    json_folder = "/home/hryang/Detecseg/images/1234/annotations.json"
    mask_folder = "/home/hryang/Detecseg/images/1234/result/"
    img_foder  ='/home/hryang/Detecseg/images/1234/img/'

    if not os.path.exists(mask_folder):
        os.mkdir(mask_folder)


    with open(json_folder, "r") as f:
        data = json.load(f)

    i = 0

    for imgname, anno in data.items():
        if anno.__len__ == 1:
            continue
        # mask = Image.new("RGB", (img_width, img_height), 0)
        # draw = ImageDraw.Draw(mask)
        imgpath = img_foder + imgname
        img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = np.zeros(img.shape, np.uint8)

        for shape in anno:
            if shape['type'] == 'polygon':
                points = shape['points']
                points, [xmin, ymin, xmax, ymax] = convert_to_tuple(points)
                mask = cv2.fillPoly(mask, [points], color=255)

            if shape['type'] == 'ellipse':
                cx = shape['cx']
                cy = shape['cy']
                rx = shape['rx']
                ry = shape['ry']
                angle = shape['angle']
                mask = cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, 255, -2)

        cv2.imencode('.jpg', mask)[1].tofile(os.path.join(mask_folder, imgname))
        i+=1




if __name__ == '__main__':
    json()
    # root = '/home/hryang/Detecseg/images/img/'
    # savepath = '/home/hryang/Detecseg/images/img_new/'
    # filelist = os.listdir(root)
    # for file in filelist:
    #     img = cv2.imread(root + file)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     img = cv2.resize(img, (512, 512))
    #     # cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
    #     cv2.imencode('.jpg', img)[1].tofile(os.path.join(savepath, '%s.jpg') % file)
