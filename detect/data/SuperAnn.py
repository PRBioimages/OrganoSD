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


    json_folder = "/home/hryang/Detecseg/images/annotations_Cap.json"
    mask_folder = "/home/hryang/Detecseg/images/img/result/"
    box_folder = "/home/hryang/Detecseg/images/img/result/"
    img_foder  ='/home/hryang/Detecseg/images/img_new/'

    if not os.path.exists(mask_folder):
        os.mkdir(mask_folder)
    if not os.path.exists(box_folder):
        os.mkdir(box_folder)

    # for json_file in os.listdir(json_folder):
    #     if not json_file.endswith(".json"):  # 看看取出来的图片格式是不是json格式的文件，如果不是的话，直接取下一个文件
    #         continue
    with open(json_folder, "r") as f:
        data = json.load(f)

    img_width = 512
    img_height = 512
    # img_width = 3800
    # img_height = 9000

    # color = (25, 225, 75)
    color1 = (171, 7, 20)
    color = (255, 255, 255)
    i = 0

    for imgname, anno in data.items():
        if anno.__len__ == 1:
            continue
        # mask = Image.new("RGB", (img_width, img_height), 0)
        # draw = ImageDraw.Draw(mask)
        imgpath = img_foder + imgname
        img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.zeros_like(img, dtype=np.uint8)

        boxes = []
        for shape in anno:
            if shape['type'] == 'polygon':
                points = shape['points']
                points, [xmin, ymin, xmax, ymax] = convert_to_tuple(points)
                boxes.append([xmin, ymin, xmax, ymax])
                mask = cv2.fillPoly(mask, [points], color=color)
                # cv2.drawContours(img, [points], -1, color, 2)
                # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

            if shape['type'] == 'ellipse':
                cx = shape['cx']
                cy = shape['cy']
                rx = shape['rx']
                ry = shape['ry']
                angle = shape['angle']
                mask = cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, color, -2)
                # cv2.ellipse(img, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, color, 2)
                # cv2.rectangle(img, (int(cx - rx), int(cy - ry)), (int(cx + rx), int(cy + ry)), color, 2)
        _, thresh = cv2.threshold(mask[:,:,0], 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(img, [contour], -1, color1, 2)
            mask = cv2.fillPoly(mask, [contour], color=color1)
        alpha = 0.1
        atten = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        image = cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)
        cv2.imencode('.jpg', image)[1].tofile(os.path.join(mask_folder, imgname))
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
