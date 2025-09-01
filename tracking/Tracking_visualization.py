import cv2
import argparse
import os
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Object Tracking')
parser.add_argument('--ID_path', default="/home/hryang/test/data_exp/tracking/sequence1.csv",
                    type=str, help='ID号存储路径')
parser.add_argument('--box_root', default="/home/hryang/test/data_exp/tracking/res/",
                    type=str, help='边界框存储目录')
parser.add_argument('--img_root', default="/home/hryang/test/data_exp/tracking/img/",
                    type=str, help='图像存储目录')
parser.add_argument('--save_root', default='/home/hryang/test/data_exp/tracking/result/',
                    type=str, help='结果保存路径')
args = parser.parse_args()

def Color_rand(n):
    Color = []
    for i in range(n):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        color = (b, g, r)
        Color.append(color)
    return Color



if __name__ == '__main__':
    directories_box = [d for d in os.listdir(args.box_root) if os.path.isdir(os.path.join(args.box_root, d))]
    for directory in directories_box:
        img_root = os.path.join(args.img_root, directory)
        ID = np.genfromtxt(args.ID_path, delimiter=',', dtype=float, skip_header=1)
        Color = Color_rand(int(ID.max().max()) + 1)

        result_root = os.path.join(args.save_root, directory)
        if not os.path.exists(result_root):
            os.mkdir(result_root)

        for i in range(len(os.listdir(img_root))):
            file = os.listdir(img_root)[i]
            img_path = os.path.join(img_root, file)
            img = Image.open(img_path)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            filename = os.path.splitext(file)[0]
            boxes_path = os.path.join(args.box_root, directory, filename+'.csv')
            boxes = np.genfromtxt(boxes_path, delimiter=',', dtype=float, skip_header=1)

            for j in range(len(ID[:, i])):
                x1, y1, x2, y2 = boxes[j]
                color = Color[int(ID[j, i])]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.imencode('.jpg', img)[1].tofile(os.path.join(result_root, '%s.jpg') % filename)
            print(f"Processed {filename}!")
        print("All images processed!")





