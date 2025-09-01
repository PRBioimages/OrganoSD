import json
import cv2
import numpy as np
from PIL import Image
import os
import csv


COLOR1 = [255,0,0]
path = 'C:/Users/Dell/Desktop/dataset/gao/box'
file_name_list = os.listdir(path)
for i in range(len(file_name_list)):
    fileName = os.path.splitext(file_name_list[i])[0]
    source = os.path.join(path, file_name_list[i])
    with open(source, encoding='utf-8') as f:
        data = json.load(f)
        objs = data['annotation']['objects']
        boxs = []
        for obj in objs:
            pos = obj['points']['exterior']
            xmin = pos[0][0]
            ymin = pos[0][1]
            xmax = pos[1][0]
            ymax = pos[1][1]
            box = [xmin, ymin, xmax, ymax]
            boxs.append(box)
    save = path + '/' + fileName + '.csv'
    with open(save, 'w', encoding='utf-8', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(boxs)

