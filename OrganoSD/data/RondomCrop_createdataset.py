import cv2
import numpy as np
from numpy import random
from pandas import read_csv
import os
import csv


COLOR1 = [255,0,0]

class Resize(object):
    def __init__(self, size=732):
        self.size = size
    def __call__(self, image, boxes=None, mask=None):
        height, width = image.shape
        image = cv2.resize(image, (self.size,self.size))
        mask = cv2.resize(mask, (self.size,self.size))
        boxes[:, 1::2] = boxes[:, 1::2] / height * self.size
        boxes[:, ::2] = boxes[:, ::2] / width * self.size
        return image, boxes, mask

class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = [[0.1, 0.3],[0.3, 0.5],[0.5, 0.6]]

    def __call__(self, image, boxes=None, mask=None):
        height, width = image.shape
        i = random.randint(3)
        mode = self.sample_options[i]
        min, max = mode

        for _ in range(50):
            w = random.uniform(min * width, max*width)
            h = random.uniform(min * height, max*height)
            if h / w < 0.5 or h / w > 2:
                continue
            left = random.uniform(width - w)
            top = random.uniform(height - h)
            rect = np.array([int(left), int(top), int(left+w), int(top+h)])

            current_image = image[rect[1]:rect[3], rect[0]:rect[2]]
            current_mask = mask[rect[1]:rect[3], rect[0]:rect[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
            m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
            mask = m1 * m2
            if not mask.any():
                continue
            current_boxes = boxes[mask, :].copy()
            current_boxes[:, :2] = np.maximum(current_boxes[:, :2],rect[:2])
            current_boxes[:, :2] -= rect[:2]
            current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],rect[2:])
            current_boxes[:, 2:] -= rect[:2]
            return current_image, current_boxes, current_mask



if __name__ == '__main__':
    crop = RandomSampleCrop()
    resize = Resize(size = 512)

    root = ''
    saveroot = ''
    path = root + '/img'

    imgsave = saveroot + '/img'
    masksave = saveroot + '/mask'
    boxsave = saveroot + '/box'
    box_visual_save = saveroot + '/box_visual'
    mask_isual_save = saveroot + '/mask_visual'

    if not os.path.exists(imgsave):
        os.mkdir(imgsave)
    if not os.path.exists(masksave):
        os.mkdir(masksave)
    if not os.path.exists(boxsave):
        os.mkdir(boxsave)
    if not os.path.exists(box_visual_save):
        os.mkdir(box_visual_save)
    if not os.path.exists(mask_isual_save):
        os.mkdir(mask_isual_save)


    file_name_list = os.listdir(path)
    for i in range(len(file_name_list)):
        fileName = os.path.splitext(file_name_list[i])[0]
        imgpath = os.path.join(path, file_name_list[i])
        maskpath = root + '/mask/' + fileName + '.png'
        boxpath = root + '/box/' + fileName + '.csv'
        names = ['xmin', 'ymin', 'xmax', 'ymax']
        boxes_ = read_csv(boxpath, names=names)
        boxes_ = boxes_.to_numpy()

        img_ = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask_ = cv2.imdecode(np.fromfile(maskpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        for j in range(2):
            try:
                img, boxes, mask = crop(img_, boxes_, mask_)
                img, boxes, mask = resize(img, boxes, mask)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                savename = fileName + '_' +str(j+5)

                cv2.imencode('.jpg', mask)[1].tofile(os.path.join(masksave, '%s.jpg') % savename)
                cv2.imencode('.jpg', img)[1].tofile(os.path.join(imgsave, '%s.jpg') % savename)
                cv2.imencode('.jpg', mask*255)[1].tofile(os.path.join(mask_isual_save, '%s.jpg') % savename)

                for pt in boxes:
                    cv2.rectangle(img_rgb, (pt[0], pt[1]), (pt[2], pt[3]), COLOR1, 1)
                cv2.imencode('.jpg', img_rgb)[1].tofile(os.path.join(box_visual_save, '%s.jpg') % savename)


                with open(os.path.join(boxsave, '%s.csv') % savename, 'w', encoding='utf-8', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(boxes)
            except:
                continue




