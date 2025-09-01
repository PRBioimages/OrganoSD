import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
from pandas import read_csv


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, mask=None):
        height, width = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, boxes, mask


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, mask=None):
        height, width = image.shape
        boxes = boxes.astype(np.float32)
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        return image, boxes, mask



class DetecSegSet(data.Dataset):
    def __init__(self, root, transform = None, target_transform = ToPercentCoords()):
        self.root = root
        self.img_path = root + '/img/'
        self.mask_path = root + '/mask_visual/'
        self.box_path = root + '/box/'
        self.transform = transform
        self.target_transform = target_transform
        self.names = ['xmin', 'ymin', 'xmax', 'ymax']
        self.file_name_list = os.listdir(self.img_path)

    def __len__(self):
        return len(self.file_name_list)

    def pull_imgname(self, index):
        fileName = os.path.splitext(self.file_name_list[index])[0]
        return fileName+'.jpg'

    def __getitem__(self, index):
        fileName = os.path.splitext(self.file_name_list[index])[0]
        imgpath = self.img_path + fileName + '.jpg'
        maskpath = self.mask_path + fileName + '.jpg'
        boxpath = self.box_path + fileName + '.csv'

        img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.fromfile(maskpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(bool).astype(np.uint8)
        boxes = read_csv(boxpath, names=self.names).to_numpy()

        if self.transform is not None:
            img, boxes, mask = self.transform(img, boxes, mask)
        if self.target_transform is not None:
            img, boxes, mask = self.target_transform(img, boxes, mask)
        boxes_ = np.zeros([120, 4], dtype=np.float32)
        boxes_[:len(boxes),:]=boxes

        return torch.tensor(img.copy()), boxes_, torch.tensor(np.array(mask.copy()))


