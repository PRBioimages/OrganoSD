import cv2
import numpy as np
from numpy import random
from pandas import read_csv
import os
import csv

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


class CenterCrop(object):
    def __init__(self):
        self.sample_options = [800, 1400, 1800, 2200, 2600, 3000, 3400]
        self.filter_thresh = 0.12

    def __call__(self, image_shape=None, boxes=None):
        height, width = image_shape
        crops = []
        boxes_sets = []

        for i in range(boxes.shape[0]):
            crop_wide = self.sample_options[random.randint(3)]
            box = boxes[i]
            c_x, c_y = int((box[2] + box[0])/2), int((box[3] + box[1])/2)
            crop_top = c_x - int(crop_wide/2); crop_bottom = c_x + int(crop_wide/2); crop_left = c_y - int(crop_wide/2); crop_right = c_y + int(crop_wide/2)
            if c_x - crop_wide/2 < 0:
                crop_top = 0; crop_bottom = crop_wide
            if c_x + crop_wide/2 > height:
                crop_top = height - crop_wide; crop_bottom = height
            if c_y - crop_wide/2 < 0:
                crop_left = 0; crop_right = crop_wide
            if c_y + crop_wide/2 > width:
                crop_left = width - crop_wide; crop_right = width
            crop = np.array([crop_top, crop_left, crop_bottom, crop_right])

            f0 = (crop_top <= boxes[:, 0]) * (boxes[:, 0] <= crop_bottom)
            f1 = (crop_left <= boxes[:, 1]) * (boxes[:, 1] <= crop_right)
            f2 = (crop_top <= boxes[:, 2]) * (boxes[:, 2] <= crop_bottom)
            f3 = (crop_left <= boxes[:, 3]) * (boxes[:, 3] <= crop_right)
            flag = f0 * f1 + f2 * f3 + f0 * f3 + f2 * f1
            if not flag.any():
                continue
            current_boxes = boxes[flag, :].copy()
            pre_x, pre_y = current_boxes[:, 2] - current_boxes[:, 0], current_boxes[:, 3] - current_boxes[:, 1]
            current_boxes[:, :2] = np.maximum(current_boxes[:, :2], crop[:2])
            current_boxes[:, :2] -= crop[:2]
            current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], crop[2:])
            current_boxes[:, 2:] -= crop[:2]
            post_x, post_y = current_boxes[:, 2] - current_boxes[:, 0], current_boxes[:, 3] - current_boxes[:, 1]
            flag = (post_x / pre_x > self.filter_thresh) * (post_y / pre_y > self.filter_thresh)
            current_boxes = current_boxes[flag, :]
            crops.append(crop)
            boxes_sets.append(current_boxes)

        return crops, boxes_sets




# if __name__ == '__main__':
#     COLOR1 = [225, 95, 45]
#     centercrop = CenterCrop()
#     resize = Resize(size = 512)
#
#     root = 'C:/Users/Dell/Desktop/label/Wpc'
#     saveroot = 'E:/YangHR/oganoid/data/Wpcval'
#     imgname = 'Wpc'
#     path = root + '/Wpc/images/'
#
#     imgsave = saveroot + '/img'
#     masksave = saveroot + '/mask'
#     boxsave = saveroot + '/box'
#     box_visual_save = saveroot + '/box_visual'
#     mask_isual_save = saveroot + '/mask_visual'
#
#     if not os.path.exists(imgsave):
#         os.mkdir(imgsave)
#     if not os.path.exists(masksave):
#         os.mkdir(masksave)
#     if not os.path.exists(boxsave):
#         os.mkdir(boxsave)
#     if not os.path.exists(box_visual_save):
#         os.mkdir(box_visual_save)
#     if not os.path.exists(mask_isual_save):
#         os.mkdir(mask_isual_save)
#
#
#     file_name_list = os.listdir(path)
#     for i in range(len(file_name_list)):
#         i = len(file_name_list) - 1
#         if os.path.splitext(file_name_list[i])[-1] == '.jpg':
#             fileName = os.path.splitext(file_name_list[i])[0]
#             imgpath = path + fileName + '.jpg'
#             maskpath = root + '/mask/' + fileName + '.jpg'
#             boxpath = root + '/box/' + fileName + '.csv'
#             names = ['xmin', 'ymin', 'xmax', 'ymax']
#             boxes_ = read_csv(boxpath, names=names)
#             boxes_ = boxes_.to_numpy()
#
#             img_ = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
#             mask_ = cv2.imdecode(np.fromfile(maskpath, dtype=np.uint8), cv2.IMREAD_COLOR)
#             mask_ = (cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY) > 50).astype(np.uint8)
#             # cv2.imencode('.jpg', mask_)[1].tofile('C:\\Users\\Dell\\Desktop\\暂存\\mask_.jpg')
#
#             height, width = img_.shape
#             crops, boxes_sets = centercrop(img_.shape, boxes_)
#             for s in range(len(crops)):
#                 crop = crops[s]
#                 boxes = boxes_sets[s]
#                 # img = img_[crop[0]:crop[2], crop[1]:crop[3]]
#                 # mask = mask_[crop[0]:crop[2], crop[1]:crop[3]]
#                 img = img_[crop[1]:crop[3], crop[0]:crop[2]]
#                 mask = mask_[crop[1]:crop[3], crop[0]:crop[2]]
#                 img, boxes, mask = resize(img, boxes, mask)
#
#                 savename = imgname + str(i) + '_' + str(s)
#                 cv2.imencode('.jpg', mask)[1].tofile(os.path.join(masksave, '%s.jpg') % savename)
#                 cv2.imencode('.jpg', img)[1].tofile(os.path.join(imgsave, '%s.jpg') % savename)
#                 cv2.imencode('.jpg', mask*255)[1].tofile(os.path.join(mask_isual_save, '%s.jpg') % savename)
#                 img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#                 for pt in boxes:
#                     cv2.rectangle(img_rgb, (pt[0], pt[1]), (pt[2], pt[3]), COLOR1, 2)
#                 cv2.imencode('.jpg', img_rgb)[1].tofile(os.path.join(box_visual_save, '%s.jpg') % savename)
#                 with open(os.path.join(boxsave, '%s.csv') % savename, 'w', encoding='utf-8', newline="") as file:
#                     writer = csv.writer(file)
#                     writer.writerows(boxes)


if __name__ == '__main__':
    root = '/home/hryang/single/'
    saveroot = '/home/hryang/single1/'
    img_list = os.listdir(root)
    for i in range(len(img_list)):
        imgpath = root + img_list[i]
        img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img1 = (img - img.min()) / (img.max() - img.min()) * 255
        img1 = img1.astype(int)
        savename = saveroot + img_list[i]
        cv2.imencode('.jpg', img1)[1].tofile(savename)
        print(f'Fnished {i}/{len(img_list)}: {savename}!')

