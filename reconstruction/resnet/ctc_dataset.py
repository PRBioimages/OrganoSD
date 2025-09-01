import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import cv2
import re


#从给定的CSV文件中加载图像和相关的特征数据
class CtcDataset(Dataset):
    #img_list_path：包含图像路径和标签的文本文件路径;data_csv_path：包含额外特征数据的 CSV 文件路径;
    # fluorescence_data_mean 和 fluorescence_data_std：用于归一化特征数据的均值和标准差
    def __init__(self, img_list_path, data_csv_path, fluorescence_data_mean=None, fluorescence_data_std=None, transform=None):
        super().__init__()
        #从提供的 img_list_path 文件中读取图像路径和标签。
        #解析图像文件名，提取样本名称、细胞编号、场景和位置等信息。
        # 根据解析的信息从CSV文件中读取额外的特征数据。
        # 如果提供了用于特征数据归一化的均值和标准差，则对特征数据进行归一化。
        # 转换后的图像和相关的特征数据存储在img_data_label_list中。
        self.img_label_list = list()
        with open(img_list_path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(" ")  # TODO
                self.img_label_list.append(line)

        self.img_data_label_list = []

        # 解析pandas数据
        self.data = pd.read_csv(data_csv_path)
        # 提取出每个数据集的数据
        for img_path_label in self.img_label_list:
            (img_path, label) = img_path_label
            sample_name = img_path.split("/")[-3]
            # str_format = "(.*)_(\d+)_(\d+)_(\d+)_(\d+).png"
            # sample_name = result.group(1)
            # cell_number = int(result.group(2))
            # field = int(result.group(3))
            # x_location = int(result.group(4))
            # y_location = int(result.group(5))
            sample = img_path.split("/")[-1]
            str_format = "(\d+)_(\d+)_(\d+)_(\d+).png"
            result = re.match(str_format, sample)
            cell_number = int(result.group(1))
            field = int(result.group(2))
            x_location = int(result.group(3))
            y_location = int(result.group(4))


            sample_data = self.data[(self.data['Sample'] == sample_name)
                                    & (self.data['2-imagefield'] == field)
                                    & (self.data['3-x'] == x_location)
                                    & (self.data['4-y'] == y_location)]

            # sample_data = self.data[(self.data['cell_number'] == cell_number)
            #                         & (self.data['field'] == field)
            #                         & (self.data['x_location'] == x_location)
            #                         & (self.data['y_location'] == y_location)]

            if len(sample_data) == 2:
                print("出错")
            feature_data = np.array(sample_data[[ '5-dapi_area', '8-dapi_mean','9-ck_area', '12-ck_mean',
                                                  '13-ck_total', '15-cd45_mean', '22-dapi_fb_mean','23-ck_fb_mean',
                                                  '24-cd45_fb_mean', '26-ck_vs_cd45', '28-ck_dapi_Area', '30-ck_impurity_fb']],
                                     dtype=np.float32)
            # feature_data = np.array(sample_data[['ck_total', 'ck_fb_mean', 'ck_area', 'ck_vs_cd45',
            #                                      'ck_mean', 'cd45_mean', 'dapi_fb_mean', 'dapi_area',
            #                                      'cd45_fb_mean', 'dapi_mean', 'ck_dapi_Area', 'ck_mean_avg']],
            #                         dtype=np.float32)
            # feature_data = np.array(sample_data[['ck_total', 'ck_fb_mean', 'ck_area', 'ck_vs_cd45',
            #                                      'ck_mean', 'cd45_mean', 'dapi_fb_mean', 'cd45_fb_mean',
            #                                      'dapi_mean', 'ck_mean_avg']],
            #                         dtype=np.float32)
            self.img_data_label_list.append({"img_path": img_path, "feature_data": feature_data, "label": label})
        if not fluorescence_data_mean is None and not fluorescence_data_std is None:

            self.mean = fluorescence_data_mean
            self.std = fluorescence_data_std

            for basket in self.img_data_label_list:
                basket['feature_data'] = (basket['feature_data'] - self.mean) / self.std

        self.transform = transform

    #返回数据集中图像的数量
    def __len__(self):
        return len(self.img_label_list)

    # 返回数据集中给定索引的样本。
    # 使用PIL库打开图像。
    # 提取与图像相关的标签和特征数据。
    # 如果指定了转换，就应用这些转换。
    # 以张量形式返回图像、特征数据和标签。
    def __getitem__(self, item):
        img_data_label_dict = self.img_data_label_list[item]
        # img = cv2.imread(img_data_label_dict['img_path'],1)
        img = Image.open(img_data_label_dict['img_path']).convert('RGB')
        label = int(img_data_label_dict['label'])
        feature_data = img_data_label_dict['feature_data']

        if self.transform is not None:
            img = self.transform(img)

        feature_data = torch.tensor(feature_data, dtype=torch.float32).reshape(-1)
        label = torch.tensor(label)
        return img, feature_data, label


# 导入额外的库，如 torchvision.transforms 和 matplotlib.pyplot。
# 使用 transforms.Compose 定义一组图像转换。
# 创建一个具有指定路径和转换的 CtcDataset 类的实例。
# 使用 DataLoader 创建一个数据加载器。
# 遍历数据加载器，打印第一批次图像、特征数据和标签的形状。同时，使用 Matplotlib 展示批次中的第二张图像。
if __name__ == '__main__':
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(degrees=(0, 180), expand=False),
                                    transforms.ToTensor()])

    img_list_path = "/home/xlzhu/heying/CTCs/resnet/data_noID.txt"
    data_csv_path = "/home/xlzhu/heying/CTCs/feature_data.csv"
    # img_list_path = "/mnt/d/Data/Pycharm/CTC/Dataset_paper/08151719/test3.list"
    # data_csv_path = "/mnt/d/Data/Pycharm/CTC/Dataset_paper/08151719/test3.csv"
    ctc_data = CtcDataset(img_list_path=img_list_path,
                          data_csv_path=data_csv_path, transform=transform)

    ctc_dataloader = DataLoader(ctc_data, batch_size=32, num_workers=0, shuffle=True)
    image_tran = transforms.ToPILImage()
    for _, (img, feature_data, label) in enumerate(ctc_dataloader):
        print(img.shape, feature_data.shape, label.shape)
        print(label)
        plt.figure()
        plt.imshow(image_tran(img[1]))
        plt.show()
        break
