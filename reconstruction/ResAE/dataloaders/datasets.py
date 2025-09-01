from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)
from batchgenerators.utilities.file_and_folder_operations import *
import cv2
import torch
import numpy as np

channel2color = {
    'DAPI': 2,
    'FITC': 1,
    'z633': 0,
    # 'Rhodamine': 0
    }


class Organoiddataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        print(cfg)
        self.imgs_list = os.listdir(self.cfg.data.img_dir)
    def __len__(self):
        return len(self.imgs_list)
    def __getitem__(self, index):
        imgName = os.path.splitext(self.imgs_list[index])[0]
        imgpath = self.cfg.data.img_dir + imgName + '.jpg'
        img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        return torch.tensor(img.copy())



#自定义数据集类 CTCDataSET
class CTCDataSET(Dataset):
    def __init__(self, df, DIR_CTC, tfms=None, transformsize=96, LabelNum=2, img_type='cell'):
        self.df = df.reset_index(drop=True)
        self.DIR_CTC = DIR_CTC
        self.LabelNum = LabelNum
        self.img_type = img_type
        self.transform = tfms
        self.transformsize = transformsize
        self.tensor_tfms = Compose([
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    #返回数据集的长度，即样本的数量
    def __len__(self):
        return len(self.df)

    #通过索引来访问数据集中的特定项
    def __getitem__(self, index):
        sample = self.df.loc[index, 'Sample']
        ID = self.df.loc[index, 'ID']
        name = sample+"rectangle"+ID+'.png'

        img_path = os.path.join(self.DIR_CTC, name)  # 获取图像路径

        if self.img_type == 'cell':
            img = self.load_crop_cell(sample, ID)
        else:
            img = self.load_crop_img_patch(sample, ID)

        #将加载的图像调整到指定大小，如果图像尺寸与期望的尺寸不匹配的话
        if not img.shape[0] == self.transformsize:
            img = cv2.resize(img, (self.transformsize, self.transformsize), interpolation=2)

        #如果提供了任何变换函数 (tfms)，它会将这些函数应用到图像数据上
        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        img = self.tensor_tfms(img)

        #提取当前索引的标签，并将其转换为独热编码
        Label = self.df.loc[index, 'Label']
        Label = self.label2onehot(Label)
        #return img_path, img, Label
        return img, Label

    #从指定目录中加载和裁剪细胞图像，接受'sample'、'ID' 和一个可选的 'mode' 参数
    def load_crop_cell(self, sample, ID, mode='rectangle'):
        '''
        :param DIR_CTC:
        :param sample:
        :param ID:
        :param mode:  'rectangle' or 'maskonly'
        :return:
        '''
        #内部辅助函数，将图像从BGR颜色空间转换为RGB颜色空间
        def load_img_patch(img_path, mode=1):
            img_patch = cv2.imread(img_path, mode)
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
            return img_patch

        #构建了要加载的图像文件的路径
        img_path = join(self.DIR_CTC, sample, mode, ID + '.png')
        img = load_img_patch(img_path)
        #返回加载和处理后的图像数据
        return img

    def label2onehot(self, Label):
        one_hot = np.zeros(self.LabelNum)
        one_hot[Label] = 1
        return one_hot

    #加载和裁剪图像块，并将多个通道的图像合并成一个多通道图像
    def load_crop_img_patch(self, sample, ID):
        '''
        :param sample:
        :param ID:
        :return: 通道图像 merge 后的图像
        '''
        def load_img_patch(img_path, mode=1):
            img_patch = cv2.imread(img_path, mode)
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
            return img_patch
        color_channels = ['z633', 'FITC', 'DAPI']
        img = []
        for channel in color_channels:
            c = f'{channel}+{ID}'
            img_path = join(self.DIR_CTC, sample, 'ResultCrop', c + '.tif')
            img_channel = load_img_patch(img_path)
            img.append(img_channel[:, :, channel2color[channel]])
        img = np.stack(img, -1)
        return img



