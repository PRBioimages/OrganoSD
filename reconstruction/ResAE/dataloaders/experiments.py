import os.path
# import sys

#from path import Path
pythonpath=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(pythonpath)
# sys.path.insert(0,pythonpath)
import torch
import torch.utils.data as data
from reconstruction.ResAE.utils import Config
from reconstruction.ResAE.configs import get_config
import pandas as pd
from reconstruction.ResAE.dataloaders.datasets import *
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from reconstruction.ResAE.dataloaders.transform_loader import get_tfms


def reset_fold(df, save_path):
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
    Samples = df.loc[:, 'Sample'].unique()
    f = np.zeros(len(df))
    for i, (train_idx, test_idx) in enumerate(kfold.split(Samples)):
        fID = Samples[test_idx]
        testid = np.where(df.Sample.isin(fID))[0]
        f[testid] = f[testid] + i
    df['fold'] = f
    df.to_csv(save_path, index=False)
    return df


def Filter_data(df):
    x = df.loc[
        (df['FITCvsz633_mean_intensity'] > 0.7) & (df['FITCvsz633_max_intensity'] > 0.45) &
        (df['FITCvsz633_75_intensity'] > 0.8), :]
    x = x.loc[(x['z633_mean_intensity'] < 90) & (x['z633_max_intensity'] < 250) & (x['z633_min_intensity'] < 50) & (
                x['z633_75_intensity'] < 95) & (x['z633_75_intensity'] < 95), :]

    x = x.loc[(x['nucleus_major_axis_length'] < 100) & (x['nucleus_sag_hemlineVperpendicular'] < 8), :]

    x['cell_majorvsminor'] = x['cell_major_axis_length'] / (x['cell_minor_axis_length'] + 1)
    x = x.loc[(x['cell_majorvsminor'] < 1.5), :]
    return x


def load_df(CTCdf_path, Normdf_path, Filter=True, radom_sample=True):
    CTCdf = pd.read_csv(CTCdf_path)
    Normdf = pd.read_csv(Normdf_path)
    if not 'fold' in CTCdf.columns:
        CTCdf = reset_fold(CTCdf, CTCdf_path)
        Normdf = reset_fold(Normdf, Normdf_path)
    CTCdf['Label'] = 1
    Normdf['Label'] = 0
    if radom_sample:
        Normdf = Normdf.sample(frac=0.25, random_state=12345)
    df = pd.concat([CTCdf, Normdf], axis=0)
    if Filter:
        df = Filter_data(df)
    return df.reset_index(drop=True)


class OrganoidTrainTestSplit:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        print(cfg)
        self.imgs_list = os.listdir(self.cfg.data.img_dir)
    def get_dataloader(self, test_only=False, train_shuffle=True, infer=False, tta=-1, tta_tfms=None):
        print('[ √ ] Using transformation: {} & {}, image size: {}'.format(
            self.cfg.transform.name, self.cfg.transform.val_name, self.cfg.transform.size
        ))
        if self.cfg.transform.name == 'None':
            train_tfms = None
        else:
            train_tfms = get_tfms(self.cfg.transform.name)
        if tta_tfms:
            val_tfms = tta_tfms
        elif self.cfg.transform.val_name == 'None':
            val_tfms = None
        else:
            val_tfms = get_tfms(self.cfg.transform.val_name)

        train_ds = Organoiddataset(self.cfg.data.img_dir)
        train_dl = DataLoader(dataset=train_ds, batch_size=self.cfg.train.batch_size,shuffle=True)

        valid_ds = Organoiddataset(self.cfg.data.img_dir)
        valid_dl = DataLoader(dataset=valid_ds, batch_size=self.cfg.eval.batch_size, drop_last=True,
                              num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)
        return train_dl, valid_dl



class RandomKTrainTestSplit:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        print(cfg)
        self.CTCdf_path = join(self.cfg.data.df_dir, 'SingleCTC_STATS.csv')
        self.Normdf_path = join(self.cfg.data.df_dir, 'SingleNorm_STATS.csv')
        #合并为一个包含训练和验证样本的整体数据集
        self.df = load_df(self.CTCdf_path, self.Normdf_path, Filter=cfg.experiment.Filter,
                          radom_sample=cfg.experiment.radom_sample)

        #将整体数据集划分为训练数据集和验证数据集
        self.train_meta, self.valid_meta = (self.df[self.df.fold != cfg.experiment.run_fold],
                                            self.df[self.df.fold == cfg.experiment.run_fold])

        # #如果配置中的 cfg.basic.debug 为 True，则会进行下采样以降低数据集的规模。
        # # 这通常用于调试目的，以加快训练和验证的速度。
        # # print(train.head())
        # if cfg.basic.debug:
        #     print('[ W ] Debug Mode!, down sample')
        #     self.train_meta = self.train_meta.sample(frac=0.005)
        #     self.valid_meta = self.valid_meta.sample(frac=0.05)

    # def save_train_valid_txt(self, save_dir):
    #     train_meta = self.train_meta[['Sample', 'ID', 'Label']].copy()
    #     valid_meta = self.valid_meta[['Sample', 'ID', 'Label']].copy()
    #     data_meta = self.df[['Sample', 'ID', 'Label']].copy()
    #
    #     train_txt_path = os.path.join(save_dir, 'train.txt')
    #     valid_txt_path = os.path.join(save_dir, 'valid.txt')
    #     data_txt_path = os.path.join(save_dir, 'data.txt')
    #
    #     # train_meta.to_csv(train_txt_path, sep=' ', index=False, header=False)
    #     # valid_meta.to_csv(valid_txt_path, sep=' ', index=False, header=False)
    #     # data_meta.to_csv(data_txt_path, sep=' ', index=False, header=False)
    #
    #     print(f"Train set paths and labels saved to {train_txt_path}")
    #     print(f"Valid set paths and labels saved to {valid_txt_path}")
    #     print(f"data set paths and labels saved to {data_txt_path}")

    def save_train_valid_txt(self, save_dir):
        train_txt_path = os.path.join(save_dir, 'train.txt')
        valid_txt_path = os.path.join(save_dir, 'valid.txt')
        data_txt_path = os.path.join(save_dir, 'data.txt')


        with open(train_txt_path, 'w') as f:
            for _, row in self.train_meta.iterrows():
                name = row['Sample'] + '/' + "rectangle" + '/' + row['ID'] + '.png'
                #img_path = os.path.join(self.DIR_CTC, name)  # 获取图像路径
                img_path = os.path.join(self.cfg.data.data_dir, name)
                f.write(f"{img_path} {row['ID']} {row['Label']}\n")

        with open(valid_txt_path, 'w') as f:
            for _, row in self.valid_meta.iterrows():
                name = row['Sample'] + '/' + "rectangle" + '/' + row['ID'] + '.png'
                img_path = os.path.join(self.cfg.data.data_dir, name)
                f.write(f"{img_path} {row['ID']} {row['Label']}\n")

        with open(data_txt_path, 'w') as f:
            for _, row in self.df.iterrows():
                name = row['Sample'] + '/' + "rectangle" + '/' + row['ID'] + '.png'
                img_path = os.path.join(self.cfg.data.data_dir, name)
                f.write(f"{img_path} {row['ID']} {row['Label']}\n")

        print(img_path)
        print(f"Train set paths and labels saved to {train_txt_path}")
        print(f"Valid set paths and labels saved to {valid_txt_path}")
        print(f"Data set paths and labels saved to {data_txt_path}")

    #创建训练和验证数据集的数据加载器 (DataLoader)
    def get_dataloader(self, test_only=False, train_shuffle=True, infer=False, tta=-1, tta_tfms=None):
        print('[ √ ] Using transformation: {} & {}, image size: {}'.format(
            self.cfg.transform.name, self.cfg.transform.val_name, self.cfg.transform.size
        ))
        if self.cfg.transform.name == 'None':
            train_tfms = None
        else:
            train_tfms = get_tfms(self.cfg.transform.name)
        if tta_tfms:
            val_tfms = tta_tfms
        elif self.cfg.transform.val_name == 'None':
            val_tfms = None
        else:
            val_tfms = get_tfms(self.cfg.transform.val_name)
        # augmentation end

        #创建训练数据集 (train_ds) 和验证数据集 (valid_ds)，使用适当的数据转换、图像大小和图像类型
        train_ds = CTCDataSET(self.train_meta, self.cfg.data.data_dir, tfms=train_tfms,
                              transformsize=self.cfg.transform.size, img_type=self.cfg.data.train_img_type)
        train_dl = DataLoader(dataset=train_ds, batch_size=self.cfg.train.batch_size,
                                  num_workers=self.cfg.transform.num_preprocessor,
                                  shuffle=train_shuffle, drop_last=True, pin_memory=True)


        valid_ds = CTCDataSET(self.valid_meta, self.cfg.data.data_dir, tfms=val_tfms,
                              transformsize=self.cfg.transform.size, img_type=self.cfg.data.train_img_type)
        valid_dl = DataLoader(dataset=valid_ds, batch_size=self.cfg.eval.batch_size, drop_last=True,
                              num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)

        return train_dl, valid_dl

# 使用示例
# cfg = get_config('config.yaml')
# splitter = RandomKTrainTestSplit(cfg)
# splitter.save_train_valid_txt(save_dir='/home/xlzhu/heying/CTCs/')



