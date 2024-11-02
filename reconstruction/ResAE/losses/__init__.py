from reconstruction.ResAE.losses.regular import ce, label_smooth_ce, label_smooth_ce_ohem, mse, mae, bce, sl1, bce_mse
from reconstruction.ResAE.losses.regular import focal_loss, bce_ohem, criterion_margin_focal_binary_cross_entropy, ce_oheb
import numpy as np
import pickle as pk
import os

#这个函数接受一个配置参数 cfg，该参数包含了损失函数的名称和参数。
#使用 globals().get(cfg.loss.name) 通过名称获取相应的损失函数，并传递配置参数 cfg.loss.param 给损失函数，然后返回该损失函数
def get_loss(cfg):
    return globals().get(cfg.loss.name)(**cfg.loss.param)

#生成类别平衡权重。它接受一个包含类别标签的 DataFrame df 和 betas 参数。
def get_class_balanced_weighted(df, betas):
    '''
    generate class balanced weight

    :param df:
    :param betas:
    :return:
    '''
    #获取 df 中每个类别（在这里表示为 grapheme_root）的样本数量统计信息，然后对这些统计信息按照类别索引进行排序。这将返回一个包含样本数量的数组。
    #利用公式计算权重
    weight = df.grapheme_root.value_counts().sort_index().values
    weight = (1 - betas[0]) / (1 - np.power(betas[0], weight))
    #将计算得到的权重保存到文件中（weight.pkl），这可以用于调试或者进一步的分析。
    with open(os.path.dirname(os.path.realpath(__file__)) + '/../debug/weight.pkl', 'wb') as fp:
        pk.dump([weight], fp)
    return weight

#计算的是对数权重
#先计算类别数量然后升序排序，然后使用了对数变换来计算权重
def get_log_weight(df):
    weight = (1 / np.log1p(df.label.value_counts().sort_index())).values
    return weight

