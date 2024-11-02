import matplotlib
matplotlib.use('Agg')
# from matplotlib import pyplot as plt
#
# #import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
# import pandas as pd
# import random
# import torch
# import torchvision
# from torchvision import transforms
# from torch.utils.data import DataLoader,random_split
# from torch import nn
# import torch.nn.functional as F
# import torch.optim as optim
from reconstruction.ResAE.dataloaders import get_dataloader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
# from batchgenerators.utilities.file_and_folder_operations import *
from reconstruction.ResAE.utils import parse_args, prepare_for_result
from reconstruction.ResAE.models import get_model
from reconstruction.ResAE.losses import get_loss
# from ResAE.losses.regular import FocalLoss
from reconstruction.ResAE.optimizers import get_optimizer
from reconstruction.ResAE.scheduler import get_scheduler
from reconstruction.ResAE.base_train import base_train
from reconstruction.ResAE.models.CAE import AE
import warnings
warnings.filterwarnings("ignore")


def main():
    args, cfg = parse_args()
    result_path = prepare_for_result(cfg)
    writer = SummaryWriter(log_dir=result_path)
    cfg.dump_json(result_path / 'config.json')

    ## 加载数据
    train_dl, valid_dl = get_dataloader(cfg)(cfg).get_dataloader()
    print('[ i ] The length of train_dl is {}, valid dl is {}'.format(len(train_dl), len(valid_dl)))

    # # 打印部分样本数据
    # for batch in train_dl:
    #     data, labels = batch
    #     print(f'[ i ] Data batch shape: {data.shape}, Labels batch shape: {labels.shape}')
    #     break
    # # 在加载验证集后添加打印语句
    # for batch in valid_dl:
    #     data, labels = batch
    #     print(f'[ i ] Validation Data batch shape: {data.shape}, Labels batch shape: {labels.shape}')
    #     break

    # model = AE_residual(encoded_space_dim=cfg.model.encoded_space_dim, num_classes=2)
    # model = get_model(cfg)
    model = AE(100)
    # model = model.cuda()

    # loss_func = get_loss(cfg)
    loss_func = mse_yhr()

    optimizer = get_optimizer(model, cfg)
    print('[ i ] Model: {}, loss_func: {}, optimizer: {}'.format(cfg.model.name, cfg.loss.name, cfg.optimizer.name))

    if not cfg.scheduler.name == 'none':
        scheduler = get_scheduler(cfg, optimizer, len(train_dl))
    else:
        scheduler = None

    base_train(cfg, model, train_dl, valid_dl, loss_func, optimizer, result_path, scheduler, writer)
    #basic_train(cfg, model, train_dl, valid_dl, optimizer, result_path, scheduler, writer)


if __name__ == '__main__':
    main()
