import argparse
import os
import torch
import numpy as np
import random
import requests as req
import json
from reconstruction.ResAE.configs import get_config, Config
from pathlib import Path
import pandas as pd

#创建结果目录，这个目录用于保存训练和验证过程中的模型、日志、图形等文件。
# 如果结果目录不存在，它会被创建，并在目录下创建子目录，如checkpoints、logs、figure
# train.log文件用于记录训练过程中的一些指标，如训练损失、验证损失、验证准确度和AUC等
def prepare_for_result(cfg: Config):
    print(cfg.train.dir)
    print(cfg.basic.id)
    print(os.path.exists(cfg.train.dir + '/' + cfg.basic.id))
    # print(cfg.train.dir)
    if not os.path.exists(cfg.train.dir):
        raise Exception('Result dir not found')
    # if os.path.exists(cfg.train.dir + '/' + cfg.basic.id):
    #     if cfg.basic.debug:
    #         print('[ X ] The output dir already exist!')
    #         output_path = Path(cfg.train.dir) / cfg.basic.id
    #         return output_path
    #     else:
    #         raise Exception('The output dir already exist')
    output_path = Path(cfg.train.dir) / cfg.basic.id
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path / 'checkpoints',exist_ok=True)
    os.makedirs(output_path / 'logs',exist_ok=True)
    os.makedirs(output_path / 'figure' / 'plotoutput', exist_ok=True)
    with open(output_path / 'train.log', 'w') as fp:
        fp.write(
            'Epochs\tlr\ttrain_loss\tvalid_loss\tvalid_accuracy\tauc\n'
        )
    return output_path


def parse_args(mode='sp'):
    '''
    Whether call this function in CLI or docker utility, we parse args here
    For inference, we use 2 worker
    GPU, run_id,

    :return:
    '''
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--debug', type=bool, default=False)
    arg('--search', type=bool, default=False)
    arg('--gpu', type=str, default='0')
    arg('-save_name', type=str, default='')
    arg('-config', type=str, default='')
    args = parser.parse_args()

    #config_filename = os.path.basename(args.config)
    # print(config_filename)
    cfg = get_config('config.yaml')

    # Initial jobs
    # # set seed, should always be done
    # torch.manual_seed(cfg.basic.seed)
    # torch.cuda.manual_seed(cfg.basic.seed)
    # torch.cuda.manual_seed_all(cfg.basic.seed)
    # np.random.seed(cfg.basic.seed)
    # random.seed(cfg.basic.seed)

    # set the gpu to use
    print('[ √ ] Using #{} GPU'.format(args.gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(','.join(args.gpu))
    # print(','.join(args.gpu))

    cfg.basic.id = args.save_name
    cfg.basic.debug = args.debug

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    return args, cfg
