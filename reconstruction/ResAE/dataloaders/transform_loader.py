import albumentations as A
import os


def get_tfms(name):
    #获取当前脚本所在的目录，并根据 name 构建数据增强配置文件的路径
    path = os.path.dirname(os.path.realpath(__file__)) + '/../configs/augmentation/{}.yaml'.format(name)
    return A.load(path, data_format='yaml')


if __name__ == '__main__':
    get_tfms('basic_gridmask_cutout.yaml')
