import cv2
import tqdm
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from torch.autograd import Variable
from reconstruction.ResAE.models.CAE import AE, AEadd, VAE
from reconstruction.ResAE.models.AE_residual import ResAE, ResVAE
from reconstruction.ResAE.dataloaders.datasets import Organoiddataset


import argparse
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='DetecSeg Testing With Pytorch')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--testset_root', default='/home/hryang/testset/',
                    help='Dataset root directory path')
parser.add_argument('--save_folder', default='/home/hryang/Whole/result/test/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--modelname', default='ResAE',
                    help='Choose a model')
parser.add_argument('--mode', default='N',
                    help='V:VAE while N:normal')
parser.add_argument('--latent_dim', default=200, type=int,
                    help='latent dim')

parser.add_argument('--resume', default='/home/hryang/Whole/result/ResAE200_4/300.pth',    #'/home/hryang/Whole/result/ResAE/40.pth'
                    type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def getmodel(modelname, latentdim):
    if modelname in ['AE']:
        return AE(latentdim)
    elif modelname in ['AEadd']:
        return AEadd(latentdim)
    elif modelname in ['VAE']:
        return VAE(latentdim)
    elif modelname in ['ResAE']:
        return ResAE(latentdim)
    elif modelname in ['ResVAE']:
        return ResVAE(latentdim)



def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = getmodel(args.modelname, args.latent_dim)
    net = net.to(device)

    print('Resuming training, loading {}...'.format(args.resume))
    net.load_weights(args.resume)
    print('successful load weight！')

    test_ds = Organoiddataset(args.testset_root, mode='test')
    test_dl = DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)

    for i, (img, rec, maxmin) in enumerate(tqdm.tqdm(test_dl)):
        img = img.float().to(device)

        if args.mode == 'N':
            out, latent = net(Variable(img, requires_grad=True))
        else:
            out, latent, mu, logvar = net(Variable(img, requires_grad=True))

        u = latent.detach().numpy()

        for j in range(img.shape[0]):
            max = maxmin[0][j].detach().numpy()
            min = maxmin[1][j].detach().numpy()
            output = (out[j].detach().numpy() * 255).astype(int)
            image = (img[j].detach().numpy() * 255).astype(int)
            oringin = (image * (max - min) / 255).astype(int) + min
            result = (output * (max - min) / 255).astype(int) + min
            image = np.hstack((oringin, image, output, result))[0]
            cv2.imencode('.jpg', image)[1].tofile(args.save_folder + f'{j + i * args.batch_size}.jpg')
            print(f'Finished {j + i * args.batch_size}/{len(test_ds)}!')


if __name__ == '__main__':
    test()
