import os

from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm
import torch
from reconstruction.ResAE.models import get_model, load_matched_state
from reconstruction.ResAE.dataloaders import get_dataloader
from reconstruction.ResAE.dataloaders.experiments import load_df
from reconstruction.ResAE.basic_train import save_img
from torchvision.transforms import ToTensor, Normalize, Compose
from reconstruction.ResAE.configs import get_config, Config
from matplotlib.path import Path
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_train_val_df(cfg: Config):
    CTCdf_path = join(cfg.data.df_dir, 'SingleCTC_STATS.csv')
    Normdf_path = join(cfg.data.df_dir, 'SingleNorm_STATS.csv')
    df = load_df(CTCdf_path, Normdf_path, Filter=cfg.experiment.Filter,
                      radom_sample=False)
    train_meta, valid_meta = (df[df.fold != cfg.experiment.run_fold],
                                        df[df.fold == cfg.experiment.run_fold])

    return train_meta, valid_meta


class CTCDataSET(Dataset):
    def __init__(self, df, DIR_CTC, tfms=None, LabelNum=2):
        self.df = df.reset_index(drop=True)
        self.DIR_CTC = DIR_CTC
        self.LabelNum = LabelNum
        self.transform = tfms
        self.tensor_tfms = Compose([
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df.loc[index, 'Sample']
        ID = self.df.loc[index, 'ID']
        img = self.load_img_patch(sample, ID)
        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']
        img = self.tensor_tfms(img)
        Label = self.df.loc[index, 'Label']
        # Label = self.label2onehot(Label)
        return {'img': img, 'Label': Label, 'sample': sample, 'ID': ID}

    def load_img_patch(self, sample, ID, mode='rectangle'):
        '''
        :param DIR_CTC:
        :param sample:
        :param ID:
        :param mode:  'rectangle' or 'maskonly'
        :return:
        '''

        def load_img_patch(img_path, mode=1):
            img_patch = cv2.imread(img_path, mode)
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
            return img_patch

        img_path = join(self.DIR_CTC, sample, mode, ID + '.png')
        img = load_img_patch(img_path)
        return img

    def label2onehot(self, Label):
        one_hot = np.zeros(self.LabelNum)
        one_hot[Label] = 1
        return one_hot


def get_data_loader(train_meta, cfg, batchsize=10, train_tfms=None):
    train_ds = CTCDataSET(train_meta, cfg.data.data_dir, tfms=train_tfms)
    train_dl = DataLoader(dataset=train_ds, batch_size=batchsize,
                          num_workers=6,
                          shuffle=False, drop_last=False, pin_memory=True)
    return train_dl


def main():
    cfg_path = Path('/home/xlzhu/Work1_SingleCellPrediciton/Results/work14_CTC/S2ConvolutionAutoEncoder/AutoEncoder1/config.json')
    cfg = Config.load_json(cfg_path)
    epoch = 100
    batchsize = 512
    model_dir = cfg_path.parent / 'checkpoints'
    save_feature_dir = cfg_path.parent / 'figure' / 'feature_vision'
    save_reconstruction_dir = cfg_path.parent / 'figure' / 'reconstruction' / 'val'

    maybe_mkdir_p(save_feature_dir)
    maybe_mkdir_p(save_reconstruction_dir)

    CELL_Model_DIR = model_dir / f'f1_epoch-{epoch}.pth'
    inference_model = get_model(cfg).to(device)
    load_matched_state(inference_model, torch.load(CELL_Model_DIR, map_location='cpu'))
    _ = inference_model.eval()

    train_df, val_df = load_train_val_df(cfg)
    # train_dl = get_data_loader(train_df, cfg, batchsize=batchsize)
    valid_dl = get_data_loader(val_df, cfg, batchsize=batchsize)

    tq = tqdm(valid_dl)
    feature_list = [str(i) for i in range(cfg.model.param['encoded_space_dim'])]
    df_features = pd.DataFrame()

    for i, ouput_batch in enumerate(tq):
        image_batch = ouput_batch['img']
        lbl = ouput_batch['Label']
        sample_batch = ouput_batch['sample']
        ID_batch = ouput_batch['ID']

        image_batch = image_batch.cuda()
        lbl = lbl.cuda()
        with torch.no_grad():
            decoded_data = inference_model(image_batch)
            encoder_feature = inference_model.encoder(image_batch)

        # image_batch = image_batch.view(batchsize, 3, 96, 96)
        decoded_data_numpy = decoded_data.cpu().detach().numpy()
        image_batch_numpy = image_batch.cpu().numpy()

        # save_img(image_batch_numpy, decoded_data_numpy, save_reconstruction_dir / f'{i}.png')

        # lbl = lbl.view(batchsize, 1)
        lbl_batch_numpy = lbl.cpu().numpy()
        feature_batch_numpy = encoder_feature.cpu().numpy()

        batch_info = np.concatenate([np.array(sample_batch).reshape(-1, 1), np.array(ID_batch).reshape(-1, 1),
                                     lbl_batch_numpy.reshape(-1, 1), feature_batch_numpy], axis=1)

        df1 = pd.DataFrame(batch_info, columns=['sample', 'ID', 'Label'] + feature_list)
        df_features = df_features.append(df1)

    df_features.to_csv(save_feature_dir / 'val_features.csv', index=False)


def vision_umap(df_path, figsave_path, val_path=None):
    import umap.umap_ as umap
    import matplotlib.pyplot as plt
    import seaborn as sns

    LABEL_TO_ALIAS = {0: 'No-CTC', 1: 'CTC'}
    COLORS = sns.color_palette(palette='Set1')[::-1][-2:]
    Reduce_Dimension = 'UMAP'
    n_components = 2

    df_path = Path(df_path)
    dataset = df_path.stem.split('_')[0]
    df = pd.read_csv(df_path)
    if val_path is not None:
        df_val = pd.read_csv(val_path)
        df = df.append(df_val).reset_index(drop=True)
        dataset = 'all'
    df_ctc = df[df['Label'] == 1].reset_index(drop=True)
    df_norm = df[df['Label'] == 0].reset_index(drop=True)
    df_norm_sample = df_norm.sample(n=200, random_state=12345)

    df_used = pd.concat([df_ctc, df_norm_sample], axis=0).reset_index(drop=True)
    df_used[['X_coor', 'Y_coor']] = 0
    feature_list = df_used.columns[3:]
    features = df_used[feature_list].values

    if Reduce_Dimension == 'UMAP':

        # UMAP
        umap_args = dict({
            'n_neighbors': 15,
            'min_dist': 0.1,
            'n_components': n_components,
            # 'metric': 'braycurtis',
            'metric': 'euclidean',
        })
        reducer = umap.UMAP(
            n_neighbors=umap_args['n_neighbors'],
            min_dist=umap_args['min_dist'],
            n_components=umap_args['n_components'],
            metric=umap_args['metric'],
            random_state=33,
            verbose=True)
        X = reducer.fit_transform(features.tolist())
        title = f"Auto-Encoder_{umap_args['n_neighbors']}_{umap_args['min_dist']}_{umap_args['metric']}"

        fig, ax = plt.subplots(figsize=(16, 10))
        for i in range(2):
            label = i
            idx = np.where(df_used['Label'] == label)[0]
            x = X[idx, 0]
            y = X[idx, 1]
            df_used.loc[idx, 'X_coor'] = x
            df_used.loc[idx, 'Y_coor'] = y
            # print(label, sub_df['Label'][idx])
            plt.scatter(x, y, color=COLORS[i], label=LABEL_TO_ALIAS[i], s=16)
        ax.set_xlabel(f"{Reduce_Dimension}1", fontsize=35)
        ax.set_ylabel(f"{Reduce_Dimension}2", fontsize=35)
        ax.tick_params(axis="both", which="major", width=3, length=10)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.28, 1.01), ncol=1, markerscale=8)
        plt.title(title, fontsize=24, y=1.05)
        plt.savefig(f"{figsave_path}/{title}_{dataset}.png", dpi=150)
        df_used.to_csv(f"{figsave_path}/{dataset}.csv", index=False)


def vision_umap_neighbour(df_path, figsave_path, neighbour_information, val_path):
    import umap.umap_ as umap
    import matplotlib.pyplot as plt
    import seaborn as sns

    LABEL_TO_ALIAS = {0: 'No-CTC', 1: 'CTC'}
    COLORS = sns.color_palette(palette='Set1')[::-1][-2:]
    Reduce_Dimension = 'UMAP'
    n_components = 2

    df_path = Path(df_path)
    dataset = df_path.stem.split('_')[0]
    df = pd.read_csv(df_path)
    if val_path is not None:
        df_val = pd.read_csv(val_path)
        df = df.append(df_val).reset_index(drop=True)
        dataset = 'all'


    print(val_path)
    neighbour_information = neighbour_information.set_index('ID')
    neighbour_information = neighbour_information.reindex(index=df['ID'])
    neighbour_information = neighbour_information.reset_index()
    neighbour_information.iloc[:, 2:] = (neighbour_information.iloc[:, 2:] - neighbour_information.iloc[:, 2:].min()) / (neighbour_information.iloc[:, 2:].max() - neighbour_information.iloc[:, 2:].min())

    df = pd.concat([df, neighbour_information.iloc[:, 2:]], 1)

    df_ctc = df[df['Label'] == 1].reset_index(drop=True)
    df_norm = df[df['Label'] == 0].reset_index(drop=True)
    df_norm_sample = df_norm.sample(n=200, random_state=12345)

    # df_used = pd.concat([df_ctc, df_norm_sample], axis=0).reset_index(drop=True)
    # df_used[['X_coor', 'Y_coor']] = 0
    the_path = '/home/xlzhu/heying/CTCs/Result/S1analysis/figure/df_used.csv'
    # df_used.to_csv(the_path, index=False)
    df_used = pd.read_csv(the_path)


    feature_list = df_used.columns[3:]
    features = df_used[feature_list].values


    has_nan = np.isnan(features).any()
    print("是否存在 NaN 值:", has_nan)
    has_inf = np.isinf(features).any()
    print("是否存在无穷大值:", has_inf)
    total_samples = features.shape[0]
    nan_count = np.isnan(features).sum()

    print("总样本数:", total_samples)
    print("NaN 值数量:", nan_count)

    if Reduce_Dimension == 'UMAP':

        # UMAP
        umap_args = dict({
            'n_neighbors': 15,
            'min_dist': 0.1,
            'n_components': n_components,
            # 'metric': 'braycurtis',
            'metric': 'euclidean',
        })
        reducer = umap.UMAP(
            n_neighbors=umap_args['n_neighbors'],
            min_dist=umap_args['min_dist'],
            n_components=umap_args['n_components'],
            metric=umap_args['metric'],
            random_state=33,
            verbose=True)
        X = reducer.fit_transform(features.tolist())

        title = f"Auto-Encoder_{umap_args['n_neighbors']}_{umap_args['min_dist']}_{umap_args['metric']}"

        fig, ax = plt.subplots(figsize=(16, 10))
        for i in range(2):
            label = i
            idx = np.where(df_used['Label'] == label)[0]
            x = X[idx, 0]
            y = X[idx, 1]
            df_used.loc[idx, 'X_coor'] = x
            df_used.loc[idx, 'Y_coor'] = y
            # print(label, sub_df['Label'][idx])
            plt.scatter(x, y, color=COLORS[i], label=LABEL_TO_ALIAS[i], s=16)
        ax.set_xlabel(f"{Reduce_Dimension}1", fontsize=35)
        ax.set_ylabel(f"{Reduce_Dimension}2", fontsize=35)
        ax.tick_params(axis="both", which="major", width=3, length=10)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.28, 1.01), ncol=1, markerscale=8)
        plt.title(title, fontsize=24, y=1.05)
        plt.savefig(f"{figsave_path}/{title}_{dataset}_neighbour.png", dpi=150)
        df_used.to_csv(f"{figsave_path}/{dataset}_neighbour.csv", index=False)


def plot_list_cell(ID_list, figsave_path):
    DIR_CTC = '/home/Datasets/CTC/ctcSample'
    channels = ['Merge', 'DAPI', 'FITC', 'z633']

    fig, axes = plt.subplots(nrows=len(ID_list), ncols=4, sharex=True, sharey=True, figsize=(3, 40))

    for i in range(len(ID_list)):
        sample = ID_list[i][0]
        ID = ID_list[i][1]
        for j, channel in enumerate(channels):
            img_path = join(DIR_CTC, sample, 'ResultCrop', f'{channel}+{ID}' + '.tif')
            img_patch = cv2.imread(img_path, 1)
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(img_patch)
            axes[i, j].xaxis.set_ticks([])
            axes[i, j].yaxis.set_ticks([])

    for ax, col in zip(axes[0], channels):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], range(len(ID_list))):
        ax.set_ylabel(row, rotation=90)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.savefig(f"{figsave_path}/Normal_imgs.png", dpi=300)
    plt.clf()
    plt.close()


def vision_umap_img(df_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    LABEL_TO_ALIAS = {0: 'No-CTC', 1: 'CTC'}
    COLORS = sns.color_palette(palette='Set1')[::-1][-2:]
    Reduce_Dimension = 'UMAP'

    df_used = pd.read_csv(df_path)
    df_path = Path(df_path)
    dataset = df_path.stem
    X = df_used[['X_coor', 'Y_coor']].values
    fig, ax = plt.subplots(figsize=(16, 10))
    for i in range(2):
        label = i
        idx = np.where(df_used['Label'] == label)[0]
        x = X[idx, 0]
        y = X[idx, 1]
        # print(label, sub_df['Label'][idx])
        ax.scatter(x, y, color=COLORS[i], label=LABEL_TO_ALIAS[i], s=16)

    df_ctc = df_used[df_used['Label'] == 1].reset_index(drop=True)
    X_ctc = df_ctc[['sample', 'ID', 'X_coor', 'Y_coor']].values
    Abnorm_X = X_ctc[np.where(X_ctc[:, 2] > 6.5)[0], :]
    Abnorm_X = sorted(Abnorm_X, key=lambda x: x[3], reverse=True)

    norm_X = X_ctc[np.where(X_ctc[:, 2] <= 13)[0], :]
    norm_X = sorted(norm_X, key=lambda x: x[3], reverse=True)


    for idx, coor in enumerate(Abnorm_X):
        ax.text(coor[2], coor[3], str(idx), ha='center', va='bottom', fontsize=12)
    ax.set_xlabel(f"{Reduce_Dimension}1", fontsize=35)
    ax.set_ylabel(f"{Reduce_Dimension}2", fontsize=35)
    ax.tick_params(axis="both", which="major", width=3, length=10)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.28, 1.01), ncol=1, markerscale=8)
    plt.savefig(f"{figsave_path}/{dataset}.png", dpi=150)
    plt.clf()
    plt.close()
    abnorm = pd.DataFrame(columns=['sample', 'ID', 'X_coor', 'Y_coor'], data=Abnorm_X)
    abnorm.to_csv(join(figsave_path, 'Abnorm_X_ctc_neighbour.csv'), index=False)
    plot_list_cell(Abnorm_X, figsave_path)


def vision_umap_img_neighbour(df_path, neighbour_information):
    import matplotlib.pyplot as plt
    import seaborn as sns

    LABEL_TO_ALIAS = {0: 'No-CTC', 1: 'CTC'}
    COLORS = sns.color_palette(palette='Set1')[::-1][-2:]
    Reduce_Dimension = 'UMAP'

    df_used = pd.read_csv(df_path)
    neighbour_information = neighbour_information.set_index('ID')
    neighbour_information = neighbour_information.reindex(index=df_used['ID'])
    neighbour_information = neighbour_information.reset_index()
    df_path = Path(df_path)
    dataset = df_path.stem
    X = df_used[['X_coor', 'Y_coor']].values
    metric = (neighbour_information.loc[:, f'target_z633_50_intensity'] -
                                            neighbour_information.loc[:, f'background_z633_50_intensity']) / (neighbour_information.loc[:, f'cells_z633_50_intensity'] + 0.00001)
    # metric = neighbour_information.loc[:, f'target_z633_50_intensity'] / (neighbour_information.loc[:, f'cells_z633_50_intensity'] + 0.00001)
    metric = metric.values
    # fig, ax = plt.subplots(figsize=(16, 10))
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(2):
        label = i
        idx = np.where(df_used['Label'] == label)[0]
        x = X[idx, 0]
        y = X[idx, 1]
        z = metric[idx]
        # print(label, sub_df['Label'][idx])
        ax.scatter(x, y, z, color=COLORS[i], label=LABEL_TO_ALIAS[i], s=16)

    # df_ctc = df_used[df_used['Label'] == 1].reset_index(drop=True)
    # X_ctc = df_ctc[['sample', 'ID', 'X_coor', 'Y_coor']].values
    # Abnorm_X = X_ctc[np.where(X_ctc[:, 2] > 13)[0], :]
    # Abnorm_X = sorted(Abnorm_X, key=lambda x: x[3], reverse=True)
    #
    # norm_X = X_ctc[np.where(X_ctc[:, 2] <= 13)[0], :]
    # norm_X = sorted(norm_X, key=lambda x: x[3], reverse=True)
    #
    #
    # for idx, coor in enumerate(norm_X):
    #     ax.text(coor[2], coor[3], str(idx), ha='center', va='bottom', fontsize=12)
    # ax.set_xlabel(f"{Reduce_Dimension}1", fontsize=35)
    # ax.set_ylabel(f"{Reduce_Dimension}2", fontsize=35)
    # ax.tick_params(axis="both", which="major", width=3, length=10)
    # ax.tick_params(axis="x", labelsize=20)
    # ax.tick_params(axis="y", labelsize=20)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.28, 1.01), ncol=1, markerscale=8)
    plt.savefig(f"{figsave_path}/{dataset}_3D.png", dpi=150)
    plt.clf()
    plt.close()
    # abnorm = pd.DataFrame(columns=['sample', 'ID', 'X_coor', 'Y_coor'], data=norm_X)
    # abnorm.to_csv(join(figsave_path, 'normal_ctc.csv'), index=False)
    # plot_list_cell(norm_X, figsave_path)


if __name__ == '__main__':
    df_dir = r'/home/xlzhu/heying/CTCs/Result/S1analysis/figure/'
    #df_dir = r'G:\AnfangBiology\Results\S2ConvolutionAutoEncoder\AutoEncoder1\figure\feature_vision'
    # df_dir = r'G:\AnfangBiology\Results\S2ConvolutionAutoEncoder\AutoEncoder2-merge\figure\feature_vision'
    # df_dir = r'G:\AnfangBiology\Results\S2ConvolutionAutoEncoder\AE_test9_imagefield\figure\feature_vision'
    df_train = join(df_dir, 'train_features.csv')
    df_val = join(df_dir, 'valid_features.csv')

    figsave_path = join(df_dir, 'UMAP')
    #os.makedirs(figsave_path, exist_ok=True)
    #vision_umap(df_train, figsave_path, val_path=df_val)
    #
    #vision_umap_img(join(df_dir, 'UMAP', 'all.csv'))

    # ###### 添加环境细胞亮度信息  #######
    information_dir = '/home/xlzhu/heying/CTCs/Result/S1analysis/s5_CalculateEnv/'
    neighbour_CTC = pd.read_csv(join(information_dir, 'SingleCTC_STATS-Neighbour.csv'))
    neighbour_Norm = pd.read_csv(join(information_dir, 'SingleNorm_STATS-Neighbour.csv'))
    neighbour_information = pd.concat([neighbour_CTC, neighbour_Norm]).reset_index(drop=True)
    #vision_umap_neighbour(join(df_dir, 'UMAP', 'all.csv'), figsave_path, neighbour_information, val_path=df_val)
    #vision_umap_img(join(df_dir, 'UMAP', 'all_neighbour.csv'))
    vision_umap_img_neighbour(join(df_dir, 'UMAP', 'all.csv'), neighbour_information)



