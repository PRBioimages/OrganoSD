from reconstruction.ResAE.models.CAE import AutoEncoder
from reconstruction.ResAE.models.AE_residual import AE_residual
from reconstruction.ResAE.configs import Config
import torch

#据配置参数 cfg 中的模型名称（cfg.model.name）来选择初始化不同的深度学习模型
def get_model(cfg: Config):
    if cfg.model.name in ['AutoEncoder']:
        print(f'[ ! ] Init a AutoEncoder.')
        return AutoEncoder(encoded_space_dim=cfg.model.param.get('encoded_space_dim', 0))
    elif cfg.model.name in ['AE_residual']:
        print(f'[ ! ] Init a AE_residual.')
        return AE_residual(encoded_space_dim=cfg.model.param.get('encoded_space_dim', 0))

#用于加载预训练模型的权重。它接受两个参数，
# model 是你要加载权重的目标模型，state_dict 是包含预训练权重的字典。
# 函数会将预训练权重逐层匹配到目标模型中。如果某个权重在目标模型中找不到对应层，它将跳过。
# 在最后，它会输出有多少层的权重被成功加载，以及有多少层的权重未能匹配成功。
def load_matched_state(model, state_dict):
    model_dict = model.state_dict()
    not_loaded = []
    for k, v in state_dict.items():
        if k in model_dict.keys():
            if not v.shape == model_dict[k].shape:
                print('Error Shape: {}, skip!'.format(k))
                continue
            model_dict.update({k: v})
        else:
            # print('not matched: {}'.format(k))
            not_loaded.append(k)
    if len(not_loaded) == 0:
        print('[ √ ] All layers are loaded')
    else:
        print('[ ! ] {} layer are not loaded'.format(len(not_loaded)))
    model.load_state_dict(model_dict).cuda()
   # model = model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #model.to(device)

