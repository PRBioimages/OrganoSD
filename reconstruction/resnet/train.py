import argparse
import torch
from ctc_dataset import CtcDataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# from model.MobileNetv3small import MobileNetV3Small, MobileNetV3Small_withoutData
# from model.ResNet18_modify_block import ResNet18_block_CTC
# from model.ResNet18_modify_conv import ResNet18_conv_CTC
# from model.eulenberg import Eulenberg_ctc
# from model.vgg16_CTC import Vgg16_CTC_img
# from model_compare.ctc_resnet50 import ResNet50_CTC
# from model_compare.ctc_vgg16 import Vgg16_CTC
# from model_compare.inception import inception_v3_CTC, inception_CTC, inception_data_CTC
from learning_rate import lr_scheduler
from tqdm import tqdm
from criteria import AverageMeter, accuracy, get_accuracy
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score, confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
from criteria import get_confusion_matrix, add_confusion_matrix
import os
import torch.optim as optim
# from model.MobileNetV3Large import MobileNetV3Large, MobileNetV3LargeWithoutData
from ResNet18 import ResNet18_CTC, ResNet18_CTC_256, ResNet18_CTC_small, ResNet18_latent_CTC, ResNet18_CTC_deploy, \
    ResNet18_CTC_11, ResNet18_CTC_without_data
# from model.ResNet18_CTC_attention import ResNet18_CTC_attention
# from model.ResNet18_in_modify import ResNet18_modify_CTC
# from model.ResNet34 import Paper_ResNet34_ablate
# from model.convnext import convnext_CTC, convnext_data_CTC
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from itertools import product
import numpy as np
import random
torch.multiprocessing.set_sharing_strategy("file_system")

def train(lr, bs):
    parser = argparse.ArgumentParser(description="train CTC classification")
    parser.add_argument("--train_img_list", default="/home/xlzhu/heying/CTCs/resnet/traindata_noID.txt", help="The path of train List")
    parser.add_argument("--train_csv_path", default="/home/xlzhu/heying/CTCs/resnet/trainfeature_data.csv", help="The path of ctc_data.csv")
    parser.add_argument("--val_img_list", default="/home/xlzhu/heying/CTCs/resnet/validdata_noID.txt", help="The path of val List")
    parser.add_argument("--val_csv_path", default="/home/xlzhu/heying/CTCs/resnet/validfeature_data.csv", help="The path of ctc_data.csv")
    parser.add_argument("--lr", default=lr, type=float, help="Initial learning rate")
    parser.add_argument("--bs", default=bs, type=int, help="Batch size")
    parser.add_argument("--epoch", default=100, type=int, help="Epoch")
    parser.add_argument("--weight", default="./model_compare/weight", type=str, help="The path of weight to save")
    parser.add_argument("--run", default="./run", type=str, help="The path of log")
    parser.add_argument("--model", default="ResNet18_CTC_Data", type=str, choices=['baseline', 'MobileNetV3Large',
                                                                                  'ResNet18_CTC', 'MobileNetV3LargeWithoutData',
                                                                              'ResNet34_CTC', 'ResNet18_CTC_256'],
                        help="The model to select")

    args = parser.parse_args()
    print(args)

    # 添加权重保存位置
    save_path = args.weight + "/{}/lr_{}_bs_{}_3/".format(args.model, args.lr, args.bs)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 添加日志保存位置
    run_path = args.run + '/{}/lr_{}_bs_{}_3/'.format(args.model, args.lr, args.bs)
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    # # 固定随机数种子
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(run_path)

    # 数据增强
    transform = {
        "train": transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation(degrees=(0, 180), expand=False),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.0312,0.0074,0.0163], std=[0.0264,0.0417,0.0820])]),
        "val": transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.0312,0.0074,0.0163], std=[0.0264,0.0417,0.0820])])
    }

    mean = np.array([1.474753458e+02, 1.51033654e+02, 1.619412531e+02, 1.288255281e+01,
                     2.054398861e+03, 9.443896501e+00, 7.284966314e+00, 4.386195443e+00,
                     2.2520262e+00, 1.393370057e+00, 1.182969569e+00, 3.964444264e+00])
    std = np.array([ 6.945275358e+01, 3.68741735e+01, 7.438464332e+01, 5.795014963e+00,
                    1.369640483e+03, 2.418323966e+00, 3.361891133e+00, 1.77018288e+00,
                    5.45516886e-01, 6.03568549e-01, 5.04169621e-01, 1.424962283e+00])
 #    mean = np.array([3.03325822e+03,2.94586134e+02,1.92541247e+02,2.61710999e+00,
 # 1.44490563e+01,7.72661234e+00,9.52841650e+00,1.58208585e+02,
 # 2.44599933e+00,1.23189706e+02,1.34666465e+00,9.47649028e+00])
 #    std = np.array([5.65746276e+03,9.20059443e+03,1.11515474e+02,5.77596481e+00,
 # 1.42533765e+01,3.07084990e+00,6.49837344e+00,1.02547596e+02,
 # 1.01043341e+00,4.44763061e+01,6.55976984e-01,4.65236347e+00])

    train_dataset = CtcDataset(img_list_path=args.train_img_list, data_csv_path=args.train_csv_path, transform=transform['train'],
                               )
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, num_workers=4, shuffle=True)

    val_dataset = CtcDataset(img_list_path=args.val_img_list, data_csv_path=args.val_csv_path, transform=transform['val'],
                             fluorescence_data_mean=mean, fluorescence_data_std=std)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False)

    # 加载模型
    if args.model == "baseline":
        model = None
    elif args.model == "Paper_MobileNetV3Large_ablate":
        model = MobileNetV3Large().to(device)
    elif args.model == "MobileNetV3LargeWithoutData":
        model = MobileNetV3LargeWithoutData().to(device)
    elif args.model == "ResNet18_CTC_Concat" or \
            args.model == "ResNet18_CTC_Data" or \
            args.model == "PaperDatasetWithoutContrast" \
            or args.model == "DatasetNN"\
            or args.model == "ResNet18_CTC_datasetv3v2"\
            or args.model == "ResNet18_CTC_datasetv3v3"\
            or args.model == "ResNet18_CTC_datasetv3v4"\
            or args.model == "DatasetWithoutEnhance"\
            or args.model == "Datasetv4_clean"\
            or args.model == "NoDialated"\
            or args.model == "NoDialatedwithCD45"\
            or args.model == "NoDialated_without_cd45"\
            or args.model == "Datasetv5_NoDialated"\
            or args.model == "NoDialated_without_cd45"\
            or args.model == "Dialated_7withoutCD45"\
            or args.model == "Dialated_9withoutCD45"\
            or args.model == "Dialated_11withoutCD45"\
            or args.model == "Dialated_11withoutCD45_data"\
            or args.model == "Datasetv6_11_45_data1"\
            or args.model == "Datasetv6_1838_clean"\
            or args.model == "11_45_data_clean"\
            or args.model == "11_45_data_clean_1" or args.model == "081517193_5_5" \
            or args.model == "Paper_ResNet18_Concat" \
            or args.model == "ResNet18_latent_CTC"\
            or args.model == "withoutEnhancing" \
            or args.model == "PaperDatasetContrastNonMask":

        model = ResNet18_CTC().to(device)
    elif args.model == "ResNet34_CTC" or args.model == "Paper_ResNet34":
        model = ResNet34_CTC().to(device)
    elif args.model == "ResNet18_CTC_256" :
        model = ResNet18_CTC_256().to(device)
    elif args.model == "ResNet18_CTC_attention":
        model = ResNet18_CTC_attention().to(device)
    elif args.model == "ResNet18_CTC_small" or args.model == "ResNet18_CTC_small_08171811":
        model = ResNet18_CTC_small().to(device)
    elif args.model == "ResNet18_modify_CTC":
        model = ResNet18_modify_CTC().to(device)
    elif args.model == "ResNet18_block_CTC":
        model = ResNet18_block_CTC().to(device)
    elif args.model == "ResNet18_conv_CTC":
        model = ResNet18_conv_CTC().to(device)
    elif args.model == "convnext_CTC"\
            or args.model == "Paper_convnext_CTC":
        model = convnext_CTC().to(device)
    elif args.model == "convnext_data_CTC" or \
        args.model == "Paper_convnext_data_CTC":
        model = convnext_data_CTC().to(device)
    elif args.model == "Paper_MobileNetV3Small":
        model = MobileNetV3Small().to(device)
    elif args.model == "Vgg16_CTC_img":
        model = Vgg16_CTC_img().to(device)
    elif args.model == "ResNet18_CTC_deploy" or \
        args.model == "ResNet18_CTC_deploy_08151719":
        model = ResNet18_CTC_deploy().to(device)
    elif args.model == "Deploy_11_2":
        model = ResNet18_CTC_11().to(device)
    elif args.model == "Paper_ResNet34_ablate":
        model = Paper_ResNet34_ablate().to(device)
    elif args.model == "ResNet18_CTC_without_data":
        model = ResNet18_CTC_without_data().to(device)
    elif args.model == "Paper_mobilenetv3small":
        model = MobileNetV3Small_withoutData().to(device)
    elif args.model == "resnet50":
        model = ResNet50_CTC().to(device)
    elif args.model == "Vgg16_CTC":
        model = Vgg16_CTC().to(device)
    elif args.model == "Eulenberg_ctc":
        model = Eulenberg_ctc().to(device)
    elif args.model == "inception":
        model = inception_v3_CTC().to(device)
    elif args.model == "inception_CTC":
        model = inception_CTC().to(device)

    elif args.model == "inception_data_CTC":
        model = inception_data_CTC().to(device)


    else:
        model = None

    print(run_path)
    print(model)


    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 调整学习率
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
    # milestones = [40, 80]  # 里程碑
    # gamma = 0.1  # 衰减因子
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # 损失函数
    class_weights = torch.tensor(([1.0, 50.0]),dtype=torch.float32).to(device)
    criteria = nn.CrossEntropyLoss(weight=class_weights)

    train_loss = []
    val_loss = []
    iteration = 0

    for epoch in range(args.epoch):
        model.train()
        processBar = tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader))
        losses = AverageMeter()
        acc = [AverageMeter(), AverageMeter()]
        for step, (imgs, feature_data, labels) in processBar:
            iteration += step
            imgs = imgs.to(device)
            feature_data = feature_data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # pred = model(imgs)
            pred = model(imgs, feature_data)
            # print(pred.shape, labels.shape)
            loss = criteria(pred, labels)

            writer.add_scalar('loss', loss, iteration)
            losses.update(loss.item())
            prec1, prec5 = accuracy(pred, labels, topk=(1, 2))
            acc[0].update(prec1)
            acc[1].update(prec5)
            loss.backward()
            optimizer.step()
            processBar.set_description("[%d] Acc@1: %.2f Acc@5: %.2f Loss: %.8f" % (epoch, acc[0].avg, acc[1].avg, losses.avg))
        scheduler.step()
        train_loss.append(losses.avg)

        val_losses = AverageMeter()
        acc = [AverageMeter(), AverageMeter()]

        allLabels = list()
        allOutputs = list()
        allScores = list()

        with torch.no_grad():
            processBar = tqdm(enumerate(val_dataloader, 1), total=len(val_dataloader))
            model.eval()
            for step, (imgs, feature_data, labels) in processBar:
                imgs = imgs.to(device)
                feature_data = feature_data.to(device)
                labels = labels.to(device)
                pred = model(imgs, feature_data)
                # pred = model(imgs)
                loss = criteria(pred, labels)
                val_losses.update(loss.item())
                prec1, prec5 = accuracy(pred, labels, topk=(1, 2))
                acc[0].update(prec1)
                acc[1].update(prec5)
                processBar.set_description("[%d] Acc@1: %.2f Acc@5: %.2f Loss: %.8f" %
                                           (epoch, acc[0].avg, acc[1].avg, val_losses.avg))

                allOutputs += [torch.argmax(pred, dim=1).detach().to("cpu")]
                allScores += [torch.sigmoid(pred[:, 1]).detach().to("cpu")]
                allLabels += [labels.detach().to("cpu")]

            val_loss.append(val_losses.avg)
            # 在epoch循环结束后更新学习率
            scheduler.step()
            ACC = acc[0].avg
            # 损失可视化
            writer.add_scalars('losses', {'train_loss': losses.avg, 'val_loss': val_losses.avg}, epoch)

            allLabels = torch.cat(allLabels, dim=0).detach().numpy()
            allOutputs = torch.cat(allOutputs, dim=0).detach().numpy()
            allScores = torch.cat(allScores, dim=0).detach().numpy()

            precision1, recall1, thresholds1 = precision_recall_curve(allLabels, allScores)

            pr_auc = auc(recall1, precision1)
            roc_auc = roc_auc_score(allLabels, allScores)
            print("F1-micro = %.4f" % (f1_score(allOutputs, allLabels, average='micro')))
            print("F1-macro = %.4f" % (f1_score(allOutputs, allLabels, average='macro')))
            print("AUC = %.4f" % (roc_auc))
            print("PR AUC = %.4f" % pr_auc)
            matrix = confusion_matrix(allLabels, allOutputs)

            add_confusion_matrix(writer, matrix, num_classes=len(['noctc', 'ctc']), class_names=['noctc', 'ctc'],
                                 tag="Train Confusion Matrix", figsize=[10, 8], epoch=epoch)
            writer.add_pr_curve('pr_curve', allLabels, allScores, global_step=epoch)

            fnr = matrix[1][0] / (matrix[1][0] + matrix[1][1])
            print("FNR = %.4F" % (fnr))
            print("每个类别的精确率和召回率:\n",
                  classification_report(allLabels, allOutputs, target_names=['noctc', 'ctc']))
            result = classification_report(allLabels, allOutputs, target_names=['noctc', 'ctc'], output_dict=True)

            writer.add_scalars('Precision',
                               {'ctc': result['ctc']['precision'],
                                'non_ctc': result['noctc']['precision']}
                               , epoch)
            writer.add_scalars('Recall',
                               {'ctc': result['ctc']['recall'],
                                'non_ctc': result['noctc']['recall']
                                }, epoch)

            F1_micro = f1_score(allOutputs, allLabels, average='micro')
            F1_macro = f1_score(allOutputs, allLabels, average='macro')
            ROC_AUC = roc_auc
            FNR = matrix[1][0] / (matrix[1][0] + matrix[1][1])

            writer.add_scalars('Evaluate',
                               {'F1_micro': F1_micro,
                                'F1_macro': F1_macro,
                                'ROC_AUC': ROC_AUC,
                                'FNR': FNR
                                }, epoch)

            torch.save({
                'fluorescence_mean': mean,
                'fluorescence_std': std,
                'model_state_dict': model.state_dict()
            }, save_path + 'checkpoint1.pth'.format())
            global MaxACC
            if ACC > MaxACC:
                print("Good")
                MaxACC = ACC
                torch.save({
                    'fluorescence_mean': mean,
                    'fluorescence_std': std,
                    'model_state_dict': model.state_dict()
                }, save_path + 'best1.pth'.format())


if __name__ == '__main__':
    MaxACC = 0
    # parameter = dict(
    #
    #     lr=[0.001, 0.0001],
    #     batch_size=[32, 64]
    # )
    # parameter_values = [v for v in parameter.values()]
    # for lr, batch_size in product(*parameter_values):
    #     print(lr, batch_size)
    # train(lr=lr, bs=batch_size)
    train(lr=0.001, bs=64)


