import torch.nn as nn
from torchvision.models import resnet18
import torch
from torch.nn import functional as F

#只有图像输入，没有特征数据输入
class ResNet18_CTC_without_data(nn.Module):
    ''' Joint Fusion Type 1'''

    def __init__(self):
        super(ResNet18_CTC_without_data, self).__init__()
        self.feature = resnet18(pretrained=True)
        self.fe = []
        self.feature.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 8),
            nn.ReLU(True)
        )

        self.coder = nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(True),
            nn.Linear(256, 8),
            nn.ReLU(True)
        )

        self.fusion = nn.Sequential(
            nn.Linear(8, 2)
        )

    def forward(self, img):
        feature1 = self.feature(img)
        # print("feature1:", feature1.shape)
        # self.fe.append(feature1)
        # feature2 = self.coder(x)
        # # print("feature2:", feature2.shape)
        # self.fe.append(feature2)
        # feature_all = torch.add(feature1, feature2)
        # print("fusion:", feature_all.shape)
        # feature_all = torch.concat([feature1, feature2], dim=1)
        pred = self.fusion(feature1)
        self.fe.append(pred)

        return pred




if __name__ == '__main__':
    model = ResNet18_CTC_without_data()
    img = torch.randn(1,3,96,96)#随机输入张量img，3通道，96x96
    #x = torch.randn(1,12)#随机输入张量x，表示特征数据，其中包含12个特征
    #y = model(img, x)#得到输出张量y
    y = model(img)
    print(y.shape)#打印出输出张量y的形状(1, 2)，表示y是1x2的二维张量，维度1-批次大小，维度2-预测结果
