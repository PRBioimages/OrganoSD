import torch.nn as nn
from torchvision.models import resnet18
import torch
from torch.nn import functional as F

class ResNet18_CTC(nn.Module):
    ''' Joint Fusion Type 1'''

    def __init__(self):
        super(ResNet18_CTC, self).__init__()
        self.feature = resnet18(pretrained=True)
        # 移除原有的全连接层
        self.feature.fc = nn.Identity()

        self.coder = nn.Sequential(
            nn.Linear(12, 12),  # 保持附加特征的维度不变
            nn.ReLU(True)
        )

        # 添加一个新的全连接层
        self.prediction = nn.Linear(524, 2)  # 512（图像特征维度） + 12（附加特征维度）

    def forward(self, img, x):
        feature1 = self.feature(img)
        feature2 = self.coder(x)

        # 直接拼接特征
        feature_all = torch.cat([feature1, feature2], dim=1)

        # 使用新的全连接层进行预测
        pred = self.prediction(feature_all)

        return pred
# class ResNet18_CTC(nn.Module):
#     ''' Joint Fusion Type 1'''
#
#     def __init__(self):
#         super(ResNet18_CTC, self).__init__()
#         self.feature = resnet18(pretrained=True)
#         # self.fe = []
#         self.feature.fc = nn.Sequential(
#             nn.Linear(512, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 8),
#             # nn.Dropout(1),
#             nn.ReLU(True)
#         )
#
#         self.coder = nn.Sequential(
#             nn.Linear(12, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 8),
#             nn.Dropout(0.2),
#             # nn.Dropout(1),
#             nn.ReLU(True)
#         )
#
#         self.fusion = nn.Sequential(
#             nn.Linear(8, 2)
#         )
#
#     def forward(self, img, x):
#         feature1 = self.feature(img)
#         # print("feature1:", feature1.shape)
#         # self.fe.append(feature1)
#         feature2 = self.coder(x)
#         # print("feature2:", feature2.shape)
#         # self.fe.append(feature2)
#         # feature1 = 0.5 * feature1
#         # feature2 = 0.5 * feature2
#
#         feature_all = torch.add(feature1, feature2)
#         # print("fusion:", feature_all.shape)
#         # feature_all = torch.concat([feature1, feature2], dim=1)
#         pred = self.fusion(feature_all)
#         # self.fe.append(pred)
#
#         return pred

class ResNet18_CTC_small(nn.Module):
    ''' Joint Fusion Type 1'''

    def __init__(self):
        super(ResNet18_CTC_small, self).__init__()
        self.feature = resnet18(pretrained=True)
        self.feature.fc = nn.Sequential(
            nn.Linear(512, 16),
            nn.ReLU(True)
        )

        self.coder = nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(True),
            nn.Linear(256, 16),
            nn.ReLU(True)
        )

        self.fusion = nn.Sequential(
            nn.Linear(16, 2)
        )

    def forward(self, img, x):
        feature1 = self.feature(img)

        feature2 = self.coder(x)
        feature_all = torch.add(feature1, feature2)
        pred = self.fusion(feature_all)
        return pred



class ResNet18_CTC_256(nn.Module):
    ''' Joint Fusion Type 1'''

    def __init__(self):
        super(ResNet18_CTC_256, self).__init__()
        self.feature = resnet18(pretrained=True)
        self.fe = []
        self.feature.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True)
        )

        self.coder = nn.Sequential(
            nn.Linear(18, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True)
        )

        self.fusion = nn.Sequential(
            nn.Linear(16, 2)
        )

    def forward(self, img, x):
        feature1 = self.feature(img)

        feature2 = self.coder(x)
        feature_all = torch.add(feature1, feature2)
        pred = self.fusion(feature_all)
        self.fe.append(pred)
        return pred


class ResNet18_latent_CTC(nn.Module):
    ''' Joint Fusion Type 1'''
    def __init__(self):
        super(ResNet18_latent_CTC, self).__init__()
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

    def forward(self, img, x):
        feature1 = self.feature(img)
        feature2 = self.coder(x)
        feature_all = torch.mul(feature1, feature2)
        pred = self.fusion(feature_all)

        return pred


class ResNet18_CTC_deploy(nn.Module):
    ''' Joint Fusion Type 1'''

    def __init__(self):
        super(ResNet18_CTC_deploy, self).__init__()
        self.feature = resnet18(pretrained=True)
        self.fe = []
        self.feature.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True)
        )

        self.coder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True)
        )

        self.fusion = nn.Sequential(
            nn.Linear(16, 2)
        )

    def forward(self, img, x):
        feature1 = self.feature(img)
        # print("feature1:", feature1.shape)
        # self.fe.append(feature1)
        feature2 = self.coder(x)
        # print("feature2:", feature2.shape)
        self.fe.append(feature2)
        feature_all = torch.add(feature1, feature2)
        # print("fusion:", feature_all.shape)
        # feature_all = torch.concat([feature1, feature2], dim=1)
        pred = self.fusion(feature_all)
        self.fe.append(pred)

        return pred


class ResNet18_CTC_11(nn.Module):
    ''' Joint Fusion Type 1'''

    def __init__(self):
        super(ResNet18_CTC_11, self).__init__()
        self.feature = resnet18(pretrained=True)
        self.fe = []
        self.feature.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 8),
            nn.ReLU(True)
        )
        #
        self.coder = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(True),
            nn.Linear(256, 8),
            nn.ReLU(True)
        )

        self.fusion = nn.Sequential(
            nn.Linear(8, 2)
        )

    def forward(self, img):
        feature1 = self.feature(img)
        # feature2 = self.coder(x)
        # self.fe.append(feature2)
        # feature_all = torch.add(feature1, feature2)
        pred = self.fusion(feature1)
        self.fe.append(pred)

        return pred

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

class Paper_ResNet18_Concat(nn.Module):
    ''' Joint Fusion Type 1'''

    def __init__(self):
        super(Paper_ResNet18_Concat, self).__init__()
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
            nn.Linear(16, 2)
        )

    def forward(self, img, x):
        feature1 = self.feature(img)
        # print("feature1:", feature1.shape)
        # self.fe.append(feature1)
        feature2 = self.coder(x)
        # print("feature2:", feature2.shape)
        self.fe.append(feature2)
        # feature_all = torch.add(feature1, feature2)
        # print("fusion:", feature_all.shape)
        feature_all = torch.concat([feature1, feature2], dim=1)
        pred = self.fusion(feature_all)
        self.fe.append(pred)

        return pred

class Paper_ResNet18_withoutImage(nn.Module):
    ''' Joint Fusion Type 1'''

    def __init__(self):
        super(Paper_ResNet18_withoutImage, self).__init__()
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

    def forward(self, img, x):
        # feature1 = self.feature(img)
        # print("feature1:", feature1.shape)
        # self.fe.append(feature1)
        feature2 = self.coder(x)
        # print("feature2:", feature2.shape)
        # self.fe.append(feature2)
        # feature_all = torch.add(feature1, feature2)
        # print("fusion:", feature_all.shape)
        # feature_all = torch.concat([feature1, feature2], dim=1)
        pred = self.fusion(feature2)
        self.fe.append(pred)

        return pred




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.ca1 = ChannelAttention(512)
        self.sa1 = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.ca1(out) * out
        out = self.sa1(out) * out
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out



if __name__ == '__main__':
    model = ResNet18_CTC()
    img = torch.randn(1,3,96,96)
    x = torch.randn(1,12)
    y = model(img, x)
    print(y.shape)
