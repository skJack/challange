import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import models


class MingNet(nn.Module):
    def __init__(self):
        super(MingNet, self).__init__()
        self.depth_net = MobileNetv2(LinearBottleneck, [1, 3, 1], reduction=16, num_classes=128)
        self.rgb_net = MobileNetv2(LinearBottleneck, [1, 3, 1], reduction=16, num_classes=128)
        self.ir_net = MobileNetv2(LinearBottleneck, [1, 3, 1], reduction=16, num_classes=128)
        self.fc = nn.Linear(128, 2, bias=False)

    def forward(self, rgb, depth, ir, rgb_rp, depth_rp, ir_rp):
        depth = self.depth_net(torch.cat([depth, depth_rp], 1))
        rgb = self.rgb_net(torch.cat([rgb, rgb_rp], 1))
        ir = self.ir_net(torch.cat([ir, ir_rp], 1))

        output = depth + rgb + ir
        output = self.fc(output)

        return output


class MobileNetv2(nn.Module):
    def __init__(self, block, num_blocks, reduction, num_classes):
        super(MobileNetv2, self).__init__()
        self.inplanes = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
        )
        self.layer2 = self.__make_layer(block, num_blocks[0],  64, reduction, stride=1)
        self.layer3 = self.__make_layer(block, num_blocks[1], 128, reduction, stride=2)
        self.layer4 = self.__make_layer(block, num_blocks[2], 256, reduction, stride=2)
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_layer(self, block, num_blocks, planes, reduction, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, reduction=reduction, downsample=downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, stride=1, reduction=reduction))
        
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class LinearBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride, t=6, reduction=16, downsample=None):
        """每经过一个线性瓶颈层, 分辨率变为原来一半 
        
        Arguments:
            inplanes {int} -- [输入的通道数]
            channel_expansion {int} -- [扩张系数]] (default: {6})
        """
        
        super(LinearBottleneck, self).__init__()
        self.inplanes = inplanes
        self.stride = stride
        self.downsample = downsample

        self.se = SElayer(inplanes * t, reduction=reduction)
        self.pw1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes*t, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes * t),
            nn.ReLU6(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(inplanes*t, inplanes*t, groups=inplanes*t, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(inplanes * t),
            nn.Dropout2d(),
            nn.ReLU6(inplace=True),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(inplanes*t, planes*self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion),
        )
    
    def forward(self, x):
        residual = self.pw1(x)
        residual = self.dw(residual)
        residual = self.se(residual)
        residual = self.pw2(residual)

        if self.downsample is not None:
            x = self.downsample(x)

        return residual + x

        
class SElayer(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction, inplanes, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        model_input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        
        return model_input * x