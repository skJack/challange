import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
import math
import torchvision.models as torch_models
import torch.nn.init as init
from models.Resnet34 import resnet34
from models.resnet import ResNet9
from models.Selayer import SElayer



class Model(nn.Module):
    def __init__(self, pretrained=False):
        super(Model, self).__init__()
        self.img_resnet = torch_models.resnet18(pretrained)
        self.rp_resnet = torch_models.resnet18(pretrained)
        self.sum_resnet = torch_models.resnet18(pretrained)
        self.gloabl_resnet = torch_models.resnet18(num_classes=2)
        # self.gloabl_resnet = torch_models.resnet34(num_classes=2)
        # single img resnet layer
        self.img_layer0 = nn.Sequential(
            self.img_resnet.conv1,
            self.img_resnet.bn1,
            self.img_resnet.relu,
            self.img_resnet.maxpool
        )
        self.img_layer1 = self.img_resnet.layer1
        self.img_layer2 = self.img_resnet.layer2
        #self.img_selayer = SElayer(256)
        self.img_layer3 = self.img_resnet.layer3
        self.img_layer4 = self.img_resnet.layer4
        self.img_fc = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.Sigmoid(),
            nn.Linear(128, 2),
        )
        # rankpooling resnet layer
        self.rp_layer0 = nn.Sequential(
            self.rp_resnet.conv1,
            self.rp_resnet.bn1,
            self.rp_resnet.relu,
            self.rp_resnet.maxpool
        )
        self.rp_layer1 = self.rp_resnet.layer1
        self.rp_layer2 = self.rp_resnet.layer2
        self.rp_selayer = SElayer(256)
        self.rp_layer3 = self.rp_resnet.layer3
        self.rp_layer4 = self.rp_resnet.layer4
        self.rp_fc = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.Sigmoid(),
            nn.Linear(128, 2),
        )
        # ir resnet layer
        self.sum_layer0 =  nn.Sequential(
            self.sum_resnet.conv1,
            self.sum_resnet.bn1,
            self.sum_resnet.relu,
            self.sum_resnet.maxpool
        )
        self.sum_layer1 = self.sum_resnet.layer1
        self.sum_layer2 = self.sum_resnet.layer2
        self.sum_selayer = SElayer(256)
        self.sum_layer3 = self.sum_resnet.layer3
        self.sum_layer4 = self.sum_resnet.layer4
        self.sum_fc = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.Sigmoid(),
            nn.Linear(128, 2),
        )

        self.catConv = nn.Sequential(
            nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.cat_layer3 = self.gloabl_resnet.layer3
        #self.cat_layer4 = self.gloabl_resnet.layer4

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512,2)
        self.fc = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.Sigmoid(),
            nn.Linear(128, 2),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()

    def forward(self, img, rank_pooling):
        '''
        img shape :[64,3,112,112]
        feat0 shape:[64,64,28,28]
        feat1 shape:[64,64,28,28]
        feat2 shape:[64,128,14,14]
        feat3 shape:[64,128,14,14]
        '''
        bs = img.shape[0]
        img_feat0 = self.img_layer0(img)  # [bs,64,112,112]
        img_feat1 = self.img_layer1(img_feat0)  # [bs,128,56,56]
        img_feat2 = self.img_layer2(img_feat1)  # [bs,256,28,28]
        img_feat3 = self.img_layer3(img_feat2)  # [bs,256,28,28]
        img_feat4 = self.img_layer4(img_feat3)
        img_pool = self.avg_pool(img_feat4).view(bs,-1) # [64,512,1,1]
        img_result = self.img_fc(img_pool)

        rp_feat0 = self.rp_layer0(rank_pooling)
        rp_feat1 = self.rp_layer1(rp_feat0)
        rp_feat2 = self.rp_layer2(rp_feat1)
        rp_feat3 = self.rp_layer3(rp_feat2)
        rp_feat4 = self.rp_layer4(rp_feat3)
        rp_pool = self.avg_pool(rp_feat4).view(bs,-1)
        rp_result = self.rp_fc(rp_pool)

        sum_feat1 = img_feat1+rp_feat1
        sum_feat2 = self.sum_layer2(sum_feat1)
        sum_feat3 = self.sum_layer3(sum_feat2)
        sum_feat4 = self.sum_layer4(sum_feat3)
        sum_pool = self.avg_pool(sum_feat4).view(bs,-1)
        sum_result = self.sum_fc(sum_pool)

        whole_pool = sum_pool + rp_pool + img_pool
        whole_result = self.fc(whole_pool)

        #cat_feat = torch.cat((rgb_feat3, depth_feat3, ir_feat3), 1)  # [64,640,14,14]
        #cat_feat0 = self.catConv(cat_feat)#[64,128,14,14]
        #cat_feat1 = self.cat_layer3(cat_feat0)  # [64,256,7,7]
        #cat_feat3 = self.avg_pool(cat_feat1)  # [64,512,1,1]

        #cat_fc = cat_feat3.view(cat_feat3.shape[0], -1)  # [64,512]
        #result = self.fc(cat_fc)  # [64,2]

        return whole_result,img_result,rp_result,sum_result
