import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
import math
import torchvision.models as torch_models
import torch.nn.init as init
# from models.Resnet34 import resnet34
# from models.resnet import ResNet9
from .Selayer import SElayer
#from .efficient import EfficientNet
from efficientnet_pytorch import EfficientNet


class Model(nn.Module):
    def __init__(self, pretrained=False, efficient_name="efficientnet-b0"):
        super(Model, self).__init__()
        # self.rgb_resnet = ResNet9(pretrained)
        # self.depth_resnet = ResNet9(pretrained)
        # self.ir_resnet = ResNet9(pretrained)
        # self.gloabl_resnet = ResNet9(num_classes=2)
        self.rgb_efficient = EfficientNet.from_name(efficient_name)
        self.depth_efficient = EfficientNet.from_name(efficient_name)
        self.ir_efficient = EfficientNet.from_name(efficient_name)
        self.gloabl_efficient = EfficientNet.from_name(efficient_name)


        if efficient_name == "efficientnet-b0":
            self.rgb_layer0 = nn.Sequential(self.rgb_efficient._conv_stem, self.rgb_efficient._bn0)
            self.rgb_layer1 = nn.Sequential(*[self.rgb_efficient._blocks[i] for i in range(6)])  # [0, 9]
            self.rgb_layer2 = nn.Sequential(*[self.rgb_efficient._blocks[i] for i in range(6, 16)])  # [10, 15]
            self.rgb_selayer = SElayer(320)


            self.depth_layer0 = nn.Sequential(self.depth_efficient._conv_stem, self.depth_efficient._bn0)
            self.depth_layer1 = nn.Sequential(*[self.depth_efficient._blocks[i] for i in range(6)])  # [0, 9]
            self.depth_layer2 = nn.Sequential(*[self.depth_efficient._blocks[i] for i in range(6, 16)])  # [10, 15]
            self.depth_selayer = SElayer(320)

            self.ir_layer0 = nn.Sequential(self.ir_efficient._conv_stem, self.ir_efficient._bn0)
            self.ir_layer1 = nn.Sequential(*[self.ir_efficient._blocks[i] for i in range(6)])  # [0, 9]
            self.ir_layer2 = nn.Sequential(*[self.ir_efficient._blocks[i] for i in range(6, 16)])  # [10, 15]
            self.ir_selayer = SElayer(320)

        if efficient_name == "efficientnet-b1":
            self.rgb_layer0 = nn.Sequential(self.rgb_efficient._conv_stem, self.rgb_efficient._bn0)
            self.rgb_layer1 = nn.Sequential(*[self.rgb_efficient._blocks[i] for i in range(10)])  # [0, 9]
            self.rgb_layer2 = nn.Sequential(*[self.rgb_efficient._blocks[i] for i in range(10, 23)])  # [10, 15]
            self.rgb_selayer = SElayer(320)


            self.depth_layer0 = nn.Sequential(self.depth_efficient._conv_stem, self.depth_efficient._bn0)
            self.depth_layer1 = nn.Sequential(*[self.depth_efficient._blocks[i] for i in range(10)])  # [0, 9]
            self.depth_layer2 = nn.Sequential(*[self.depth_efficient._blocks[i] for i in range(10, 23)])  # [10, 15]
            self.depth_selayer = SElayer(320)

            self.ir_layer0 = nn.Sequential(self.ir_efficient._conv_stem, self.ir_efficient._bn0)
            self.ir_layer1 = nn.Sequential(*[self.ir_efficient._blocks[i] for i in range(10)])  # [0, 9]
            self.ir_layer2 = nn.Sequential(*[self.ir_efficient._blocks[i] for i in range(10, 23)])  # [10, 15]
            self.ir_selayer = SElayer(320)

        self.catConv = nn.Sequential(
            nn.Conv2d(320 * 3, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512,2)
        self.fc = nn.Sequential(
            nn.Linear(256, 128, bias=False),
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

    def forward(self, rgb_img, depth_img, ir_img):
        '''
        img shape :[64,3,112,112]
        feat0 shape:[64,64,28,28]
        feat1 shape:[64,64,28,28]
        feat2 shape:[64,128,14,14]
        feat3 shape:[64,128,14,14]
        '''
        rgb_feat0 = self.rgb_layer0(rgb_img)  # [bs,64,112,112]
        rgb_feat1 = self.rgb_layer1(rgb_feat0)  # [bs,128,56,56]
        rgb_feat2 = self.rgb_layer2(rgb_feat1)  # [bs,256,28,28]
        rgb_feat3 = self.rgb_selayer(rgb_feat2)  # [bs,256,28,28]

        depth_feat0 = self.depth_layer0(depth_img)
        depth_feat1 = self.depth_layer1(depth_feat0)
        depth_feat2 = self.depth_layer2(depth_feat1)
        depth_feat3 = self.depth_selayer(depth_feat2)

        ir_feat0 = self.ir_layer0(ir_img)
        ir_feat1 = self.ir_layer1(ir_feat0)
        ir_feat2 = self.ir_layer2(ir_feat1)
        ir_feat3 = self.ir_selayer(ir_feat2)

        # hsv_feat0 = self.hsv_layer0(hsv_img)
        # hsv_feat1 = self.hsv_layer1(hsv_feat0)
        # hsv_feat2 = self.hsv_layer2(hsv_feat1)
        # hsv_feat3 = self.hsv_selayer(hsv_feat2)

        # ycb_feat0 = self.ycb_layer0(ycb_img)
        # ycb_feat1 = self.ycb_layer1(ycb_feat0)
        # ycb_feat2 = self.ycb_layer2(ycb_feat1)
        # ycb_feat3 = self.ycb_selayer(ycb_feat2)

        cat_feat = torch.cat((rgb_feat3, depth_feat3, ir_feat3), 1)  # [64,640,14,14]
        cat_feat0 = self.catConv(cat_feat)  # [64,128,14,14]
        # cat_feat1 = self.cat_layer3(cat_feat0)  # [64,256,7,7]
        cat_feat3 = self.avg_pool(cat_feat0)  # [64,512,1,1]

        cat_fc = cat_feat3.view(cat_feat3.shape[0], -1)  # [64,512]
        result = self.fc(cat_fc)  # [64,2]

        return result

# net = Model()
# x = torch.randn(1, 3, 224, 224)
# y = net(x)
# print(y.shape)