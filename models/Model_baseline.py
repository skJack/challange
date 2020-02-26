import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
import math
import torchvision.models as torch_models
import torch.nn.init as init
from models.Resnet34 import resnet34
from models.Selayer import SElayer
class Model(nn.Module):
    def __init__(self,pretrained = False):
        super(Model,self).__init__()
        #define six CNN
        self.rgb_resnet = torch_models.resnet34(pretrained)
        self.depth_resnet = torch_models.resnet34(pretrained)
        self.ir_resnet = torch_models.resnet34(pretrained)
        self.hsv_resnet = torch_models.resnet34(pretrained)
        self.ycb_resnet = torch_models.resnet34(pretrained)
        self.gloabl_resnet = resnet34(num_classes=2)
        #self.gloabl_resnet = torch_models.resnet34(num_classes=2)
        #rgb resnet layer
        self.rgb_layer0 = nn.Sequential(
            self.rgb_resnet.conv1,
            self.rgb_resnet.bn1,
            self.rgb_resnet.relu,
            self.rgb_resnet.maxpool
        )

        self.rgb_layer1 = self.rgb_resnet.layer1
        self.rgb_layer2 = self.rgb_resnet.layer2
        self.rgb_selayer = SElayer(128)
        #depth resnet layer
        self.depth_layer0 = nn.Sequential(
            self.depth_resnet.conv1,
            self.depth_resnet.bn1,
            self.depth_resnet.relu,
            self.depth_resnet.maxpool
        )
        self.depth_layer1 = self.depth_resnet.layer1
        self.depth_layer2 = self.depth_resnet.layer2
        self.depth_selayer = SElayer(128)
        #ir resnet layer
        self.ir_layer0 = nn.Sequential(
            self.ir_resnet.conv1,
            self.ir_resnet.bn1,
            self.ir_resnet.relu,
            self.ir_resnet.maxpool
        )
        self.ir_layer1 = self.ir_resnet.layer1
        self.ir_layer2 = self.ir_resnet.layer2
        self.ir_selayer = SElayer(128)
        #hsv resnet layer
        self.hsv_layer0 = nn.Sequential(
            self.hsv_resnet.conv1,
            self.hsv_resnet.bn1,
            self.hsv_resnet.relu,
            self.hsv_resnet.maxpool
        )
        self.hsv_layer1 = self.hsv_resnet.layer1
        self.hsv_layer2 = self.hsv_resnet.layer2
        self.hsv_selayer = SElayer(128)
        #ycb resnet layer
        self.ycb_layer0 = nn.Sequential(
            self.ycb_resnet.conv1,
            self.ycb_resnet.bn1,
            self.ycb_resnet.relu,
            self.ycb_resnet.maxpool
        )
        self.ycb_layer1 = self.ycb_resnet.layer1
        self.ycb_layer2 = self.ycb_resnet.layer2
        self.ycb_selayer = SElayer(128)

        self.catConv = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.cat_layer3 = self.gloabl_resnet.layer3
        self.cat_layer4 = self.gloabl_resnet.layer4

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512,2)
        self.fc = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.Sigmoid(),
            nn.Linear(128,2),
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
        rgb_feat0 = self.rgb_layer0(rgb_img)#[64,64,28,28]
        rgb_feat1 = self.rgb_layer1(rgb_feat0)#[64,64,28,28]
        rgb_feat2 = self.rgb_layer2(rgb_feat1)#[64,128,14,14]
        rgb_feat3 = self.rgb_selayer(rgb_feat2)#[64,128,14,14]
        
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
        
        cat_feat = torch.cat((rgb_feat3,depth_feat3,ir_feat3),1) #[64,640,14,14]
        #cat_feat0 = self.catConv(cat_feat)#[64,128,14,14]
        cat_feat1 = self.cat_layer3(cat_feat)#[64,256,7,7]
        cat_feat2 = self.cat_layer4(cat_feat1)#[64,512,4,4]
        cat_feat3 = self.avg_pool(cat_feat2)#[64,512,1,1]

        cat_fc = cat_feat3.view(cat_feat3.shape[0],-1)#[64,512]
        result = self.fc(cat_fc)#[64,2]

        return result
