import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
import math
import torchvision.models as torch_models
import torch.nn.init as init
from models.single_sdnet import Model as sdnet


class Model(nn.Module):
    def __init__(self, pretrained=False):
        super(Model, self).__init__()
        self.rgb_sdnet = sdnet(pretrained)
        self.ir_sdnet = sdnet(pretrained)
        self.depth_sdnet = sdnet(pretrained)
        #self.gloabl_resnet = ResNet9(num_classes=2)
        # self.gloabl_resnet = torch_models.resnet34(num_classes=2)
        # single img resnet layer

        self.catConv = nn.Sequential(
            nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        #self.cat_layer3 = self.gloabl_resnet.layer3
        #self.cat_layer4 = self.gloabl_resnet.layer4

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512,2)
        self.fc = nn.Linear(3*2,2)
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

    def forward(self, rgb_img, depth_img, ir_img, rp_rgb_img,rp_depth_img,rp_ir_img):
        '''
        img shape :[64,3,112,112]
        feat0 shape:[64,64,28,28]
        feat1 shape:[64,64,28,28]
        feat2 shape:[64,128,14,14]
        feat3 shape:[64,128,14,14]
        '''
        bs = rgb_img.shape[0]
        rgb_score,_,_,_ = self.rgb_sdnet(rgb_img,rp_rgb_img)
        #rgb_score = rgb_score.view(bs, 1, rgb_score.shape[1])

        ir_score,_,_,_ = self.ir_sdnet(ir_img,rp_ir_img)
        #ir_score = ir_score.view(bs, 1, ir_score.shape[1])

        depth_score,_,_,_ = self.depth_sdnet(depth_img,rp_depth_img)
        #depth_score = depth_score.view(bs, 1, depth_score.shape[1])
        cat_feat = torch.cat((rgb_score,ir_score,depth_score),1)

        result = self.fc(cat_feat)
        #cat_feat = torch.cat((rgb_feat3, depth_feat3, ir_feat3), 1)  # [64,640,14,14]
        #cat_feat0 = self.catConv(cat_feat)#[64,128,14,14]
        #cat_feat1 = self.cat_layer3(cat_feat0)  # [64,256,7,7]
        #cat_feat3 = self.avg_pool(cat_feat1)  # [64,512,1,1]

        #cat_fc = cat_feat3.view(cat_feat3.shape[0], -1)  # [64,512]
        #result = self.fc(cat_fc)  # [64,2]

        return result,rgb_score,ir_score,depth_score
