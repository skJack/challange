import numpy as np
import os
import cv2
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import pdb

ALL = 'all'
RGB = 'rgb'
DEPTH = 'depth'
NIR = 'nir'
# root_path = '/media/sdc/datasets/Face-anti-spoofing/CASIA-CeFA/phase1'
# txt_path = '4@1_train.txt'
# raw_list = []
# # with open(os.path.join(root_path,txt_path),'r') as f:
# #     raw_list = f.read().splitlines()
# # label_list = [x.split(" ")[0].replace("profile","ir") for x in raw_list]
# # print(label_list)

class CISIA_CeFA(Dataset):
    def __init__(self,root_path = '/media/sdc/datasets/Face-anti-spoofing/CASIA-CeFA/phase1',txt_path = '',mode = 'train',transform_depth=None,transform_ir=None):
        self.mode = mode
        self.root_path = root_path
        self.txt_path = txt_path
        self.transform_depth = transform_depth
        self.transform_ir = transform_ir
        
        try:
            with open(os.path.join(root_path,txt_path),'r') as f:
                self.raw_list = f.read().splitlines()
            self.label_list = [x.split(" ")[1] for x in self.raw_list]
            self.rgb_list = [x.split(" ")[0] for x in self.raw_list]
            self.depth_list = [x.split(" ")[0].replace("profile","depth") for x in self.raw_list]
            self.ir_list = [x.split(" ")[0].replace("profile","ir") for x in self.raw_list]
        except:
            print("can not open files,may be filelist is not exist")
            exit()
    def __getitem__(self, index):
        rgb_img = Image.open(os.path.join(self.root_path,self.rgb_list[index])).convert('RGB')
        depth_img = Image.open(os.path.join(self.root_path,self.depth_list[index])).convert('RGB')
        ir_img = Image.open(os.path.join(self.root_path,self.ir_list[index])).convert('RGB')

        hsv_img = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2HSV)
        hsv_img = Image.fromarray(hsv_img)
        YCbCr_img = rgb_img.convert('YCbCr')
        if self.transform_ir:
            rgb_img = self.transform_ir(rgb_img)
            depth_img = self.transform_depth(depth_img)
            ir_img = self.transform_ir(ir_img)
            hsv_img = self.transform_ir(hsv_img)
            YCbCr_img = self.transform_ir(YCbCr_img)


        label = int(self.label_list[index])
        label = np.array(label)
        return rgb_img,depth_img,ir_img,hsv_img,YCbCr_img,label,self.rgb_list[index]
        # if self.mode == 'test':
        #     return rgb_img,depth_img,ir_img,label
    def __len__(self):
        return len(self.label_list)
        
class CISIA_CeFA_test(Dataset):
    def __init__(self,root_path = 'media/sdc/datasets/Face-anti-spoofing/CASIA-CeFA/phase1',txt_path = '',sample = '0020',transform_depth=None,transform_ir=None):
        self.root_path = root_path
        self.txt_path = txt_path
        self.transform_depth = transform_depth
        self.transform_ir = transform_ir
        self.sample = sample

        try:
            with open(os.path.join(root_path,txt_path),'r') as f:
                self.raw_list = f.read().splitlines()
            self.dev_list = [x.split(" ")[0] for x in self.raw_list]
            self.label_list = [x.split(" ")[1] for x in self.raw_list]

        except:
            print("can not open files,may be filelist is not exist")
            exit()
    def __getitem__(self, index):
        rgb_img = Image.open(os.path.join(self.root_path,self.dev_list[index]+'/profile/'+self.sample+'.jpg')).convert('RGB')
        depth_img = Image.open(os.path.join(self.root_path,self.dev_list[index]+'/depth/'+self.sample+'.jpg')).convert('RGB')
        ir_img = Image.open(os.path.join(self.root_path,self.dev_list[index]+'/ir/'+self.sample+'.jpg')).convert('RGB')
        hsv_img_cv = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2HSV)
        hsv_img = Image.fromarray(hsv_img_cv)
        YCbCr_img = rgb_img.convert('YCbCr')
        if self.transform_ir:
            rgb_img = self.transform_ir(rgb_img)
            depth_img = self.transform_depth(depth_img)
            ir_img = self.transform_ir(ir_img)
            hsv_img = self.transform_ir(hsv_img)
            YCbCr_img = self.transform_ir(YCbCr_img)

        label = int(self.label_list[index])
        label = np.array(label)
        return rgb_img,depth_img,ir_img,hsv_img,YCbCr_img,label,self.dev_list[index]
    def __len__(self):
        return len(self.dev_list)

def load_cisia_csfa(root = '/media/sdc/datasets/Face-anti-spoofing/CASIA-CeFA/phase1',protoal = "4@1",img_size = 112,train_batch = 64,test_batch = 64):
    if test_batch is None:
        test_batch = train_batch
    if protoal=="4@1":

        depth_norm = transforms.Normalize(mean=[0.70745318, 0.70745318, 0.70745318],
                                                std=[0.26528493, 0.26528493, 0.26528493])
        ir_norm = transforms.Normalize(mean=[0.22784027, 0.22784027, 0.22784027],
                                            std=[0.10182471, 0.10182471, 0.10182471])
    if protoal=="4@2":
        depth_norm = transforms.Normalize(mean=[0.67922, 0.67922, 0.67922],
                                                std=[0.2890028, 0.2890028, 0.2890028])
        ir_norm = transforms.Normalize(mean=[0.34846747, 0.34846747, 0.34846747],
                                            std=[0.17954783, 0.17954783, 0.17954783])
    if protoal=="4@3":
    
        depth_norm = transforms.Normalize(mean=[0.7110071, 0.7110071, 0.7110071],
                                                    std=[0.28458922, 0.28458922, 0.28458922])
        ir_norm = transforms.Normalize(mean=[0.31522677, 0.31522677, 0.31522677],
                                                std=[0.16015441, 0.16015441, 0.16015441])
    transformer_train_ir = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1, 0.1),
            transforms.ToTensor(),
            ir_norm,
        ]
    )
    transformer_train_depth = transforms.Compose(
    [
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        depth_norm,
    ]
    )
    transformer_test_ir = transforms.Compose(
        [
            #transforms.RandomResizedCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            ir_norm
        ]
    )
    transformer_test_depth = transforms.Compose(
        [
            #transforms.RandomResizedCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            depth_norm
        ]
    )
    data_txt = protoal+"_train.txt"
    train_data = CISIA_CeFA(root_path=root,txt_path=data_txt,transform_depth=transformer_train_depth,transform_ir=transformer_train_ir,mode = 'train')
    dev_txt = protoal + "_dev_ref.txt"
    dev_data = CISIA_CeFA_test(root_path=root, txt_path=dev_txt, transform_depth=transformer_test_depth,transform_ir=transformer_test_ir)

    total_size = train_data.__len__()
    idx = np.arange(total_size)
    
    train_loader = DataLoader(dataset=train_data, batch_size=train_batch, shuffle=False, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(dataset=dev_data, batch_size=test_batch, shuffle=False, num_workers=4,
                            pin_memory=True)
    return train_loader,val_loader

    
if __name__ == '__main__':
    train_loader,val_loader = load_cisia_csfa(protoal = "4@2")
    for i,data in enumerate(val_loader,1):
        rgb_img,depth_img,ir_img,label = data[0],data[1],data[2],data[3]
        lis = data[4]
        pdb.set_trace()
        norm = dict()





          

            

