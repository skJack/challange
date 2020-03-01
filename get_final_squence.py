import numpy as np
import os
import cv2
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
import argparse
from torch.autograd.variable import Variable
#from models.all_sdnet import Model
from models.Model_newresnet import Model
#from models.mingnet import MingNet as Model
#from models.efficient_baseline import Model
import pdb
import yaml
parser = argparse.ArgumentParser(description='Generate final list')

parser.add_argument('--protoal', '--p', default='4@1', type=str,
                        help='protoal of dev list')
parser.add_argument('--config', default='config/cfg.yaml')
parser.add_argument('--save_path', default='/result/')
parser.add_argument('--which', default='test', type=str)

class CISIA_CeFA(Dataset):
    def __init__(self,root_path = 'media/sdc/datasets/Face-anti-spoofing/CASIA-CeFA/phase1',txt_path = '',transform_ir=None,transform_depth=None,sample = '0020'):
        self.root_path = root_path
        self.txt_path = txt_path
        self.transform_ir = transform_ir
        self.transform_depth = transform_depth
        self.sample = sample
        try:
            with open(os.path.join(root_path,txt_path),'r') as f:
                self.raw_list = f.read().splitlines()
            self.dev_list = self.raw_list
        except:
            print("can not open files,may be filelist is not exist")
            exit()
    def __getitem__(self, index):
        rgb_img_path = os.path.join(self.root_path,self.dev_list[index]+'/profile')
        frame_list = os.listdir(rgb_img_path)
        # delth_img_path = os.path.join(self.root_path,self.dev_list[index]+'/depth')
        # self.rgb_img_list = os.listdir(rgb_img_path)
        # ir_img_path = os.path.join(self.root_path,self.dev_list[index]+'/ir')
        # self.rgb_img_list = os.listdir(rgb_img_path)
        rgb_img_list = []
        depth_img_list = []
        ir_img_list = []
        hsv_img_list = []
        YCbCr_img_list = []
        for i,frame in enumerate(frame_list):
            if i%4==0:#隔两帧采样
                rgb_path = os.path.join(self.root_path,self.dev_list[index]+'/profile/'+frame)
                depth_path = os.path.join(self.root_path,self.dev_list[index]+'/depth/'+frame)
                ir_path = os.path.join(self.root_path,self.dev_list[index]+'/ir/'+frame)
                rgb_img = Image.open(rgb_path)
                depth_img = Image.open(depth_path)
                ir_img = Image.open(ir_path)
                hsv_img_cv = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2HSV)
                hsv_img = Image.fromarray(hsv_img_cv)
                YCbCr_img = rgb_img.convert('YCbCr')
                if self.transform_ir:
                    rgb_img = self.transform_ir(rgb_img)
                    depth_img = self.transform_depth(depth_img)
                    ir_img = self.transform_ir(ir_img)
                    hsv_img = self.transform_ir(hsv_img)
                    YCbCr_img = self.transform_ir(YCbCr_img)
                rgb_img_list.append(rgb_img)
                depth_img_list.append(depth_img)
                ir_img_list.append(ir_img)
                hsv_img_list.append(hsv_img)
                YCbCr_img_list.append(YCbCr_img)
        rgb_img = torch.stack(rgb_img_list, dim=0)
        depth_img = torch.stack(depth_img_list, dim=0)
        ir_img = torch.stack(ir_img_list, dim=0)
        hsv_img = torch.stack(hsv_img_list, dim=0)
        YCbCr_img = torch.stack(YCbCr_img_list, dim=0)
        return rgb_img,depth_img,ir_img,hsv_img,YCbCr_img,self.dev_list[index]
    def __len__(self):
        return len(self.dev_list)

def load_cisia_csfa_dev(root = '/media/sdc/datasets/Face-anti-spoofing/CASIA-CeFA/phase1',protoal = "4@1",img_size = 112,batch_size = 1,sample = '0020'):

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
    normalize = transforms.Normalize(mean=[0.14300402, 0.1434545, 0.14277956],  ##accorcoding to casia-surf val to commpute
                                     std=[0.10050353, 0.100842826, 0.10034215])
    # transformer_test = transforms.Compose(
    #     [
    #         transforms.RandomResizedCrop(img_size),      
    #         transforms.ToTensor(),
    #         normalize,
    #     ]
    # )
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
    data_txt = protoal+"_dev_res.txt"
    dev_data = CISIA_CeFA(root_path=root,txt_path=data_txt,transform_ir=transformer_test_ir,transform_depth=transformer_test_depth,sample = sample)
    total_size = dev_data.__len__()
    dev_loader = DataLoader(dataset=dev_data,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    return dev_loader

def load_cisia_csfa_test(root = '/media/sdc/datasets/Face-anti-spoofing/CASIA-CeFA/phase2',protoal = "4@1",img_size = 112,batch_size = 2,sample = '0020'):
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
    data_txt = protoal + "_test_res.txt"
    test_data = CISIA_CeFA(root_path=root,txt_path=data_txt,transform_ir=transformer_test_ir,transform_depth=transformer_test_depth,sample = sample)
    total_size = test_data.__len__()
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader


def inference():
    model = Model()
    #model = torch.nn.DataParallel(model)
    model.to(device)
    if args.protoal == '4@1':
        checkpoint = torch.load(p1_path)
        print(f"load checkpoint from {p1_path}")
    if args.protoal == '4@2':
        checkpoint = torch.load(p2_path)
        print(f"load checkpoint from {p2_path}")
    if args.protoal == '4@3':
        checkpoint = torch.load(p3_path)
        print(f"load checkpoint from {p3_path}")
    model.load_state_dict(checkpoint['state_dict'])

    #dev_loader = load_cisia_csfa_dev(protoal = args.protoal)
    if args.which=='test':
        print(args.test_path)
        data_loader = load_cisia_csfa_test(root = args.test_path,protoal=args.protoal,sample='0001',img_size = img_size,batch_size=1)
    elif args.which=='dev':
        data_loader = load_cisia_csfa_dev(root = args.train_path,protoal=args.protoal,sample='0005')
    path = 'submission1/{}_{}_{}_submission.txt'.format(args.which,args.protoal,args.arch)
    if os.path.isfile(path):
        os.remove(path)
    for i,data in enumerate(data_loader):
        rgb_img,depth_img,ir_img,hsv_img,YCbCr_img,dirs = data[0],data[1],data[2],data[3],data[4],data[5]
        rgb_var = Variable(rgb_img).float().to(device).squeeze()
        depth_var = Variable(depth_img).float().to(device).squeeze()
        ir_var = Variable(ir_img).float().to(device).squeeze()
        hsv_img_var = Variable(hsv_img).float().to(device).squeeze()
        YCbCr_img_var = Variable(YCbCr_img).float().to(device).squeeze()

        output = model(rgb_var, depth_var, ir_var,hsv_img_var,YCbCr_img_var,args.weight_list)
        output = torch.mean(output,0)
        #output,_ = torch.max(output,0)
        soft_output = torch.softmax(output,dim=-1).view(1,-1)
        preds = soft_output.to('cpu').detach().numpy()
        _,predicted = torch.max(soft_output.data, 1)
        predicted = predicted.to('cpu').detach().numpy()
        
        for i_batch in range(preds.shape[0]):
            print(f"inference {dirs[i_batch]}...")
            f = open(path, 'a+')
            f.write(dirs[i_batch]+ " "+str(preds[i_batch,1]) +'\n')
            f.close()
    print("done!")

if __name__ == '__main__':

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)
    img_size = args.input_size
    arch = args.arch
    choose_epoch = args.choose_epoch
    p1_path = f'./logs/4@1_{arch}/{arch}_{choose_epoch}.pth.tar'
    p2_path = f'./logs/4@2_{arch}/{arch}_{choose_epoch}.pth.tar'
    p3_path = f'./logs/4@3_{arch}/{arch}_{choose_epoch}.pth.tar'


    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    inference()


