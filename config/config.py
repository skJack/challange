import logging
import time
import shutil
import os
RESNET34 = "resnet34"
RESNET50 = "resnet50"
RESNET32 = "resnet32"
resume = False#是否载入保存模型
device = 'cuda'
num_workers = 4
print_freq = 100
momentum = 0.9
weight_decay = 5e-4
#weight_decay = 0.01
max_epoch = 100
data_aug = True
pretrain = True

