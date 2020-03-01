
import yaml
import argparse
import os

parser = argparse.ArgumentParser(description='Generate final list')
parser.add_argument('--config', default='../config/cfg.yaml')
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f)
for k, v in config['common'].items():
    setattr(args, k, v)
detail = args.arch

def submission(test_4_1,test_4_2,test_4_3,dev_4_1 = '4@1_baseline_submission.txt',dev_4_2 = '4@2_baseline_submission.txt',dev_4_3 = '4@3_baseline_submission.txt'):
    all_list = []

    with open(dev_4_1, 'r') as f:
        all_list.extend(f.read().splitlines())
    with open(test_4_1, 'r') as f:
        all_list.extend(f.read().splitlines())
    with open(dev_4_2, 'r') as f:
        all_list.extend(f.read().splitlines())
    with open(test_4_2, 'r') as f:
        all_list.extend(f.read().splitlines())
    with open(dev_4_3, 'r') as f:
        all_list.extend(f.read().splitlines())
    with open(test_4_3, 'r') as f:
        all_list.extend(f.read().splitlines())
    if os.path.isfile('final.txt'):
        os.remove('final.txt')
    file = open('final.txt','w')
    for i,t in enumerate(all_list):
        print(f'write {i} file~')
        file.write(t+'\n')
    file.close()
dev_4_1 = 'dev_4@1_'+detail+'_submission.txt'
dev_4_2 = 'dev_4@2_'+detail+'_submission.txt'
dev_4_3 = 'dev_4@3_'+detail+'_submission.txt'

test_4_1 = 'test_4@1_'+detail+'_submission.txt'
test_4_2 = 'test_4@2_'+detail+'_submission.txt'
test_4_3 = 'test_4@3_'+detail+'_submission.txt'
# test_4_1 = '/home/kesun/competition/clean/submission/_resnet9_cut_4@1_0.99.txt'
# test_4_2 = '/home/kesun/competition/clean/submission/_resnet9_cut_4@2_0.95.txt'
# test_4_3 = '/home/kesun/competition/clean/submission/_resnet9_cut_4@3_0.999.txt'
submission(test_4_1,test_4_2,test_4_3,dev_4_1,dev_4_2,dev_4_3)
