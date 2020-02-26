# Cross-ethnicity Face anti-spoofing Recognition Challenge

This our code for Chalearn Multi-modal Cross-ethnicity Face anti-spoofing Recognition Challenge@CVPR2020（Phase2）


## Change datapath
change the train_path and test_path in /config/cfg.yaml


## Train

####  Train 
```
1. python train.py --protoal 4@1
2. python train.py --protoal 4@2
3. python train.py --protoal 4@3
```


## Inference

#### test
```
1. python get_final_score.py --protoal 4@1
2. python get_final_score.py --protoal 4@2
3. python get_final_score.py --protoal 4@3
```

## merge and submit
1. cd submission
2. python get_final.py
3. The output file named final.txt
