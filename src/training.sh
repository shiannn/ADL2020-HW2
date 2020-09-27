#! /bin/bash

trainJson=$1

### mode 0 for training 
### mode 1 for validating
python3.6 modifyConfig.py $trainJson 0
python3.6 preprocess.py datasets 0
python3.6 train.py datasets/train.pkl