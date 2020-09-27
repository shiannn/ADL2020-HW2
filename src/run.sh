#! /bin/bash

testJson=$1
predictJson=$2

### mode 0 for training 
### mode 1 for validating
python3.6 modifyConfig.py $testJson 1
python3.6 preprocess.py datasets 1
python3.6 predict.py checkpoint/BertLinear0.pt 0.1 datasets/valid.pkl $predictJson