#! /bin/bash

for i in 1 3 5 7 9
do
    #echo $i
    threshold=0$(echo "scale=1; $i / 10" | bc -l )
    echo $threshold
    echo pre$i.json
    python3.6 predict.py checkpoint/BertLinear0.pt $threshold ../datasets/valid.pkl pre$i.json
    python3.6 ../scripts/evaluate.py ../src/data/dev.json pre$i.json ../result$i.json ../scripts/data/
done