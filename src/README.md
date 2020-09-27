# ADL2020 HW2 Bert for QA README
## training
1.  mkdir checkpoint
2.  ./training.sh [json file for training data]

-   this will produce model each epoch in directory 'checkpoint'

## predicting
1.  ./run.sh [json file for testing data] [name for predicting file]

## ploting Performance versus each Answerable Threshold
-   ckiptagger and evaluation script are needed
    -   cd scripts
    -   wget https://ckip.iis.sinica.edu.tw/data/ckiptagger/data.zip
    -   unzip data.zip

1.  ./performanceVsThreshold.sh [json file for validating data]

-   this will produce resultX.json of different threshold (threshold=0.1,0.3,...0.9, X=1,3,...,9)

2.  python3.6 plotAnswerableThreshold.py 
    [Name of Folder contains resultX.json of different threshold] [Folder to save chart]

## ploting Answer Length Distribution
1.  python3.6 plotAnswerLengthDistribution.py
    [json file for training data] [Folder to save chart]

    -   it will produce ansLength.npy file to fast up