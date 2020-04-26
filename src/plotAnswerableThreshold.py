import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python3 plotAnswerableThreshold.py FolderName SaveName')
        exit(0)

    FoldetName = sys.argv[1]
    SaveName = sys.argv[2]
    ### the folder path of result[1-5].json
    result1Path = FoldetName / Path('result1.json')
    result3Path = FoldetName / Path('result3.json')
    result5Path = FoldetName / Path('result5.json')
    result7Path = FoldetName / Path('result7.json')
    result9Path = FoldetName / Path('result9.json')

    with open(result1Path, 'r') as f:
        result1 = json.load(f)
    with open(result3Path, 'r') as f:
        result3 = json.load(f)
    with open(result5Path, 'r') as f:
        result5 = json.load(f)
    with open(result7Path, 'r') as f:
        result7 = json.load(f)
    with open(result9Path, 'r') as f:
        result9 = json.load(f)

    results = [result1, result3, result5, result7, result9]

    overallEMs = []
    overallF1s = []
    answerableEMs = []
    answerableF1s = []
    unanswerableEMs = []
    unanswerableF1s = []
    for result in results:
        overallEMs.append(result['overall']['em'])
        overallF1s.append(result['overall']['f1'])
        answerableEMs.append(result['answerable']['em'])
        answerableF1s.append(result['answerable']['f1'])
        unanswerableEMs.append(result['unanswerable']['em'])
        unanswerableF1s.append(result['unanswerable']['f1'])
    
    Thresholds = [0.1,0.3,0.5,0.7,0.9]
    print(np.arange(0.75, 1, step=0.025))

    plt.suptitle('Performance on Different Threshold')
    plt.subplot(1,2,1)
    plt.title('F1')
    plt.xticks(np.array(Thresholds))
    plt.yticks(np.arange(0.725, 1, step=0.025))
    plt.ylim(0.70, 1)
    plt.plot(Thresholds, overallF1s, marker='o')
    plt.plot(Thresholds, answerableF1s, marker='o')
    plt.plot(Thresholds, unanswerableF1s, marker='o')

    plt.subplot(1,2,2)
    plt.title('EM')
    plt.xticks(np.array(Thresholds))
    plt.yticks(np.arange(0.725, 1, step=0.025), labels=[])
    plt.ylim(0.70, 1)
    plt.plot(Thresholds, overallEMs, marker='o', label='overall')
    plt.plot(Thresholds, answerableEMs, marker='o', label='answerable')
    plt.plot(Thresholds, unanswerableEMs, marker='o', label='unanswerable')
    plt.legend(bbox_to_anchor=(0.5, 0.6, 0.5, 0.5))
    #loc='upper right'
    plt.text(-0.4, 0.67, 'Answerable Threshold')

    plt.savefig(SaveName/Path('thres.png'))