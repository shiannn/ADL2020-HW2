import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import BertTokenizer

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python3 plotAnswerLengthDistribution.py dataName saveName')
        exit(0)
    dataName = sys.argv[1]
    saveName = sys.argv[2]
    ansLengthFile = Path('ansLength.npy')
    print(ansLengthFile.exists())
    if(not ansLengthFile.exists()):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        distributionList = []
        with open(dataName, 'r') as f:
            A = json.load(f)
            #view all answer
            for data in A['data']:
                #print(data['paragraphs'])
                for paragraph in data['paragraphs']:
                    for qas in paragraph['qas']:
                        for ans in qas['answers']:
                            #temp = ('#' in ans['text'])
                            #if temp == True:
                            #    print('a')
                            print(ans)
                            ansTokens = tokenizer.tokenize(ans['text'])
                            print(ansTokens)
                            distributionList.append(len(ansTokens))
        
        np.save('ansLength', np.array(distributionList))
    
    ansLength = np.load(ansLengthFile)
    print(len(ansLength))
    bins = np.arange(0,120+5, step=5)
    print(bins)
    
    plt.hist(ansLength, bins=bins, edgecolor='black', cumulative=True, density=True)
    plt.xlabel('Length')
    plt.ylabel('Count (%)')
    plt.title('Cumulative Answer Length')
    plt.savefig(saveName/Path('length.png'))