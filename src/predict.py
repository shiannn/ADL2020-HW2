import sys
import torch
import torch.nn as nn
from torch.nn import Sigmoid
import pickle
import json
import torch.utils.data as Data
from dataset import BertDataset
from BertLinear import BertLinear
from tqdm import tqdm 

BATCHSIZE = 5


if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    if(len(sys.argv)!=3):
        print('usage: python3 predict.py TestingData.pkl predict.json')
        exit(0)
    
    testDataName = sys.argv[1]
    predictName = sys.argv[2]

    with open('../datasets/config.json') as f:
        config = json.load(f)

    with open(testDataName, 'rb') as f:
        A = pickle.load(f)
    
    loader = Data.DataLoader(
        dataset=A,
        batch_size=BATCHSIZE,
        collate_fn=A.collate_fn
    )
    #model = BertLinear()
    model = BertLinear.from_pretrained('bert-base-chinese')

    modelName = config["checkpoint"] + 'BertLinear1000.pt'

    model.load_state_dict(torch.load(modelName))
    model = model.to(device)
    model.eval()
    f = Sigmoid()

    to_Write = {}
    with torch.no_grad():
        with open(predictName,'w') as f_predict:
            for batch in tqdm(loader):
                questionId = batch['id']
                #print(questionId)
                X = batch['input_ids']
                X = torch.tensor(X).to(device)

                token_type_ids = batch['token_type_ids']
                token_type_ids = torch.tensor(token_type_ids).to(device)
                attention_mask = batch['attention_mask']
                attention_mask = torch.tensor(attention_mask).to(device)

                answerable_scores = model(input_ids=X, attention_mask=attention_mask, token_type_ids=token_type_ids)
                decide = f(answerable_scores)

                oneS = torch.ones(answerable_scores.shape).to(device)
                zeroS = torch.zeros(answerable_scores.shape).to(device)
                predictLabel = torch.where(decide>0.5,oneS,zeroS)

                #print(answerable_scores)
                #print(predictLabel)
                for i in range(len(questionId)):
                    if predictLabel[i].item() == 1:
                        to_Write[questionId[i]] = "有答案"
                    else:
                        to_Write[questionId[i]] = ""
                
            json.dump(to_Write, f_predict)