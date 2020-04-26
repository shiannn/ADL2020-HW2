import sys
import re
import torch
import torch.nn as nn
from torch.nn import Sigmoid
import pickle
import json
import torch.utils.data as Data
from dataset import BertDataset
from BertLinear import BertLinear
from transformers import BertTokenizer
from tqdm import tqdm 

BATCHSIZE = 5
TOPK = 10

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tokens2word(tokens, tokenizer):
    ret = tokenizer.convert_ids_to_tokens(tokens)
    return ret

def postprocessing(startTopVal, startTopIdx, endTopVal, endTopIdx):
    print(startTopVal)
    print(startTopIdx)
    print(endTopVal)
    print(endTopIdx)
    StartEnds = []
    for startIdxs, endingIdxs, startVals, endVals in zip(startTopIdx, endTopIdx, startTopVal, endTopVal):
        st = 0
        ed = 0
        while((ed < len(endingIdxs) and st < len(startIdxs))\
        and (endingIdxs[ed] <= startIdxs[st] or endingIdxs[ed].item()-startIdxs[st].item()>30)):
            # move st/ed
            if startVals[st] > endVals[ed]:
                ed += 1
            else:
                st += 1
        if (ed >= len(endingIdxs) or st >= len(startIdxs)):
            StartEnds.append([-1,-1])
        else:
            StartEnds.append([startIdxs[st].item(), endingIdxs[ed].item()])

    return StartEnds


if __name__ == '__main__':
    if(len(sys.argv)!=5):
        print('usage: python3 predict.py model.pt threshold TestingData.pkl predict.json')
        exit(0)
    
    modelName = sys.argv[1]
    threshold = float(sys.argv[2])
    testDataName = sys.argv[3]
    predictName = sys.argv[4]

    with open('../datasets/config.json') as f:
        config = json.load(f)

    with open(testDataName, 'rb') as f:
        A = pickle.load(f)
    
    loader = Data.DataLoader(
        dataset=A,
        batch_size=BATCHSIZE,
        collate_fn=A.collate_fn
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertLinear.from_pretrained('bert-base-chinese')

    #modelName = config["checkpoint"] + 'BertLinear1000.pt'

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
                print('X', X)

                token_type_ids = batch['token_type_ids']
                token_type_ids = torch.tensor(token_type_ids).to(device)
                attention_mask = batch['attention_mask']
                attention_mask = torch.tensor(attention_mask).to(device)

                answerable_scores, start_score, end_score = model(input_ids=X, attention_mask=attention_mask, token_type_ids=token_type_ids)
                answerable_scores = answerable_scores.squeeze(1)
                start_score = start_score.squeeze()
                end_score = end_score.squeeze()
                print(start_score)
                print(end_score)
                print('start_score.shape', start_score.shape)
                print('end_score.shape', end_score.shape)

                ### should mask the score of (question) and (padding)
                context_mask = token_type_ids ^ attention_mask
                negInf = torch.full(start_score.shape, float('-inf')).to(device)
                start_score = torch.where(context_mask==0, negInf, start_score) #inf
                end_score = torch.where(context_mask==0, negInf, end_score) #inf
                print(start_score)
                print(end_score)
                startTopVal, startTopIdx = torch.topk(start_score, TOPK, dim=1)
                endTopVal, endTopIdx = torch.topk(end_score, TOPK, dim=1)
                
                ### postprocessing get (start token id) and (end token id)
                retStEd = postprocessing(startTopVal, startTopIdx, endTopVal, endTopIdx)

                print(retStEd)

                decide = f(answerable_scores)

                oneS = torch.ones(answerable_scores.shape).to(device)
                zeroS = torch.zeros(answerable_scores.shape).to(device)
                predictLabel = torch.where(decide>threshold,oneS,zeroS)

                #print(answerable_scores)
                #print(predictLabel)
                for i in range(len(questionId)):
                    st = retStEd[i][0]
                    ed = retStEd[i][1]
                    if predictLabel[i].item() == 0:
                        to_Write[questionId[i]] = ""
                    elif(st==-1 or ed==-1):
                        to_Write[questionId[i]] = ""
                    elif predictLabel[i].item() == 1:
                        #to_Write[questionId[i]] = "有答案"
                        ### tokenizer.convert [st:ed]
                        temp = tokenizer.convert_ids_to_tokens(X[i][st:ed])
                        temp = list(filter(\
                        lambda x:(x!='[UNK]')\
                        , temp))
                        print(temp)
                        temp = ''.join(temp)
                        print(temp)
                        temp = re.sub('#', '', temp)
                        temp = re.sub('「', '', temp)
                        temp = re.sub('」', '', temp)
                        temp = re.sub('《', '', temp)
                        temp = re.sub('》', '', temp)
                        
                        print(temp)
                        # Also remove the << and up <
                        to_Write[questionId[i]] = temp
                        
                
            json.dump(to_Write, f_predict)