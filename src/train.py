import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import json
import logging
import torch.utils.data as Data
from dataset import BertDataset
from BertLinear import BertLinear
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

EPOCH = 1
BATCHSIZE = 8

def countClassNum(training):
    zeroNum = 0
    oneNum = 0
    for td in training:
        #print(td['answerable'])
        if td['answerable']==1:
            oneNum += 1
        else:
            zeroNum += 1
    return zeroNum, oneNum

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__=='__main__':
    with open('../datasets/config.json') as f:
        config = json.load(f)

    with open(config['training'], 'rb') as f:
        A = pickle.load(f)

    zeroNum, oneNum = countClassNum(A)
    print(zeroNum, oneNum)
    pos_weight_cal = torch.tensor(zeroNum / oneNum, dtype=torch.float)
    pos_weight_cal = pos_weight_cal.to(device)
    
    loader = Data.DataLoader(
        dataset=A,
        batch_size=BATCHSIZE,
        collate_fn=A.collate_fn
    )
    #print(A[0])
    logging.info('loading model!')
    model = BertLinear.from_pretrained('bert-base-chinese').to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight_cal)
    #loss_function = nn.BCELoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.00001)
    optimizer = AdamW(model.parameters(), lr=0.00001, eps=1e-8)

    ### ['id', 'answersId', 'answersStart', 'answerable', 'sequence'] ###
    ### ['id', 'answersId', 'answersStart', 'answerable', 'input_ids', 'token_type_ids', 'attention_mask']
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    for epoch in range(EPOCH):
        for idx, batch in enumerate(loader):
            model.train()
            optimizer.zero_grad()
            X = batch['input_ids']

            #print(tokenizer.convert_ids_to_tokens(X[0]))
            #print(tokenizer.convert_ids_to_tokens(X[1]))
            #print(tokenizer.convert_ids_to_tokens(X[2]))

            token_type_ids = batch['token_type_ids']
            token_type_ids = torch.tensor(token_type_ids).to(device)
            attention_mask = batch['attention_mask']
            attention_mask = torch.tensor(attention_mask).to(device)

            
            X = torch.tensor(X).to(device)
            Y = batch['answerable']
            Y = torch.tensor(Y, dtype=torch.float64).to(device)
            #Y = Y.t()
            #print(Y)
            #print(X)
            answerable_scores = model(input_ids=X, attention_mask=attention_mask, token_type_ids=token_type_ids)

            answerable_scores = answerable_scores.squeeze(1)
            #print(answerable_scores)
            loss = loss_function(answerable_scores, Y)
            loss.backward()
            optimizer.step()
            print('epoch:{}/{} {}/{} loss:{}'.format(epoch, EPOCH, (idx+1)*BATCHSIZE, len(loader.dataset), loss))
            if idx % 1000 == 0 and idx > 0:
                torch.save(model.state_dict(), config.get('checkpoint')+'BertLinear'+str(idx)+'.pt')