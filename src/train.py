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
import numpy as np

np.set_printoptions(threshold=10000000)
torch.set_printoptions(threshold=10000000)

FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

EPOCH = 3
BATCHSIZE = 8

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tokens2word(tokens_batch, tokenizer):
    ret = []
    for tokens in tokens_batch:
        tokens = torch.tensor(tokens).to(device)
        ret.append(tokenizer.convert_ids_to_tokens(tokens))
    return ret

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
    answerable_loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight_cal)
    start_loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    #start_loss_function = nn.NLLLoss(ignore_index=-1)
    end_loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    #end_loss_function = nn.NLLLoss(ignore_index=-1)
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

            token_type_ids = batch['token_type_ids']
            token_type_ids = torch.tensor(token_type_ids).to(device)
            #print('token_type_ids', token_type_ids)
            attention_mask = batch['attention_mask']
            #print('attention_mask', attention_mask)
            attention_mask = torch.tensor(attention_mask).to(device)

            context_mask = token_type_ids ^ attention_mask
            #print('context_mask', context_mask)
            ### context length
            context_Length = (context_mask==1).sum(dim=1)
            #print('context_Length', context_Length)
            ### using context length to see if answerable after truncated

            X = torch.tensor(X).to(device)
            ### print token
            ansToken = batch['answersText']
            #print('ansToken', ansToken)
            ansWords = tokens2word(ansToken, tokenizer)
            #context_words = tokens2word(X, tokenizer)
            print(ansWords)

            Y_answerable = batch['answerable']
            Y_answerable = torch.tensor(Y_answerable, dtype=torch.float).to(device)
            Y_start = batch['answer_Tokens_Start']
            Y_start = torch.tensor(Y_start, dtype=torch.float).to(device)
            Y_end = batch['answer_Tokens_End']
            Y_end = torch.tensor(Y_end, dtype=torch.float).to(device)

            check = []
            for x, yst, yed in zip(X, Y_start, Y_end):
                #print(yst, yed)
                stt = int(yst.item())
                edd = int(yed.item())
                #print(x[stt:edd])
                check.append(tokenizer.convert_ids_to_tokens(x[stt:edd]))
                
            print(check)
            #ans_word = tokens2word(X[:,Y_start:Y_end], tokenizer)
            #print(ans_word)
        
            ### mask the loss when end is out of context
            mask_st_ed = Y_end > context_Length
            igSted = torch.ones(Y_start.shape)* -1.
            igSted = igSted.to(device)
            Y_start = torch.where(mask_st_ed, igSted, Y_start).to(torch.long)
            Y_end = torch.where(mask_st_ed,igSted , Y_end).to(torch.long)
            ### make it unanswerable
            #print(Y_answerable)
            #print('mask_st_ed', mask_st_ed)
            unans = torch.zeros(Y_answerable.shape).to(device, dtype=torch.float)
            Y_answerable = torch.where(mask_st_ed, unans , Y_answerable)
            #print(Y_answerable)

            #print(Y_start)
            #print(Y_end)
            #Y = Y.t()
            #print(X)
            answerable_scores, start_score, end_score = model(input_ids=X, attention_mask=attention_mask, token_type_ids=token_type_ids)
            answerable_scores = answerable_scores.squeeze(1)
            start_score = start_score.squeeze()
            end_score = end_score.squeeze()

            #print(answerable_scores.shape, Y_answerable.shape)
            #print(start_score.shape, Y_start.shape)
            #print(end_score.shape, Y_end.shape)
            ### (batch_size, seq_len, 1)
            #print(answerable_scores)
            answerable_loss = answerable_loss_function(answerable_scores, Y_answerable)
            #print('Y_start', Y_start)
            ### answer is out of bound (513 > 512)
            ### count loss with question (actual position in context 
            ### is truncate  and become question)
            start_loss = start_loss_function(start_score, Y_start)
            #print('Y_end', Y_end)
            end_loss = end_loss_function(end_score, Y_end)

            #print(start_loss)
            #print(end_loss)

            TotalLoss = answerable_loss+start_loss+end_loss
            #print(TotalLoss)
            
            TotalLoss.backward()
            optimizer.step()
            print('epoch:{}/{} {}/{} loss:{}'.format(epoch, EPOCH, (idx+1)*BATCHSIZE, len(loader.dataset), TotalLoss))
            
        torch.save(model.state_dict(), config.get('checkpoint')+'BertLinear'+str(epoch)+'.pt')