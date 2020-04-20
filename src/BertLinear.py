import torch.nn as nn
import torch
from transformers import BertModel, BertPreTrainedModel


if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#nn.Module
class BertLinear(BertPreTrainedModel):
    def __init__(self, config):
        super(BertLinear, self).__init__(config)
        self.hidden_pool = 768
        self.dropout = nn.Dropout(0.2)
        #self.bert = BertModel.from_pretrained(config)
        self.bert = BertModel(config)
        self.answerable = nn.Linear(self.hidden_pool, 1)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        ### input_ids (batch_size, seq_len)
        #print(input_ids)
        #One = torch.ones(input_ids.shape).to(device)
        #Zero = torch.zeros(input_ids.shape).to(device)
        #attentionMask = torch.where(input_ids>0, One, Zero)
        #print(attentionMask)
        #print('attention_mask')
        #print(attention_mask)
        #print('token_type_ids')
        #print(token_type_ids)
        last_hidden_state, pooler_output = self.bert(input_ids,attention_mask=attention_mask, token_type_ids=token_type_ids)
        answerable_score = self.answerable(self.dropout(pooler_output))

        return answerable_score