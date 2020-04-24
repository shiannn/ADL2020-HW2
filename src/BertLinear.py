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
        self.startLinear = nn.Linear(self.hidden_pool, 1)
        self.endLinear = nn.Linear(self.hidden_pool, 1)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        ### input_ids (batch_size, seq_len)
        last_hidden_state, pooler_output = self.bert(input_ids,attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        answerable_score = self.answerable(self.dropout(pooler_output))
        
        ### last_hidden_state (batch_size, seq_len, hidden_size) [8, 440, 768]
        start_score = self.startLinear(last_hidden_state)
        end_score = self.endLinear(last_hidden_state)

        return answerable_score, start_score, end_score