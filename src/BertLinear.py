import torch.nn as nn
import torch
from transformers import BertModel

class BertLinear(nn.Module):
    def __init__(self):
        super(BertLinear, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
    
    def forward(self, sequences):
        output = self.bert(sequences)

        return output