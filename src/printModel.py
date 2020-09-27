import sys
import torch
from BertLinear import BertLinear

modelName = sys.argv[1]

model = BertLinear.from_pretrained('bert-base-chinese')
model.load_state_dict(torch.load(modelName))

print(model)