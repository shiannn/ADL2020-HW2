import torch
from torch.utils.data import Dataset

class BertDataset(Dataset):
    def __init__(self, data, padding=0, max_context_len=300, max_question_len=80):
        self.data = data
        self.padding = padding
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'id': self.data[index]['id'],
            'context': self.data[index]['context'][:self.max_context_len],
            'question': self.data[index]['question'][:self.max_question_len],
            'answersId': self.data[index]['answersId'],
            'answersText': self.data[index]['answersText'],
            'answersStart': self.data[index]['answersStart'],
            'answerable': self.data[index]['answerable']
        }