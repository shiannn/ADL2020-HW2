import torch
from torch.utils.data import Dataset
from utils import pad_to_len

class BertDataset(Dataset):
    def __init__(self, data, tokenizer=None, max_context_len=300, max_question_len=80):
        self.data = data

        self.pad_token = tokenizer.pad_token
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token

        self.cls_token_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        print(self.cls_token_id)
        self.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        print(self.sep_token_id)
        self.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        print(self.pad_token_id)

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
    
    def collate_fn(self, samples):
        batch = {}
        for key in ['id', 'answersId', 'answersStart', 'answerable']:
            batch[key] = [sample[key] for sample in samples]

        ### key == 'context' 'question'
        to_len = max([
                len(sample['context'])+len(sample['question'])+3
                for sample in samples
            ])

        padded = pad_to_len(
            [
                [self.cls_token_id]+sample['context']+[self.sep_token_id]+ 
                sample['question'] +[self.sep_token_id] for sample in samples
            ],
            to_len, self.pad_token_id
        )
        batch['sequence'] = padded
        
        return batch