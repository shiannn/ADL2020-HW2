import torch
from torch.utils.data import Dataset
from utils import pad_to_len

class BertDataset(Dataset):
    def __init__(self, data, tokenizer=None, max_context_len=300, max_question_len=80):
        self.data = data

        self.tokenizer = tokenizer
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
            'context': self.data[index]['context'],
            'question': self.data[index]['question'],
            'answersId': self.data[index]['answersId'],
            'answersText': self.data[index]['answersText'],
            'answer_Tokens_Start': self.data[index]['answer_Tokens_Start'],
            'answer_Tokens_End': self.data[index]['answer_Tokens_End'],
            'answerable': self.data[index]['answerable']
        }
    
    def collate_fn(self, samples):
        batch = {}
        for key in ['id', 'answersId', 'answersText', 'answer_Tokens_Start', 'answer_Tokens_End', 'answerable']:
            batch[key] = [sample[key] for sample in samples]

        ### key == 'context' 'question' ###
        #to_len = 500
        to_len = max([
                len(sample['context'])+len(sample['question'])+3
                for sample in samples
            ])
        #to_len = min(512, to_len)
        to_len = 512
        """
        padded = pad_to_len(
            [
                [self.cls_token_id]+sample['context']+[self.sep_token_id]+ 
                sample['question'] +[self.sep_token_id] for sample in samples
            ],
            to_len, self.pad_token_id
        )
        """
        batch['input_ids'] = []
        batch['token_type_ids'] = []
        batch['attention_mask'] = []
        for sample in samples:
            #print(sample['question'])
            retDict = self.tokenizer.prepare_for_model(sample['context'], sample['question'], max_length=to_len, truncation_strategy='only_first', pad_to_max_length=True, add_special_tokens=True)
            #print(retDict.keys())
            batch['input_ids'].append(retDict['input_ids'])
            batch['token_type_ids'].append(retDict['token_type_ids'])
            batch['attention_mask'].append(retDict['attention_mask'])
        """
        padded = [
            self.tokenizer.prepare_for_model(sample['context'], sample['question'], max_length=to_len,
            truncation_strategy='only_first', pad_to_max_length=True) for sample in samples
        ]
        """
        #print(padded)
        #batch['sequence'] = padded
        
        return batch