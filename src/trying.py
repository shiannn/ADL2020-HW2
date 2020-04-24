import argparse
from pathlib import Path
import logging
import os
import json
import pickle
from tqdm import tqdm

import logging
import torch
from transformers import BertTokenizer

from dataset import BertDataset
import torch.utils.data as Data

FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

def main(args):
    logging.info('loading model!')
    with open(args.output_dir / 'train.pkl', 'rb') as f:
        A = pickle.load(f)

    loader = Data.DataLoader(
        dataset=A,
        batch_size=5,
        collate_fn=A.collate_fn
    )
    #print(A[0])
    logging.info('loading tokenizer!')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    for idx, batch in enumerate(loader):
        #print(len(batch['sequence'][0]))
        #print(len(batch['sequence'][1]))
        print(batch.keys())
        #print(batch['input_ids'])
        print(tokenizer.convert_ids_to_tokens(batch['input_ids'][0]))
        print(tokenizer.convert_ids_to_tokens(batch['input_ids'][1]))
        print(tokenizer.convert_ids_to_tokens(batch['input_ids'][2]))
        print(tokenizer.convert_ids_to_tokens(batch['input_ids'][3]))
        print(tokenizer.convert_ids_to_tokens(batch['input_ids'][4]))
        print(len(batch['input_ids'][0]))
        print(len(batch['input_ids'][1]))
        print(len(batch['input_ids'][2]))
        print(len(batch['input_ids'][3]))
        print(len(batch['input_ids'][4]))
        #print(type(batch['sequence']))
        #print(batch['sequence'][0])
        exit(0)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    main(args)