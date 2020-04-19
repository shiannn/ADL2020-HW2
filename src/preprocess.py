import argparse
from pathlib import Path
import logging
import os
import json
import pickle
from tqdm import tqdm

import torch
from transformers import BertTokenizer

from dataset import BertDataset

def process_samples(tokenizer, samples):
    processeds = []
    for sample in tqdm(samples):
        paragraphs = sample['paragraphs']
        for paragraph in paragraphs:
            for qas in paragraph['qas']:
                processed = {
                    'context': tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(paragraph['context'])
                    ),
                    'id': qas['id'],
                    'question': tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(qas['question'])
                    )
                }
                if 'answers' in qas:
                    ans = qas['answers'][0]
                    processed['answersId'] = ans['id']

                    processed['answersText'] = tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(ans['text'])
                    )
                    processed['answersStart'] = ans['answer_start']
                if 'answerable' in qas:
                    processed['answerable'] = qas['answerable']
                
                processeds.append(processed)

    return processeds

def create_dataset(samples, save_path, config, padding=0):
    dataset = BertDataset(
        samples, padding=padding,
        max_context_len=config.get('max_context_len') or 300,
        max_question_len=config.get('max_question_len') or 80
    )
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

def main1(args):
    logging.info('Creating train dataset...')
    with open(args.output_dir / 'config.json') as f:
        config = json.load(f)
    print(config)
    with open(config['train']) as f:
        train = [json.loads(line) for line in f]
    with open(config['dev']) as f:
        valid = [json.loads(valid) for valid in f]
    with open(config['test']) as f:
        test = [json.loads(line) for line in f]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    processeds = process_samples(tokenizer, train[0]['data'])

    with open(args.output_dir / 'train.pkl', 'wb') as f:
        pickle.dump(processeds, f)

def main2(args):
    with open(args.output_dir / 'config.json') as f:
        config = json.load(f)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    with open(args.output_dir / 'train.pkl', 'rb') as f:
        A = pickle.load(f)
    #print(A)
    create_dataset(
        A, args.output_dir/'temp.pkl', 
        config, 
        tokenizer.pad_token
    )

def main(args):
    with open(args.output_dir / 'temp.pkl', 'rb') as f:
        A = pickle.load(f)

    print(A[0])

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    
    #print(args)
    main(args)