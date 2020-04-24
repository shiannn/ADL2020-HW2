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
import torch.utils.data as Data

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
                if 'answerable' in qas:
                    #processed['answerable'] = qas['answerable']
                    processed['answerable'] = 1 if qas['answerable'] == True else 0

                if 'answers' in qas:
                    ans = qas['answers'][0]
                    processed['answersId'] = ans['id']

                    #processed['answersText'] = tokenizer.convert_tokens_to_ids(
                    #    tokenizer.tokenize(ans['text'])
                    #)
                    #processed['answersStart'] = ans['answer_start']
                    answersStart = ans['answer_start']
                    processed['answer_Tokens_Start'] = len(
                        tokenizer.tokenize(paragraph['context'][:answersStart])
                    ) if processed['answerable'] == 1 else -1

                    processed['answer_Tokens_End'] =\
                    processed['answer_Tokens_Start']+len(tokenizer.tokenize(ans['text']))\
                    if processed['answerable'] == 1 else -1
                
                """
                print('context')
                print(paragraph['context'])
                print('-----')
                print(qas['answers'])
                print(processed)
                print(processed['context'][processed['answer_Tokens_Start']:processed['answer_Tokens_End']])
                print(tokenizer.convert_ids_to_tokens(processed['context'][processed['answer_Tokens_Start']:processed['answer_Tokens_End']]))
                """
                processeds.append(processed)
        #exit(0)
    return processeds

def create_dataset(samples, save_path, config, tokenizer):
    dataset = BertDataset(
        samples, tokenizer=tokenizer,
        max_context_len=config.get('max_context_len') or 300,
        max_question_len=config.get('max_question_len') or 80
    )
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

def main(args):
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

    logging.info('Creating train dataset...')
    create_dataset(
        processeds, args.output_dir/'train.pkl', 
        config, 
        tokenizer
    )

    processeds = process_samples(tokenizer, valid[0]['data'])
    
    logging.info('Creating valid dataset...')
    create_dataset(
        processeds, args.output_dir/'valid.pkl', 
        config, 
        tokenizer
    )

    #with open(args.output_dir / 'train.pkl', 'wb') as f:
    #    pickle.dump(processeds, f)

def main2(args):
    with open(args.output_dir / 'train.pkl', 'rb') as f:
        A = pickle.load(f)

    loader = Data.DataLoader(
        dataset=A,
        batch_size=2,
        collate_fn=A.collate_fn
    )
    #print(A[0])
    for idx, batch in enumerate(loader):
        print(len(batch['sequence'][0]))
        print(len(batch['sequence'][1]))
        if idx == 40:
            print(batch['sequence'])
            exit(0)

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