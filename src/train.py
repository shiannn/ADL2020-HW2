import torch
import pickle
import torch.utils.data as Data
from dataset import BertDataset
from BertLinear import BertLinear

if __name__=='__main__':
    with open('../datasets/train.pkl', 'rb') as f:
        A = pickle.load(f)

    loader = Data.DataLoader(
        dataset=A,
        batch_size=2,
        collate_fn=A.collate_fn
    )
    #print(A[0])
    model = BertLinear()

    for idx, batch in enumerate(loader):
        X = batch['sequence']
        X = torch.tensor(X)
        print(X)
        scores = model(X)
        print(scores)
        exit(0)