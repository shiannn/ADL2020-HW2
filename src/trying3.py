import json



with open('pre.json', 'r') as f:
    A = json.load(f)
    for a in A:
        print(A[a])

"""
with open('data/dev.json', 'r') as f:
    A = json.load(f)
    #view all answer
    for data in A['data']:
        #print(data['paragraphs'])
        for paragraph in data['paragraphs']:
            for qas in paragraph['qas']:
                for ans in qas['answers']:
                    temp = ('#' in ans['text'])
                    if temp == True:
                        print('a') 
"""