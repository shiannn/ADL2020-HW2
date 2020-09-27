from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
X = '1234567 7654321'
Y = 'we all live in the yellow submarine'
Z = '教練，我想打籃球'
retX = tokenizer.tokenize(X)
retY = tokenizer.tokenize(Y)
retZ = tokenizer.tokenize(Z)
print(X)
print(retX)
print(Y)
print(retY)
print(Z)
print(retZ)