class Tokenizer:
    def __init__(self, vocab=None, do_lower_case=True):
        self.nlp = BertTokenizer.from_pretrained('bert-base-chinese')