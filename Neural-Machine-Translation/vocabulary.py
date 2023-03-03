from mosestokenizer import MosesTokenizer
from collections import Counter

class Vocabulary:
    def __init__(self, 
    config,
    data_path,
    frequency,
    lang,
    add_sos,
    add_eos,
    add_pad,
    add_unk):
        super(Vocabulary, self).__init__()
        self.config = config
        self.data_path = data_path
        self.tokenizer = MosesTokenizer(lang)
        self.counter = Counter()
        self.frequency = frequency
        self.add_sos = add_sos
        self.add_eos = add_eos
        self.add_pad = add_pad
        self.add_unk = add_unk

    def word2idx(self):
        word2idx = {}
        with open(self.data_path, encoding = 'utf-8', mode = 'r') as f:
            for line in f:
                self.counter.update(self.tokenizer(line))

        if self.add_pad:
            word2idx[self.config.pad_token] = len(word2idx)
        if self.add_unk:
            word2idx[self.config.unk_token] = len(word2idx)
        if self.add_sos:
            word2idx[self.config.start_token] = len(word2idx)
        if self.add_eos:
            word2idx[self.config.end_token] = len(word2idx)

        for word, freq in self.counter.items():
            if freq >= self.frequency:
                word2idx[word] = len(word2idx)

        return word2idx