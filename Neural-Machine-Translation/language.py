from mosestokenizer import MosesTokenizer

class Language:
    def __init__(self,
    config,
    lang,
    word2idx,
    idx2word):
        self.config = config
        self.tokenizer = MosesTokenizer(lang)
        self.word2idx = word2idx
        self.idx2word = idx2word

    def encode(self, sentence):
        sequence = []
        for word in self.tokenizer(sentence):
            if word in self.word2idx:
                sequence.append(self.word2idx[word])
            else:
                sequence.append(self.word2idx[self.config.unk_token])
        return sequence

    def add_border(self, sequence):
        return [self.word2idx[self.config.start_token]] + sequence + [self.word2idx[self.config.end_token]]
    
    def decode(self, sequence):
        sentence = [self.idx2word[idx] for idx in sequence]
        sentence = " ".join(sentence)
        return sentence

