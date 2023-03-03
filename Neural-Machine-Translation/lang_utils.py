class Lang_utils:
    def __init__(self,
    word2idx,
    idx2word,
    lang,
    train_path,
    val_path,
    test_path):
        self.word_index = word2idx
        self.index_word = idx2word
        self.encode = lang.encode
        self.add_border = lang.add_border
        self.decode = lang.decode
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
