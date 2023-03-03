class Config:
    def __init__(self,
    min_len = 0,
    max_len = 50,
    batch_size = 256,
    embedding_size = 512,
    hidden_units = 512,
    epochs = 20,
    learning_rate = 0.0008,
    dropout = 0.2,
    start_token = '<sos>',
    end_token = '<eos>',
    pad_token = '<pad>',
    unk_token = '<unk>',
    lang_1 = 'en',
    lang_2 = 'vi'):
        super(Config, self).__init__()
        self.min_len = min_len
        self.max_len = max_len
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.LANG_1 = lang_1
        self.LANG_2 = lang_2
