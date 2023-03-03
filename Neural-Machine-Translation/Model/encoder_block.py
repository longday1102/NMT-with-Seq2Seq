import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional

class Encoder(tf.keras.Model):
    def __init__(self, config, en_vocab_size):
        super(Encoder, self).__init__()
        self.Embedding = Embedding(en_vocab_size,
                                   config.embedding_size)
        self.Bi_LSTM = Bidirectional(LSTM(config.hidden_units,
                                   return_sequences = True,
                                   return_state = True,
                                   dropout = config.dropout),
                                   merge_mode = 'sum')

    def __call__(self, x):
        enc_embedding = self.Embedding(x)
        enc_outputs, fw_h, fw_c, bw_h, bw_c = self.Bi_LSTM(enc_embedding)
        state_h = fw_h + bw_h
        state_c = fw_c + bw_c
        enc_states = [[state_h, state_c], [state_h, state_c]]
        return enc_outputs, enc_states