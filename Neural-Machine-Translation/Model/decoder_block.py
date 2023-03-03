import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Dense

class Decoder(tf.keras.Model):
    def __init__(self, config, attention, vi_vocab_size):
        super(Decoder, self).__init__()
        self.Embedding = Embedding(vi_vocab_size,
                                  config.embedding_size)
        self.LSTM_1 = LSTM(config.hidden_units,
                          dropout = config.dropout,
                          return_sequences = True,
                          return_state = True)
        self.LSTM_2 = LSTM(config.hidden_units,
                          dropout = config.dropout,
                          return_sequences = True,
                          return_state = True)
        self.Fc = Dense(vi_vocab_size, activation = 'softmax')
        self.attention = attention
        
    def __call__(self, x, enc_outputs, states):
        x = tf.expand_dims(x, axis = 1)
        dec_embedding = self.Embedding(x)
        dec_outputs1 = self.LSTM_1(dec_embedding,
                                  initial_state = states[0])
        dec_outputs2 = self.LSTM_2(dec_outputs1[0],
                                   initial_state = states[1])
        
        dec_outputs = dec_outputs2[0]
        context, _ = self.attention(enc_outputs, dec_outputs)
        dec_concat = tf.concat([dec_outputs, context], axis = -1)
        final_concat = tf.reshape(dec_concat, (-1, dec_concat.shape[2]))
        
        final_outs = self.Fc(final_concat)
        dec_states = [dec_outputs1[1:], dec_outputs2[1:]]
        return final_outs, dec_states