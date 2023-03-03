import tensorflow as tf
from tensorflow.keras.layers import Dense

class Luong_Attention(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Luong_Attention, self).__init__()
        self.Wa = Dense(config.hidden_units)
    
    def __call__(self, enc_outputs, dec_outputs):
        score = tf.matmul(dec_outputs, self.Wa(enc_outputs), transpose_b = True)
        alignment = tf.nn.softmax(score, axis = 2)
        context = tf.matmul(alignment, enc_outputs)
        return context, score