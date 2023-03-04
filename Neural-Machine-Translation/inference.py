import time
from train import encoder, decoder, en_utils, vi_utils, config
import tensorflow as tf

class Translation:
    def __init__(self,
    encoder,
    decoder,
    en_utils,
    vi_utils,
    config):
        self.encoder = encoder
        self.decoder = decoder
        self.en_utils = en_utils
        self.vi_utils = vi_utils
        self.config = config
    
    def predict(self, input_sentence, redundant_max_len = 10):
        sequence = []
        input_sequence = self.en_utils.add_border(self.en_utils.encode(input_sentence))
        max_len = len(input_sequence) + redundant_max_len
        
        X_test = tf.expand_dims(input_sequence, axis = 0)
        enc_outputs, last_states = self.encoder(X_test)
        dec_inputs = tf.constant([self.vi_utils.word_index[self.config.start_token]])
        for _ in range(max_len):
            start_time = time.time()
            dec_outputs, last_states = self.decoder(dec_inputs, enc_outputs, last_states)
            pred_id = tf.argmax(dec_outputs, axis = 1).numpy()
            dec_inputs = pred_id
            sequence.append(pred_id[0])
            if pred_id[0] == self.vi_utils.word_index[self.config.end_token]:
                break
        translated = self.vi_utils.decode(sequence)
        translated_sentence = " ".join([word for word in translated.split(" ") if word not in [self.config.pad_token,
                                                                                               self.config.start_token,
                                                                                               self.config.end_token]])
        print(f'Input sentence: {input_sentence}')
        print(f'Translated sentence: {translated_sentence}')
        print('Translation time: %.2fs' % (time.time() - start_time))

translation = Translation(encoder,
                         decoder,
                         en_utils,
                         vi_utils,
                         config)

# input_sentence = 'What is your name?'
# print(translation.predict(input_sentence))

# input_sentence = "They had been dancing for an hour when there was a knock on the door"
# print(translation.predict(input_sentence))

# input_sentence = "You have made the very same mistake again"
# print(translation.predict(input_sentence))
