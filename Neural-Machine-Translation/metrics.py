import tensorflow as tf
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

class Evaluate:
    def __init__(self,
    config,
    vi_utils):
        self.config = config
        self.vi_utils = vi_utils
    
    def BLEU(self, y_pred, y_true):
        smoothing_function = SmoothingFunction()
        bleu_score = 100 * corpus_bleu(list_of_references = [[seq] for seq in y_true],
                                                             hypotheses = y_pred,
                                                             smoothing_function = smoothing_function.method0)
        return round(bleu_score, 2)
    
    def remove(self, sequence):
        sentence = self.vi_utils.decode(sequence)
        new_sequence = [word for word in sentence.split(" ") if word not in [self.config.pad_token,
                                                                             self.config.start_token,
                                                                             self.config.end_token]]
        return new_sequence

    def evaluation(self, encoder, decoder, dataset):
        y_true = []
        y_pred = []
        for x_test, Y_test in dataset.shuffle(buffer_size = 1, seed = 1).take(len(dataset)):
            X_test = tf.expand_dims(x_test, axis = 0)
            enc_outputs, last_states = encoder(X_test)
            dec_inputs = tf.constant([self.vi_utils.word_index[self.config.start_token]])
            sequence = []
            for _ in range(len(Y_test)):
                dec_outputs, last_states = decoder(dec_inputs, enc_outputs, last_states)
                pred_id = tf.argmax(dec_outputs, axis = 1).numpy()
                dec_inputs = pred_id
                sequence.append(pred_id[0])
            y_pred.append(self.remove(sequence))   
            y_true.append(self.remove(Y_test.numpy()))
            
        BLEU_score = self.BLEU(y_pred, y_true)
        return BLEU_score 
