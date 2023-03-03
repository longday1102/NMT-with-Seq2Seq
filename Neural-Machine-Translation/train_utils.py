from tqdm import tqdm
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

class Trainer:
    def __init__(self,
    encoder,
    decoder,
    config,
    train_ds,
    val_ds,
    test_ds,
    vi_utils,
    evaluate,
    maksedloss,
    lr_schedule,
    decay_rate = 0.5,
    warmup_epoch = 9,
    max_norm = 1.0):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.vi_utils = vi_utils
        self.evaluate = evaluate
        self.loss = maksedloss
        self.lr_schedule = lr_schedule(config.learning_rate, decay_rate, warmup_epoch)
        self.optimizer = Adam()
        self.max_norm = max_norm

    def training(self):
        for epoch in range(self.config.epochs):
            start_time = time.time()
            total_loss = 0
            self.optimizer.lr.assign(self.lr_schedule(epoch))
            print('Current learning rate: ', self.optimizer.learning_rate.numpy())

            for _, (x, y) in tqdm(enumerate(self.train_ds.take(len(self.train_ds)))):
                loss = 0
                with tf.GradientTape() as tape:
                    enc_outputs, last_states = self.encoder(x)
                    dec_inputs = tf.constant([self.vi_utils.word_index[self.config.start_token]]*len(x))
                    for i in range(1, y.shape[1]):
                        dec_outputs, last_states = self.decoder(dec_inputs, enc_outputs, last_states)
                        loss += self.loss(y[:, i], dec_outputs)
                        dec_inputs = y[:, i]

                    train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
                    grads = tape.gradient(loss, train_vars)
                    clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_norm)
                    self.optimizer.apply_gradients(zip(clipped_grads, train_vars))

                total_loss += loss
            
            print(f'Epoch: {epoch + 1} -- Loss: {total_loss}')
            print('Time taken: %.2fs' % (time.time() - start_time))
            print('----------------------------------------------------------------')
        
        BLEU_2012 = self.evaluate.evaluation(self.encoder, self.decoder, self.val_ds)
        BLEU_2013 = self.evaluate.evaluation(self.encoder, self.decoder, self.test_ds)
        print()
        print('***************************************** END OF TRAINING *****************************************')
        print()
        print(f'BLEU score is calculated on test set 2012: {BLEU_2012}')
        print(f'BLEU score is calculated on test set 2013: {BLEU_2013}')

