import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import SparseCategoricalCrossentropy

class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(MaskedLoss, self).__init__()
        self.loss = SparseCategoricalCrossentropy(from_logits = True)

    def __call__(self, y_true, y_pred):
        mask = 1 - np.equal(y_true, 0)
        loss = self.loss(y_true, y_pred)*mask
        return tf.reduce_mean(loss)
