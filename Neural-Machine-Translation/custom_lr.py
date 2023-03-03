import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
    initial_learning_rate,
    decay_rate, warmup_epoch):
        super(CustomSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.warmup_epoch = warmup_epoch
        
    def __call__(self, epoch):
        if epoch <= self.warmup_epoch:
            return self.initial_learning_rate
        else:
            return self.initial_learning_rate * (self.decay_rate**(epoch - self.warmup_epoch))