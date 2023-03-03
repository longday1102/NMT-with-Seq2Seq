import tensorflow as tf

class Dataloader:
    def __init__(self,
    en_utils,
    vi_utils,
    config):
        self.en_utils = en_utils
        self.vi_utils = vi_utils
        self.config = config
        self.train_process_en = []
        self.train_process_vi = []
        self.val_process_en = []
        self.val_process_vi = []
        self.test_process_en = []
        self.test_process_vi = []

    def read_data(self, path):
        with open(path, encoding = 'utf-8', mode = 'r') as f:
            file_opened = f.read().strip().split('\n')
        return file_opened

    def process_long_sequence(self, seq_1, seq_2):
        list_seq_1 = [seq_1[i:i + self.config.max_len] for i in range(0, len(seq_1), self.config.max_len)]
        list_seq_2 = [seq_2[i:i + self.config.max_len] for i in range(0, len(seq_2), self.config.max_len)]
        if len(list_seq_1) > len(list_seq_2):
            list_seq_1 = list_seq_1[:len(list_seq_2)]
        else:
            list_seq_2 = list_seq_2[:len(list_seq_1)]
        
        assert len(list_seq_1) == len(list_seq_2), "Length difference between 2 list"
        return list_seq_1, list_seq_2

    def padding(self, seq_batch, lang_utils):
        max_len = max([len(seq) for seq in seq_batch])
        for i in range(len(seq_batch)):
            seq = seq_batch[i]
            seq += [lang_utils.word_index[self.config.pad_token]]*(max_len - len(seq))
            seq_batch[i] = seq
        return seq_batch 
        
    def dataset(self):
        src_train_lang = self.read_data(self.en_utils.train_path)
        src_val_lang = self.read_data(self.en_utils.val_path)
        src_test_lang = self.read_data(self.en_utils.test_path)
        tgt_train_lang = self.read_data(self.vi_utils.train_path)
        tgt_val_lang = self.read_data(self.vi_utils.val_path)
        tgt_test_lang = self.read_data(self.vi_utils.test_path)

        seq_src_train = self.texts_to_sequences(src_train_lang, self.en_utils)
        seq_src_val = self.texts_to_sequences(src_val_lang, self.en_utils)
        seq_src_test = self.texts_to_sequences(src_test_lang, self.en_utils)
        seq_tgt_train = self.texts_to_sequences(tgt_train_lang, self.vi_utils)
        seq_tgt_val = self.texts_to_sequences(tgt_val_lang, self.vi_utils)
        seq_tgt_test = self.texts_to_sequences(tgt_test_lang, self.vi_utils)

        for en_seq, vi_seq in zip(seq_src_train, seq_tgt_train):  
            if self.config.min_len < len(en_seq) < self.config.max_len and self.config.min_len < len(vi_seq) < self.config.max_len:    
                self.train_process_en.append(self.en_utils.add_border(en_seq))
                self.train_process_vi.append(self.vi_utils.add_border(vi_seq))
            else:
                cut_en_seq, cut_vi_seq = self.process_long_sequence(en_seq, vi_seq)
                for en_gram, vi_gram in zip(cut_en_seq, cut_vi_seq):
                    self.train_process_en.append(self.en_utils.add_border(en_gram))
                    self.train_process_vi.append(self.vi_utils.add_border(vi_gram))
  
        assert len(self.train_process_en) == len(self.train_process_vi), "The size of the 2 training sets is not equal"
    
        for en_seq, vi_seq in zip(seq_src_val, seq_tgt_val):
            if self.config.min_len < len(en_seq) < self.config.max_len and self.config.min_len < len(vi_seq) < self.config.max_len:
                self.val_process_en.append(self.en_utils.add_border(en_seq))
                self.val_process_vi.append(self.vi_utils.add_border(vi_seq))
            else:
                cut_en_seq, cut_vi_seq = self.process_long_sequence(en_seq, vi_seq)
                for en_gram, vi_gram in zip(cut_en_seq, cut_vi_seq):
                    self.val_process_en.append(self.en_utils.add_border(en_gram))
                    self.val_process_vi.append(self.vi_utils.add_border(vi_gram))

        assert len(self.val_process_en) == len(self.val_process_vi), "The size of the 2 validation sets is not equal"
        
        for en_seq, vi_seq in zip(seq_src_test, seq_tgt_test):
            if self.config.min_len < len(en_seq) < self.config.max_len and self.config.min_len < len(vi_seq) < self.config.max_len:
                self.test_process_en.append(self.en_utils.add_border(en_seq))
                self.test_process_vi.append(self.vi_utils.add_border(vi_seq))
            else:
                cut_en_seq, cut_vi_seq = self.process_long_sequence(en_seq, vi_seq)
                for en_gram, vi_gram in zip(cut_en_seq, cut_vi_seq):
                    self.test_process_en.append(self.en_utils.add_border(en_gram))
                    self.test_process_vi.append(self.vi_utils.add_border(vi_gram))
                    
        assert len(self.test_process_en) == len(self.test_process_vi), "The size of the 2 testing sets is not equal"

        X_train = tf.convert_to_tensor(self.padding(self.train_process_en, self.en_utils))
        Y_train = tf.convert_to_tensor(self.padding(self.train_process_vi, self.vi_utils))
        X_val = tf.convert_to_tensor(self.padding(self.val_process_en, self.en_utils))
        Y_val = tf.convert_to_tensor(self.padding(self.val_process_vi, self.vi_utils))
        X_test = tf.convert_to_tensor(self.padding(self.test_process_en, self.en_utils))
        Y_test = tf.convert_to_tensor(self.padding(self.test_process_vi, self.vi_utils))
    
        print('Shape of train source tensor: ', X_train.shape)
        print('Shape of train target tensor: ', Y_train.shape)
        print('Shape of val source tensor: ', X_val.shape)
        print('Shape of val target tensor: ', Y_val.shape)
        print('Shape of test source tensor: ', X_test.shape)
        print('Shape of test target tensor: ', Y_test.shape)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(self.config.batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        
        print('Batch number of training set: ', len(train_ds))
        return train_ds, val_ds, test_ds