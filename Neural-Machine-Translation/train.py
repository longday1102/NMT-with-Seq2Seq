from config import Config
from vocabulary import Vocabulary
from language import Language
from lang_utils import Lang_utils
from dataloader import Dataloader
from Model.encoder_block import Encoder
from Model.attention_layer import Luong_Attention
from Model.decoder_block import Decoder
from loss import MaskedLoss
from metrics import Evaluate
from custom_lr import CustomSchedule
from train_utils import Trainer

config = Config()

en_train_file_path = '/content/NMT-with-Seq2Seq/Neural-Machine-Translation/Dataset/train.en.txt'
vi_train_file_path = '/content/NMT-with-Seq2Seq/Neural-Machine-Translation/Dataset/train.vi.txt'
en_val_file_path = '/content/NMT-with-Seq2Seq/Neural-Machine-Translation/Dataset/tst2012.en.txt'
vi_val_file_path = '/content/NMT-with-Seq2Seq/Neural-Machine-Translation/Dataset/tst2012.vi.txt'
en_test_file_path = '/content/NMT-with-Seq2Seq/Neural-Machine-Translation/Dataset/tst2013.en.txt'
vi_test_file_path = '/content/NMT-with-Seq2Seq/Neural-Machine-Translation/Dataset/tst2013.vi.txt'

source_add = {'add_sos': True,
              'add_eos': True,
              'add_pad': True,
              'add_unk': True}
target_add = {'add_sos': True,
              'add_eos': True,
              'add_pad': True,
              'add_unk': True}

frequency = 5

source_vocab = Vocabulary(config,
                          en_train_file_path,
                          frequency,
                          config.LANG_1,
                          source_add['add_sos'],
                          source_add['add_eos'],
                          source_add['add_pad'],
                          source_add['add_unk'])
target_vocab = Vocabulary(config,
                          vi_train_file_path,
                          frequency,
                          config.LANG_2,
                          target_add['add_sos'],
                          target_add['add_eos'],
                          target_add['add_pad'],
                          target_add['add_unk'])

source_word2idx = source_vocab.word2idx()
target_word2idx = target_vocab.word2idx()
source_idx2word = {value: key for key, value in source_word2idx.items()}
target_idx2word = {value: key for key, value in target_word2idx.items()}

source_lang = Language(config,
                       config.LANG_1,
                       source_word2idx,
                       source_idx2word)
target_lang = Language(config,
                       config.LANG_2,
                       target_word2idx,
                       target_idx2word)

en_utils = Lang_utils(source_word2idx,
                      source_idx2word,
                      source_lang,
                      en_train_file_path,
                      en_val_file_path,
                      en_test_file_path)
vi_utils = Lang_utils(target_word2idx,
                      target_idx2word,
                      target_lang,
                      vi_train_file_path,
                      vi_val_file_path,
                      vi_test_file_path)

train_ds, val_ds, test_ds = Dataloader(en_utils, vi_utils, config).dataset()

en_vocab_size = len(en_utils.word_index)
vi_vocab_size = len(vi_utils.word_index)

encoder = Encoder(config, en_vocab_size)
attention = Luong_Attention(config)
decoder = Decoder(config, attention, vi_vocab_size)

maskedloss = MaskedLoss()
evaluate = Evaluate(config, vi_utils)

print('EN_VOCAB_SIZE: ', en_vocab_size)
print('VI_VOCAB_SIZE: ', vi_vocab_size)

history = Trainer(encoder,
                  decoder,
                  config,
                  train_ds,
                  val_ds,
                  test_ds,
                  vi_utils,
                  evaluate,
                  maskedloss,
                  CustomSchedule)

history.training()
