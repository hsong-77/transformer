import os
import spacy
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k

class Tokenizer:

    def __init__(self):
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except IOError:
            os.system("python -m spacy download de_core_news_sm")
            self.spacy_de = spacy.load("de_core_news_sm")

        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except IOError:
            os.system("python -m spacy download en_core_web_sm")
            self.spacy_en = spacy.load("en_core_web_sm")

    def tokenize_de(self, text):
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token

    def make_dataset(self):
        if self.ext == ('.de', '.en'):
          self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token, lower=True, batch_first=True)
          self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token, lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
          self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token, lower=True, batch_first=True)
          self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token, lower=True, batch_first=True)

          train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
          return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test), batch_size=batch_size)
        return train_iterator, valid_iterator, test_iterator


class Data:
  
    def __init__(self):
        pass

    def load_data(self, batch_size):
        tokenizer = Tokenizer()
        loader = DataLoader(ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')

        train, valid, test = loader.make_dataset()
        loader.build_vocab(train_data=train, min_freq=2)
        self.train_iter, self.valid_iter, self.test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size)

        self.src_pad_idx = loader.source.vocab.stoi['<pad>']
        self.tgt_pad_idx = loader.target.vocab.stoi['<pad>']

        self.src_vocab_size = len(loader.source.vocab)
        self.tgt_vocab_size = len(loader.target.vocab)

# data = Data()
# data.load_data(128)
# print(data.src_vocab_size)
# print(data.tgt_vocab_size)
# print(data.src_pad_idx)
# print(data.tgt_pad_idx)
