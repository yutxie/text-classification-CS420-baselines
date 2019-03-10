import os
import itertools
import logging as log

import torch
import torch.utils.data as data

from torchtext import data, datasets
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import sent_tokenize, calc_tfidf_matrix, load_word_vector


class SeqTask():

    def __init__(self, args):
        self.n_classes = 2

        text_field = data.Field(
            sequential=True,
            init_token='<SOS>',
            eos_token='<EOS>',
            lower=True,
            tokenize=sent_tokenize,
            batch_first=True
        )
        targ_field = data.Field(
            sequential=False,
            use_vocab=False,
            is_target=True
        )

        train_data, test_data = data.TabularDataset.splits(
            path=args.data_dir,
            train='train.txt',
            test='test.txt',
            format='tsv',
            fields=[('targ', targ_field), ('text', text_field)])
        text_field.build_vocab(train_data)
        
        self.train_set = train_data
        self.test_set = test_data

        self.vocab = text_field.vocab

        word2idx, vectors = load_word_vector(args.data_dir, args.d_feature)
        self.vocab.set_vectors(word2idx, vectors, dim=args.d_feature)

class NonSeqTask():

        # load data
        corpus = []
        # with open(os.path.join(args.data_dir, 'stop_words.txt'), 'r') as f:
        #     stop_words = f.readlines()
        #     stop_words = [line.strip() for line in stop_words]
        #     stop_words = []
        for split in ['train', 'test']:
            file_name = split + '.txt'
            with open(os.path.join(args.data_dir, file_name), 'r', encoding='utf-8') as f:
                lines = f.readlines()                               # targ \t text
                lines = [line.strip().split('\t') for line in lines]        # [[targ, text], ...]
                log.info('Finished loading file %s' % file_name)
            texts = [sent_tokenize(text, se=args.model!='MLP') for targ, text in lines]   # [[word0, word1], ...]
            # texts = [[word for word in sent if word not in stop_words] for sent in texts]
            targs = [int(targ) for targ, text in lines]                  # [targs]
            corpus += texts
            setattr(self, split + '_texts', texts)
            setattr(self, split + '_targs', targs)

        class Dataset(data.Dataset):
            def __init__(self, tfidf, texts, targs):
                self.metrics = [CategoricalAccuracy(), F1Measure()]

                self.tfidf = tfidf
                self.texts = texts
                self.targs = targs

            def __len__(self):
                assert self.tfidf is None or self.tfidf.shape[0] == len(self.targs)
                assert len(self.texts) == len(self.targs)
                return len(self.targs)

            def __getitem__(self, index):
                return self.tfidf[index], self.texts, self.targs[index]

        # form dataset
        tfidf, word_list = calc_tfidf_matrix(corpus, args.d_feature, stop_words=None)
        args.d_feature = len(word_list)
        for split in ['train', 'test']:
            texts = getattr(self, split + '_texts')
            targs = getattr(self, split + '_targs')
            split_size = len(targs)
            inputs = tfidf[:split_size]
            tfidf = tfidf[split_size:]
            setattr(self, split + '_set', Dataset(inputs, texts, targs))
        log.info('Finished building datasets')

    def collate_fn(self, batch):
        inputs, targs = [], []
        for x, y in batch:
            x = torch.tensor(x.toarray(), dtype=torch.float)
            y = torch.tensor(y, dtype=torch.long)
            inputs.append(x)
            targs.append(y)

        inputs = torch.cat(inputs)
        targs = torch.stack(targs)
        return inputs, targs
