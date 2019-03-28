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
            pad_first=False,
            batch_first=True
        )
        targ_field = data.Field(
            sequential=False,
            use_vocab=False,
            is_target=True
        )

        self.train_set = data.TabularDataset(
            path=os.path.join(args.data_dir, 'train_shuffle.txt'),
            format='tsv',
            fields=[('targ', targ_field), ('text', text_field)])
        self.test_set = data.TabularDataset(
            path=os.path.join(args.data_dir, 'test_shuffle.txt'),
            format='tsv',
            fields=[('targ', targ_field), ('text', text_field)])
        text_field.build_vocab(self.train_set)
        self.vocab = text_field.vocab
        log.info('Finished building a vocab of size %i' % len(self.vocab))

        word2idx, vectors = load_word_vector(args.data_dir, args.d_feature)
        self.vocab.set_vectors(word2idx, vectors, dim=args.d_feature)

class NonSeqTask():

    def __init__(self, args):
        self.n_classes = 2

        # load data
        corpus = []
        # with open(os.path.join(args.data_dir, 'stop_words.txt'), 'r') as f:
        #     stop_words = f.readlines()
        #     stop_words = [line.strip() for line in stop_words]
        #     stop_words = []
        for split in ['train', 'test']:
            file_name = split + '_shuffle.txt'
            with open(os.path.join(args.data_dir, file_name), 'r') as f:
                lines = f.readlines()                                       # targ \t text
                lines = [line.strip().split('\t') for line in lines]        # [[targ, text], ...]
                print('Finished loading file %s' % file_name)
            texts = [sent_tokenize(text) for targ, text in lines]           # [[word0, word1], ...]
            # texts = [[word for word in sent if word not in stop_words] for sent in texts]
            targs = [int(targ) for targ, text in lines] # [targs]
            exec(split + '_texts = texts')
            exec(split + '_targs = targs')
            corpus += texts

        class Dataset(data.Dataset):
            def __init__(self, inputs, targs):
                self.inputs = inputs
                self.targs = targs

            def __len__(self):
                return self.inputs.shape[0]

            def __getitem__(self, index):
                if self.targs is None: return self.inputs[index], None
                else: return self.inputs[index], self.targs[index]

        # form dataset
        tfidf, word_list = calc_tfidf_matrix(corpus, args.d_feature, stop_words=None)
        args.d_feature = len(word_list)
        for split in ['train', 'test']:
            texts = eval(split + '_texts')
            targs = eval(split + '_targs')
            split_size = len(texts)
            inputs = tfidf[:split_size]
            tfidf = tfidf[split_size:]
            setattr(self, split + '_set', Dataset(inputs, targs))
        log.info('Finished building datasets')

        print(len(self.train_set), len(self.test_set))

    def collate_fn(self, batch):
        inputs, targs = [], []
        for x, y in batch:
            x = torch.tensor(x.toarray(), dtype=torch.float)
            if y is not None: y = torch.tensor(y, dtype=torch.long)
            inputs.append(x)
            targs.append(y)

        inputs = torch.cat(inputs)
        if targs[0] is not None: targs = torch.stack(targs)
        else: targs = None
        return inputs, targs
