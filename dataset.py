import os
import itertools
import logging as log

import torch
import torch.utils.data as data
# import allennlp.training.metrics.f1_measure.F1Measure as F1Measure
# import allennlp.training.metrics.categorical_accuracy.CategoricalAccuracy as Accuracy

from sklearn.feature_extraction.text import TfidfVectorizer

from metrics import CategoricalAccuracy, F1Measure
from utils import sent_tokenize, build_vocab, calc_tfidf_matrix


class Task():

    def __init__(self, args):
        self.n_classes = 2

        # load data
        corpus = []
        with open(os.path.join(args.data_dir, 'stop_words.txt'), 'r') as f:
            stop_words = f.readlines()
            stop_words = [line.strip() for line in stop_words]
            stop_words = []
        for split in ['train', 'test']:
            file_name = split + '.txt'
            with open(os.path.join(args.data_dir, file_name), 'r') as f:
                lines = f.readlines()                               # targ \t text
                lines = [line.strip().split('\t') for line in lines]        # [[targ, text], ...]
                log.info('Finished loading file %s' % file_name)
            texts = [sent_tokenize(text, se=args.model!='MLP') for targ, text in lines]   # [[word0, word1], ...]
            texts = [[word for word in sent if word not in stop_words] for sent in texts]
            targs = [int(targ) for targ, text in lines]                  # [targs]
            corpus += texts
            setattr(self, split + '_texts', texts)
            setattr(self, split + '_targs', targs)

        # form dataset
        if args.model == 'BiLSTM':
            self.collate_fn = self.collate_fn_seq
            self.vocab, self.word2idx = build_vocab(corpus, args.vocab_size)
            args.vocab_size = len(self.vocab)
            for split in ['train', 'test']:
                texts = getattr(self, split + '_texts')
                targs = getattr(self, split + '_targs')
                setattr(self, split + '_set', Dataset(texts, targs))
        elif args.model == 'MLP':
            self.collate_fn = self.collate_fn_tfidf
            tfidf, word_list = calc_tfidf_matrix(corpus, args.d_feature, stop_words=None)
            args.d_feature = len(word_list)
            for split in ['train', 'test']:
                split_size = len(getattr(self, split + '_texts'))
                inputs = tfidf[:split_size]
                tfidf = tfidf[split_size:]
                targs = getattr(self, split + '_targs')
                setattr(self, split + '_set', Dataset(inputs, targs))
        else: raise NotImplementedError
        log.info('Finished building datasets')

    def collate_fn_seq(self, batch):
        inputs, targs = [], []
        for x, y in batch:
            y = torch.tensor(y, dtype=torch.long)
            inputs.append(x)
            targs.append(y)

        targs = torch.stack(targs)
        return inputs, targs

    def collate_fn_tfidf(self, batch):
        inputs, targs = [], []
        for x, y in batch:
            x = torch.tensor(x.toarray(), dtype=torch.float)
            y = torch.tensor(y, dtype=torch.long)
            inputs.append(x)
            targs.append(y)

        inputs = torch.cat(inputs)
        targs = torch.stack(targs)
        return inputs, targs

class Dataset(data.Dataset):

    def __init__(self, inputs, targs):
        self.metrics = [CategoricalAccuracy(), F1Measure()]

        self.inputs = inputs
        self.targs = targs

    def __len__(self):
        assert len(self.inputs) == len(self.targs)
        # assert self.inputs.shape[0] == len(self.targs)
        return len(self.targs)

    def __getitem__(self, index):
        return self.inputs[index], self.targs[index]

    def metrics_count(self, pred, targ):
        for metric in self.metrics:
            metric(pred, targ)

    def metrics_report(self, reset=False):
        report = []
        for metric in self.metrics:
            _ = metric.get_metric(reset=False)
            if isinstance(metric, CategoricalAccuracy): report.append(('acc', _))
            elif isinstance(metric, F1Measure): report += [('rec', _[0]), ('pre', _[1]), ('f1', _[2])]
            else: raise NotImplementedError
        return report

    def metrics_reset(self):
        for metric in self.metrics:
            metric.reset()