import os
import itertools
import logging as log

import torch.utils.data as data
import allennlp.training.metrics.f1_measure.F1Measure as F1Measure
import allennlp.training.metrics.categorical_accuracy.CategoricalAccuracy as Accuracy

from sklearn.feature_extraction.text import TfidfVectorizer

from utils import sent_tokenize, build_vocab, calc_tfidf_matrix


class TaskDataset():

    def __init__(self, args):
        self.n_classes = 2
        
        corpus = self._load_data(args)
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
            tfidf, word_list = calc_tfidf_matrix(corpus, args.d_feature)
            args.d_feature = len(word_list)
            for split in ['train', 'test']:
                split_size = len(getattr(self, split, '_texts'))
                inputs = tfidf[:split_size]
                tfidf = tfidf[split_size:]
                targs = getattr(self, split + '_targs')
                setattr(self, split + '_set', Dataset(inputs, targs))
        else: raise NotImplementedError

    def _load_data(self, args):
        corpus = []
        for split in ['train', 'test']:
            file_name = split + '.txt'
            with open(os.path.join(args.data_dir, file_name), 'r') as f:
                lines = f.readlines().split('\n')                   # targ \t text
                lines = [line.split('\t') for line in lines]        # [[targ, text], ...]
            texts = [sent_tokenize(text) for targ, text in lines]   # [[word0, word1], ...]
            targs = [targ for targ, text in lines]                  # [targs]
            corpus += texts
            setattr(self, split + '_texts', texts)
            setattr(self, split + '_targs', targs)
        return corpus

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
            y = y.equal(self.first_group).long()
            inputs.append(x)
            targs.append(y)

        inputs = torch.cat(inputs)
        targs = torch.stack(targs)
        return inputs, targs

class Dataset(data.Dataset):

    def __init__(self, inputs, targs):
        self.metrics = [Accuracy(), F1Measure()]

        self.inputs = inputs
        self.targs = targs

    def __len__(self):
        assert len(self.inputs) == len(self.targs)
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targs[index]

    def metrics_count(self, pred, targ):
        for metric in self.metrics:
            metric(pred, targ)

    def metrics_report(self, reset=False):
        report = []
        for metric in self.metrics:
            _ = metric.get_metric(reset=False)
            if isinstance(metric, Accuracy): report.append(('acc', _))
            elif isinstance(metric, F1Measure): report.append(('rec', _[0]), ('pre', _[1]), ('f1', _[2]))
            else: raise NotImplementedError
        return report

    def metrics_reset(self):
        for metric in self.metrics:
            metric.reset()