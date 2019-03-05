import os
import itertools
import logging as log

import torch.utils.data as data
import allennlp.training.metrics.f1_measure.F1Measure as F1Measure
import allennlp.training.metrics.categorical_accuracy.CategoricalAccuracy as Accuracy

from sklearn.feature_extraction.text import TfidfVectorizer

from utils import sent_tokenize


class TaskDataset():

    def __init__(self, args):
        self.n_classes = 2
        self.vocab = self.load_data(args.data_dir)

    def load_data(self, data_dir):
        vocab = []

        def _load(split='train'):
            file_name = split + '.txt'
            with open(os.path.join(data_dir, file_name), 'r') as f:
                lines = f.readlines().split('\n')                   # targ \t text
                lines = [line.split('\t') for line in lines]        # [[targ, text], ...]
            texts = [sent_tokenize(text) for targ, text in lines]   # [[word0, word1], ...]
            targs = [targ for targ, text in lines]                  # [targs]
            vocab += sum(texts) # list of all words
            setattr(self, split + '_set', Dataset(texts, targs))

        _load('train')
        _load('test')
        return set(vocab)

    def collate_fn(self, batch):
        inputs, targs = [], []
        for x, y in batch:
            y = torch.tensor(y, dtype=torch.long)
            inputs.append(x)
            targs.append(y)

        targs = torch.stack(targs)
        return inputs, targs

class Dataset(data.Dataset):

    def __init__(self, texts, targs):
        self.metrics = [Accuracy(), F1Measure()]

        self.inputs = texts
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