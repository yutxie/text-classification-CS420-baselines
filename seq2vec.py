import os
import pickle
import itertools
import logging as log

import torch
import torch.nn.functional as F

from torchtext.data import Iterator
from torch.utils.data import DataLoader


def seq2vec(args, model, task):
    log.info('Starting to computing features')
    model.eval()

    for split in ['train', 'test']:
        dataset = task.train_set if split == 'train' else task.test_set
        data_iter = Iterator(
            dataset,
            args.batch_size,
            device=args.device,
            shuffle=False
        )

        feats_list = []
        for batch in data_iter:
            inputs, targs = batch.text, batch.targ
            feats = model.seq2vec(inputs)
            feats_list.append(feats)
        feats = torch.cat(feats_list, dim=0)
        feats = feats.detach().cpu().numpy()

        with open(os.path.join(args.data_dir, 'feats_%s' % split), 'wb') as f:
            pickle.dump(feats, f)