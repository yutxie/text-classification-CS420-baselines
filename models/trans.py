import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import logging as log

from .modules import TransformerEncoder

class Transformer(nn.Module):

    def __init__(self, args, task):
        super().__init__()
        self.vocab = task.vocab

        self.embedding = nn.Embedding(
            len(self.vocab),
            args.d_feature,
            # _weight=self.vocab.vectors
        )
        self.trans = TransformerEncoder(
            dimension=args.d_feature,
            n_heads=4,
            hidden=args.d_hidden,
            num_layers=args.n_layers,
            dropout=args.dropout
        )
        self.output = nn.Linear(
            args.d_hidden * (args.n_layers + 1),
            task.n_classes
        )

        self.device = args.device
        self.to(self.device)

    def forward(self, x):
        '''
        inputs:
            x: batch_size x seq_len
            task_idx: int
        '''
        x = x.to(self.device)
        batch_size = x.shape[0]
        x = self.embedding(x)
        x = self.trans(x)
        x = [x[i][:,-1,:] for i in range(len(x))]
        x = torch.cat(x, dim=1)
        x = self.output(x)    # batch_size x n_classes
        return torch.softmax(x, dim=1) # batch_size x n_classes

    def seq2vec(self, x):
        x = x.to(self.device)
        batch_size = x.shape[0]
        x = self.embedding(x)
        x = self.trans(x)
        x = [x[i][:,-1,:] for i in range(len(x))]
        x = torch.cat(x, dim=1)
        return x