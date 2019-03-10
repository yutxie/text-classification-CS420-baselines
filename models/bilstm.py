import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import logging as log

class BiLSTM(nn.Module):

    def __init__(self, args, task):
        super().__init__()
        self.vocab = task.vocab

        self.embedding = nn.Embedding(
            len(self.vocab),
            args.d_feature,
            _weight=self.vocab.vectors
        )
        self.bilstm = nn.LSTM(
            args.d_feature,
            args.d_hidden,
            args.n_layers,
            batch_first=True,
            bidirectional=True
        )
        self.output = nn.Linear(
            args.d_hidden * args.n_layers * 2,
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
        x.to(self.device)
        batch_size = x.shape[0]
        x = self.embedding(x)
        x, (h_n, c_n) = self.bilstm(x)  # n_layers * 2 x batch_size x d_hidden
        h_n = h_n.transpose(0, 1).contiguous().view(batch_size, -1) # batch_size x d_hidden * n_layers * 2
        x = self.output(h_n)    # batch_size x n_classes
        return torch.sigmoid(x) # batch_size x n_classes
