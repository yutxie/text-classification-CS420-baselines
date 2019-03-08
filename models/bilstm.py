import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import logging as log

class BiLSTM(nn.Module):

    def __init__(self, args, task):
        super().__init__()
        self.vocab = task.vocab
        self.word2idx = task.word2idx

        self.embedding = nn.Embedding(
            len(self.vocab),
            args.d_feature
        )
        self.bilstm = nn.LSTM(
            args.d_feature,
            args.d_hidden,
            args.n_layers,
            bidirectional=True
        )
        self.output = nn.Linear(
            args.d_hidden * args.n_layers * 2,
            task.n_classes
        )

    def init_embedding(self, weights):
        pass

    def forward(self, x):
        '''
        inputs:
            x: [[word0, word1, ...], [...], ...]
            task_idx: int
        '''
        batch_size = len(x)
        idxs = [[
            self.word2idx[word] for word in sent
            ] for sent in x
        ]
        print(idxs)

        x = self.embedding(torch.tensor(idxs, dtype=torch.long))    # batch_size x d_feature
        x, (h_n, c_n) = self.bilstm(x)                              # n_layers * 2 x batch_size x d_hidden
        h_n = h_n.transpose(0, 1).view(batch_size, -1)              # batch_size x d_hidden * n_layers * 2
        x = self.output(h_n)                             # batch_size x n_classes
        return F.sigmoid(x)                                         # batch_size x n_classes
