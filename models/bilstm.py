import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import logging as log

class BiLSTM(nn.Module):

    def __init__(self, args, mtl_dataset):
        super().__init__()
        self.tasks = mtl_dataset.tasks
        self.vocab = mtl_dataset.vocab
        self.word_to_idx = mtl_dataset.word_to_idx

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
        self.outputs = nn.ModuleList([
            nn.Linear(
                args.d_hidden * args.n_layers * 2,
                task.n_classes
            ) for task in self.tasks
        ])

    def init_embedding(self, weights):
        pass

    def forward(self, x, task_idx):
        '''
        inputs:
            x: [[word0, word1, ...], [...], ...]
            task_idx: int
        '''
        batch_size = x.shape[0]
        idxs = [[
            self.word_to_idx[word] for word in sent
            ] for sent in x
        ]

        x = self.embedding(torch.tensor(idxs, dtype=torch.long))    # batch_size x d_feature
        x, (h_n, c_n) = self.bilstm(x)                              # n_layers * 2 x batch_size x d_hidden
        h_n = h_n.transpose(0, 1).view(batch_size, -1)              # batch_size x d_hidden * n_layers * 2
        x = self.outputs[task_idx](h_n)                             # batch_size x n_classes
        return F.sigmoid(x)                                         # batch_size x n_classes
