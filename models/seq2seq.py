import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import logging as log

class Seq2Seq(nn.Module):

    def __init__(self, args, task):
        super().__init__()
        self.vocab = task.vocab
        self.vocab_size = len(self.vocab)
        self.d_feature = args.d_feature

        self.embedding = nn.Embedding(
            self.vocab_size,
            args.d_feature,
            _weight=self.vocab.vectors
        )
        self.encoder = nn.GRU(
            args.d_feature,
            args.d_hidden,
            args.n_layers,
            batch_first=True,
            bidirectional=False
        )
        self.decoder = nn.GRU(
            args.d_feature,
            args.d_hidden,
            args.n_layers,
            batch_first=True,
            bidirectional=False
        )
        self.output2word = nn.Linear(
            args.d_hidden,
            self.vocab_size
        )

        self.device = args.device
        self.to(self.device)

    def forward(self, sents):
        '''
        Parameters
        ------------------
        sents: batch_size x seq_len
        '''
        sents = sents.to(self.device)
        batch_size = sents.shape[0]
        seq_len = sents.shape[1]
        
        # encoding
        embeded = self.embedding(sents)     # batch_size x seq_len x d_feature
        _, hidden = self.encoder(embeded)   # n_directions * n_layers x batch_size x d_hidden

        # decoding
        # embeded = self.embedding(sents[:,:-1])
        embeded = torch.zeros(batch_size, seq_len - 1, self.d_feature).to(self.device)
        output, _ = self.decoder(embeded, hidden) # batch_size x seq_len x n_directions * d_hidden
        pred = self.output2word(output)    # batch_size x seq_len x vocab_size
        pred = F.softmax(pred, dim=-1).view(-1, self.vocab_size)
        loss = F.cross_entropy(pred, sents[:,1:].contiguous().view(-1), ignore_index=1)
        pred = pred.argmax(dim=-1).view(batch_size, -1)

        return hidden, pred, loss
