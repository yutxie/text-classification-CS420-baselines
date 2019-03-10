import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, args, task):
        super().__init__()

        self.linears = nn.Sequential(
            nn.Linear(args.d_feature, args.d_hidden), nn.ReLU(), nn.Dropout(p=args.dropout),      # input
          *[nn.Sequential(
                nn.Linear(args.d_hidden, args.d_hidden), nn.ReLU(), nn.Dropout(p=args.dropout)    # hiddens
            ) for _ in range(args.n_layers)]
        )
        self.output = nn.Linear(args.d_hidden, task.n_classes)

        self.device = args.device
        self.to(self.device)

    def forward(self, x):
        x.to(self.device)

        x = self.linears(x)             # batch_size x d_hidden
        x = self.output(x)   # batch_size x n_classes
        return F.sigmoid(x)             # batch_size x n_classes