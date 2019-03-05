import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, args, mtl_dataset):
        super().__init__()
        self.tasks = mtl_dataset.tasks

        self.linears = nn.Sequential([
            nn.Linear(args.d_feature, args.d_hidden), nn.ReLU(), nn.Dropout(),      # input
          *[nn.Sequential(
                nn.Linear(args.d_hidden, args.d_hidden), nn.ReLU(), nn.Dropout()    # hiddens
            ) for _ in range(args.n_layers)]
        ])
        self.outputs = nn.ModuleList([
            nn.Linear(args.d_hidden, task.n_classes
            ) for task in self.tasks
        ])

    def forward(self, x, task_idx):
        x = self.linears(x)             # batch_size x d_hidden
        x = self.outputs[task_idx](x)   # batch_size x n_classes
        return F.sigmoid(x)             # batch_size x n_classes