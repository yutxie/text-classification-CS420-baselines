import os
import itertools
import logging as log

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from evaluate import evaluate


def train(args, model, mtl_dataset):

    writer = SummaryWriter(os.path.join(args.run_dir, 'tensorboard'))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    n_passes = 0
    for epoch in range(args.n_epochs):

        data_loaders = [
            DataLoader(
                task.train_set,
                args.batch_size,
                shuffle=True,
                collate_fn=task.collate_fn
            ) for task in mtl_dataset.tasks
        ]
        batches = sum([[(
            task_idx, batch
            ) for batch in data_loader
            ] for task_idx, data_loader in enumerate(data_loaders)
        ])

        for task_idx, batch in batches:
            inputs, targs = batch
            inputs = inputs.to(args.device)
            targs = targs.to(args.device)

            model.train()
            preds = model(inputs, task_idx)
            optimizer.zero_grad()
            loss = F.cross_entropy(preds, targs)
            loss.backward()
            optimizer.step()

            train_set = mtl_dataset.tasks[task_idx].train_set
            trainset.metrics_count(preds, targs)
            n_passes += 1

            # log train
            if n_passes % args.log_train == 0:
                metrics = train_set.metrics_report(reset=True)
                metrics += [('loss', loss.item())]
                writer.add_scalars('train', dict(metrics), n_passes)
                log.info('Epoch #%i train: %s' % (epoch, str(metrics)))

            # save model
            if n_passes % args.save_every == 0:
                torch.save(model.state_dict(), os.path.join(args.run_dir, 'params_%i.model' % n_passes))

            # evaluate
            if n_passes % args.eval_every == 0:
                evaluate(args, model, mtl_dataset, tensorboard_writer=writer, n_passes=n_passes)