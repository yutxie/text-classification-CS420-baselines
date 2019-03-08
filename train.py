import os
import itertools
import logging as log

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from evaluate import evaluate


def train(args, model, task):

    writer = SummaryWriter(os.path.join(args.run_dir, 'tensorboard'))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    log.info('Start to train')

    n_passes = 0
    for epoch in range(args.n_epochs):

        data_loader = DataLoader(
            task.train_set,
            args.batch_size,
            shuffle=True,
            collate_fn=task.collate_fn
        )

        for batch in data_loader:
            inputs, targs = batch
            # inputs = inputs.to(args.device)
            targs = targs.to(args.device)

            model.train()
            preds = model(inputs)
            optimizer.zero_grad()
            loss = F.cross_entropy(preds, targs)
            loss.backward()
            optimizer.step()

            task.train_set.metrics_count(preds, targs)
            n_passes += 1

            # log train
            if n_passes % args.log_every == 0:
                metrics = task.train_set.metrics_report(reset=True)
                metrics += [('loss', loss.item())]
                writer.add_scalars('train', dict(metrics), n_passes)
                log.info('Pass #%i train: %s' % (n_passes, str(metrics)))

            # save model
            if n_passes % args.save_every == 0:
                torch.save(model.state_dict(), os.path.join(args.run_dir, 'params_%i.model' % n_passes))

            # evaluate
            if n_passes % args.eval_every == 0:
                evaluate(args, model, task, tensorboard_writer=writer, n_passes=n_passes)

    log.info('Finished training')