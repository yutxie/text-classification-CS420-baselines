import os
import itertools
import logging as log

import torch
import torch.nn.functional as F

from torchtext.data import Iterator
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from metrics import Metrics
from evaluate import evaluate


def train(args, model, task):

    metrics = Metrics()
    writer = SummaryWriter(os.path.join(args.run_dir, 'tensorboard'))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    log.info('Start to train')

    n_passes = 0
    for epoch in range(args.n_epochs):
        data_iter = Iterator(
        # data_loader = DataLoader(
            task.train_set,
            args.batch_size,
            # collate_fn=task.collate_fn,
            device=args.device,
            shuffle=True,
        )

        for batch in data_iter:
        # for batch in data_loader:
            texts, targs = batch.text, batch.targ
            inputs = texts if args.model != 'MLP' else None
            # texts, targs = batch
            # inputs, targs = batch
            # targs = targs.to(args.device)

            model.train()
            preds = model(inputs)
            optimizer.zero_grad()
            loss = F.cross_entropy(preds, targs)
            loss.backward()
            optimizer.step()

            metrics.count(preds, targs)
            n_passes += 1

            # log train
            if n_passes % args.log_every == 0:
                report = metrics.report(reset=True)
                report += [('loss', loss.item())]
                writer.add_scalars('train', dict(report), n_passes)
                log.info('Pass #%i train: %s' % (n_passes, str(report)))

            # save model
            # if n_passes % args.save_every == 0:
            #     torch.save(model.state_dict(), os.path.join(args.run_dir, 'params_%i.model' % n_passes))

            # evaluate
            if n_passes % args.eval_every == 0:
                evaluate(args, model, task, tensorboard_writer=writer, n_passes=n_passes)

    log.info('Finished training')