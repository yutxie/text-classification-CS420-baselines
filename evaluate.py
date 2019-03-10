import logging as log

import torch

from torchtext.data import BucketIterator
from torch.utils.data import DataLoader

from metrics import Metrics


def evaluate(args, model, task, tensorboard_writer=None, n_passes=-1):
    model.eval()

    metrics = Metrics()
    data_iter = BucketIterator(
        task.test_set,
        args.batch_size,
        device=args.device,
        shuffle=True,
    )

    for batch in data_iter:
        inputs, targs = batch
        inputs = inputs.to(args.device)
        targs = targs.to(args.device)
        preds = model(inputs)
        metrics.count(preds, targs)

    report = metrics.report(reset=True)
    log.info('Pass #%i evaluate: %s' % (n_passes, str(report)))
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalars('evaluate', dict(report), n_passes)