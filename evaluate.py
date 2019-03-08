import logging as log

import torch

from torch.utils.data import DataLoader


def evaluate(args, model, task, tensorboard_writer=None, n_passes=-1):
    model.eval()

    data_loader = DataLoader(
        task.test_set,
        args.batch_size,
        shuffle=False,
        collate_fn=task.collate_fn
    )

    for batch in data_loader:
        inputs, targs = batch
        inputs = inputs.to(args.device)
        targs = targs.to(args.device)
        preds = model(inputs)
        task.test_set.metrics_count(preds, targs)

    metrics = task.test_set.metrics_report(reset=True)
    log.info('Pass #%i evaluate: %s' % (n_passes, str(metrics)))
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalars('evaluate', dict(metrics), n_passes)