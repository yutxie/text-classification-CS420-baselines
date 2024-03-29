import logging as log

import torch

from torchtext.data import Iterator, BucketIterator
from torch.utils.data import DataLoader

from metrics import Metrics


def evaluate(args, model, task, tensorboard_writer=None, n_passes=-1):
    model.eval()

    metrics = Metrics()
    data_iter = Iterator(
    # data_loader = DataLoader(
        task.test_set,
        args.batch_size,
        # collate_fn=task.collate_fn,
        device=args.device,
        shuffle=False
    )

    preds_list = []
    for batch in data_iter:
    # for batch in data_loader:
        inputs, targs = batch.text, batch.targ
        # inputs, targs = batch
        # targs = targs.to(args.device)
        if args.model == 'Seq2Seq':
            hidden, pred, loss = model(inputs)
        else:
            preds = model(inputs)
            preds_list.append(preds)
            metrics.count(preds, targs)

    print(inputs[0:10], '\n', pred[0:10])

    # report = metrics.report(reset=True)
    # log.info('Pass #%i evaluate: %s' % (n_passes, str(report)))
    log.info('Pass #%i evaluate: %f' % (n_passes, loss.item()))
    # if tensorboard_writer is not None:
    #     tensorboard_writer.add_scalars('evaluate', dict(report), n_passes)

    return preds_list