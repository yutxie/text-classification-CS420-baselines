import os
import time
import argparse
import logging as log

import torch

import models
import dataset

from train import train
from evaluate import evaluate
from seq2vec import seq2vec

parser = argparse.ArgumentParser(description='Text Classification')

# environment
parser.add_argument('--device',         type=int,   default=3,      help='gpu device id, -1 if cpu')
# log
parser.add_argument('--log_every',      type=int,   default=200,     help='log train how many every passes')
parser.add_argument('--eval_every',     type=int,   default=2000,     help='evaluate how many every passes')
# parser.add_argument('--save_every',     type=int,   default=1000,   help='save model how many every passes')
# train
parser.add_argument('--n_epochs',       type=int,   default=1000,    help='how many epochs')
parser.add_argument('--batch_size',     type=int,   default=32,     help='how many instances in a batch')
parser.add_argument('--lr',             type=float, default=1e-4,   help='learning rate')
# data
parser.add_argument('--data_dir',       type=str,   default='data/')
parser.add_argument('--run_dir',        type=str,   default='run/')
# model
parser.add_argument('--model',          type=str,   default='Seq2Seq')
parser.add_argument('--d_feature',      type=int,   default=300)
parser.add_argument('--d_hidden',       type=int,   default=300)
parser.add_argument('--n_layers',       type=int,   default=1)
parser.add_argument('--dropout',        type=float, default=.5)

args = parser.parse_args()

# make dirs
os.makedirs(args.run_dir, exist_ok=True)

# set log
log.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%m/%d %I:%M:%S %p', level=log.INFO)
log.getLogger().addHandler(log.FileHandler(os.path.join(args.run_dir, 'log.txt'), mode='w'))
log.info(str(vars(args)))

# parse device
args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
args.device = torch.device(args.device)

#########################################

if __name__ == "__main__":

    # dataset
    Task = dataset.NonSeqTask \
        if args.model == 'MLP' \
        else dataset.SeqTask
    task = Task(args)

    # model
    Model = getattr(models, args.model)
    model = Model(args, task)
    model.load_state_dict(torch.load(os.path.join(args.run_dir, 'params_lang.model')))

    # train
    train(args, model, task)
    # preds = evaluate(args, model, task)
    # preds = torch.cat(preds)
    # preds = preds[:,1]

    seq2vec(args, model, task)

    # with open(os.path.join(args.run_dir, 'submission.csv'), 'w') as f:
    #     f.write('id,pred\n')
    #     for idx in range(preds.shape[0]):
    #         f.write('%i,%f\n' % (idx, preds[idx]))